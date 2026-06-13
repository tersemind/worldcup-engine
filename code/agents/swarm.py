"""
Queen-led Swarm 调度器

参考 参考报告第 2.3 章：
- Hierarchical 三层架构（控制度高、可审计）
- Queen 节点（战略层）做任务委派 + 共识聚合
- Worker 节点（战术 + 执行）独立分析
- Byzantine 容错（2/3 多数 + 加权共识）
"""
import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))
from agents.base import Agent, AgentOutput, ConsensusAggregator, Critic
from agents.llm_base import LLMAgent
from agents.strategic import MacroTrendAgent, FormatAnalysisAgent, RiskPerceptionAgent
from agents.tactical import EloAgent, PoissonAgent, XGAgent, XTAgent, CatBoostAgent, SquadValueAgent, HealthAgent, ContextAgent, MarketAgent
from agents.execution import OptimistAgent, PessimistAgent, PathwayAgent, PsychologyAgent, MarketBiasAgent
from agents.debate import DebateEngine, batch_debate, summarize
from utils.io import load_teams, DATA_OUTPUTS, DATA_RAW


class LLMQuorumError(RuntimeError):
    """LLM Agent 可用率低于阈值时抛出（fail-fast）"""
    pass


class QueenSwarm:
    """Hierarchical Queen-led Swarm 协调器"""
    
    def __init__(self):
        # 三层 Agent 注册
        self.strategic_agents = [
            MacroTrendAgent(),
            FormatAnalysisAgent(),
            RiskPerceptionAgent(),
        ]
        self.tactical_agents = [
            EloAgent(),
            PoissonAgent(),
            XGAgent(),
            XTAgent(),
            CatBoostAgent(),    # 参考模型 10：CatBoost + pi-ratings
            SquadValueAgent(),  # P5: Transfermarkt 身价（5% 权重）
            HealthAgent(),
            ContextAgent(),
            MarketAgent(),
        ]
        self.execution_agents = [
            OptimistAgent(),
            PessimistAgent(),
            PathwayAgent(),
            PsychologyAgent(),
            MarketBiasAgent(),
        ]
        self.aggregator = ConsensusAggregator()
        self.critic = Critic()
    
    def all_agents(self) -> List[Agent]:
        return self.strategic_agents + self.tactical_agents + self.execution_agents
    
    def build_context(self) -> Dict[str, Any]:
        """构建共享上下文（Queen 给所有 Agent 的输入）"""
        teams = load_teams()
        
        context = {
            "teams": {n: d for n, d in teams.items() if d.get("group") != "_"},
        }
        
        # 蒙特卡洛输出
        mc_path = DATA_OUTPUTS / "mc_simulation_n100000.json"
        if mc_path.exists():
            with open(mc_path) as f:
                context["mc_results"] = json.load(f)
        else:
            context["mc_results"] = {}
        
        # 三情景
        scen_path = DATA_OUTPUTS / "three_scenarios.json"
        if scen_path.exists():
            with open(scen_path) as f:
                three_sc = json.load(f)
                context["scenarios"] = {
                    k: v.get("probs", {}) 
                    for k, v in three_sc.get("scenarios", {}).items()
                }
        else:
            context["scenarios"] = {}
        
        # 伤病库
        inj_path = DATA_RAW / "injuries.json"
        if inj_path.exists():
            with open(inj_path) as f:
                context["injuries"] = json.load(f).get("teams", {})
        else:
            context["injuries"] = {}
        
        # 外部预测
        ext_path = DATA_RAW / "external_predictions.json"
        if ext_path.exists():
            with open(ext_path) as f:
                context["external_predictions"] = json.load(f).get("models", {})
        else:
            context["external_predictions"] = {}

        # 数据可用性四问（Layer 1：参考报告 2.1.2）
        try:
            from data.availability_check import run_all_checks
            context["data_quality"] = run_all_checks(verbose=False)
        except Exception as e:
            context["data_quality"] = {
                "overall_weight": 1.0,
                "warning": f"availability check failed: {e}",
            }

        return context
    
    def run(self, verbose: bool = True) -> Dict[str, Any]:
        """
        执行完整 Swarm 流程：
        1. Queen 构建上下文
        2. 所有 Agent 并行（这里是顺序，但概念上是并行）执行
        3. Queen 聚合共识
        4. Critic 评分
        5. 输出最终概率
        """
        if verbose:
            print("=" * 80)
            print("👑 Queen-led Swarm 启动")
            print("=" * 80)
        
        context = self.build_context()
        
        if verbose:
            print(f"\n📊 上下文：")
            print(f"   球队数: {len(context['teams'])}")
            print(f"   MC 输出: {'✅' if context['mc_results'] else '❌'}")
            print(f"   三情景: {'✅' if context['scenarios'] else '❌'}")
            print(f"   伤病库: {len(context['injuries'])} 队")
            print(f"   外部模型: {len(context['external_predictions'])}")
        
        # 执行所有 Agent
        outputs = []
        
        if verbose:
            print(f"\n🤖 Strategic Layer ({len(self.strategic_agents)} Agents, weight=3.0):")
        for agent in self.strategic_agents:
            t0 = time.time()
            out = agent.analyze(context)
            elapsed = time.time() - t0
            outputs.append(out)
            if verbose:
                n_teams = len(out.probability_dist or {})
                print(f"   ✓ {agent.name:<18} → {n_teams} 队, {elapsed*1000:.1f}ms, conf={out.confidence}")
        
        if verbose:
            print(f"\n🛠️  Tactical Layer ({len(self.tactical_agents)} Agents, weight=1.0):")
        for agent in self.tactical_agents:
            t0 = time.time()
            out = agent.analyze(context)
            elapsed = time.time() - t0
            outputs.append(out)
            if verbose:
                n_teams = len(out.probability_dist or {})
                print(f"   ✓ {agent.name:<18} → {n_teams} 队, {elapsed*1000:.1f}ms, conf={out.confidence}")
        
        if verbose:
            print(f"\n⚙️  Execution Layer ({len(self.execution_agents)} Agents, weight=1.0):")
        for agent in self.execution_agents:
            t0 = time.time()
            out = agent.analyze(context)
            elapsed = time.time() - t0
            outputs.append(out)
            if verbose:
                n_teams = len(out.probability_dist or {})
                print(f"   ✓ {agent.name:<18} → {n_teams} 队, {elapsed*1000:.1f}ms, conf={out.confidence}")
        
        # ====== LLM Quorum 检查（fail-fast 阀门）======
        # 仅统计 LLMAgent 子类；规则 Agent (Elo/Market/Health 等) 不计入
        # 阈值由 WORLDCUP_LLM_REQUIRE 控制：允许 fallback 比例上限
        #   = 0.0 → 禁止任何 fallback（最严）
        #   = 0.5 → 允许最多 50% fallback（默认）
        #   = 1.0 → 等同旧行为（无强制）
        max_fallback_ratio = float(os.getenv("WORLDCUP_LLM_REQUIRE", "0.5"))
        llm_outputs = [o for o, a in zip(outputs, self.all_agents())
                        if isinstance(a, LLMAgent)]
        n_llm_total = len(llm_outputs)
        n_fallback = sum(
            1 for o in llm_outputs
            if not any("[source:LLM]" in e for e in (o.evidence or []))
        )
        fallback_ratio = (n_fallback / n_llm_total) if n_llm_total else 0.0
        llm_health = {
            "total_llm_agents": n_llm_total,
            "fallback_count": n_fallback,
            "fallback_ratio": round(fallback_ratio, 3),
            "threshold": max_fallback_ratio,
            "passed": fallback_ratio <= max_fallback_ratio,
            "fallback_agents": [
                o.agent_name for o in llm_outputs
                if not any("[source:LLM]" in e for e in (o.evidence or []))
            ],
        }
        if verbose:
            status = "✅" if llm_health["passed"] else "❌"
            print(f"\n🔌 LLM Quorum: {status} {n_fallback}/{n_llm_total} fallback "
                  f"(ratio={fallback_ratio:.2f}, threshold={max_fallback_ratio})")
            if n_fallback > 0:
                print(f"   降级 Agent: {llm_health['fallback_agents']}")
        if not llm_health["passed"]:
            raise LLMQuorumError(
                f"LLM Agent 大面积失败：{n_fallback}/{n_llm_total} 走 fallback "
                f"(ratio={fallback_ratio:.2f} > {max_fallback_ratio})。"
                f"降级 Agent: {llm_health['fallback_agents']}。"
                f"请检查 LINGYA_API_KEY 是否设置或 LLM 网关是否可用。"
                f"如需绕过，可设置 WORLDCUP_LLM_REQUIRE=1.0。"
            )

        # Queen 聚合
        if verbose:
            print(f"\n👑 Queen 共识聚合（Byzantine 容错 2/3 多数）...")
        
        consensus = self.aggregator.aggregate_distributions(outputs)
        
        # Critic 评分
        if verbose:
            print(f"\n🛡️  Critic 红队评分...")
        agent_scores = {}
        for out in outputs:
            # 用 Elo Agent 的输出作为基准 consensus（点估计仅评分用）
            if out.point_estimate is None and not out.probability_dist:
                continue
            score = self.critic.score(out)
            agent_scores[out.agent_name] = score
        
        # 整理 Top 队结果
        # 不同 Agent 输出不同：有的是绝对概率（如 Elo），有的是调整量（如 MacroTrend）
        # 这里用 Elo Agent 作为基线，其他 Agent 都视为调整量
        elo_output = next((o for o in outputs if o.agent_name == "Elo"), None)
        market_output = next((o for o in outputs if o.agent_name == "Market"), None)
        
        baseline = {}
        if elo_output and elo_output.probability_dist:
            baseline = dict(elo_output.probability_dist)
        
        # 应用所有"调整量"型 Agent 的输出
        adjustment_agents = ["MacroTrend", "FormatAnalysis", "RiskPerception",
                              "Poisson", "xG", "Health", "Context",
                              "Optimist", "Pessimist", "Pathway",
                              "Psychology", "MarketBias"]
        
        adjusted = dict(baseline)
        for out in outputs:
            if out.agent_name in adjustment_agents and out.probability_dist:
                # 应用 Layer 加权
                layer_w = ConsensusAggregator.LAYER_WEIGHTS.get(out.layer, 1.0)
                effective_w = layer_w * out.weight * out.confidence / 10.0

                for team, adj_pp in out.probability_dist.items():
                    if team in adjusted:
                        adjusted[team] += adj_pp * effective_w

        # ====== Psychology Agent 特殊通道：直接应用 pp 调整 ======
        # Psychology 包含"Elo 滞后校正"信号，必须按 pp 实际值生效，不被缩放消音
        psych_out = next((o for o in outputs if o.agent_name == "Psychology"), None)
        if psych_out and psych_out.probability_dist:
            for team, adj_pp in psych_out.probability_dist.items():
                if team in adjusted and abs(adj_pp) >= 0.5:
                    # 大幅调整（如 -1.5pp Colombia）直接生效
                    adjusted[team] += adj_pp * psych_out.confidence
                    # 防负
                    adjusted[team] = max(0.1, adjusted[team])
        
        # 边界保护
        for team in adjusted:
            adjusted[team] = max(0.01, min(50.0, adjusted[team]))
        
        # 归一化（保证和近似 100）
        total = sum(adjusted.values())
        if total > 0:
            for team in adjusted:
                adjusted[team] = adjusted[team] * 100 / total

        # ====== 辩论协议四级降级（参考报告 2.3.3）======
        if verbose:
            print(f"\n⚖️  辩论协议（四级分歧检测）...")

        # 给 Top 12 球队跑辩论：用所有给出绝对概率（>1pp）的 Agent
        top_teams = sorted(adjusted.items(), key=lambda x: -x[1])[:12]
        team_outputs_for_debate: Dict[str, List[AgentOutput]] = {}
        for team, _ in top_teams:
            relevant = []
            for o in outputs:
                if not o.probability_dist or team not in o.probability_dist:
                    continue
                # 只收集"绝对概率"型输出（>=1pp 才算有意义估计）
                if o.agent_name in ("Elo", "Market", "CatBoost"):
                    val = o.probability_dist[team]
                    if val >= 0.5:  # 排除微小调整
                        relevant.append(o)
            if len(relevant) >= 2:
                team_outputs_for_debate[team] = relevant

        debate_results = batch_debate(team_outputs_for_debate)
        debate_summary = summarize(debate_results)

        # 用辩论结果调整最终概率（仅对真触发分歧的）
        for team, dr in debate_results.items():
            if dr.level >= 2 and team in adjusted:
                # 用辩论 consensus 替换原始 baseline 部分，并应用 confidence 降权
                orig = adjusted[team]
                # 加权混合：辩论 consensus × multiplier + 原结果 × (1-multiplier)
                # multiplier 高 → 辩论意见生效弱；低 → 强烈使用辩论结果（说明分歧大）
                debate_weight = 1.0 - dr.confidence_multiplier  # L2:0.2, L3:0.5, L4:0.8
                adjusted[team] = (
                    dr.consensus_pp * debate_weight + orig * (1 - debate_weight)
                )

        # 重新归一化
        total = sum(adjusted.values())
        if total > 0:
            for team in adjusted:
                adjusted[team] = adjusted[team] * 100 / total

        if verbose:
            print(f"   {debate_summary['n_teams']} 队参与辩论，"
                  f"L1={debate_summary['level_distribution'][1]} "
                  f"L2={debate_summary['level_distribution'][2]} "
                  f"L3={debate_summary['level_distribution'][3]} "
                  f"L4={debate_summary['level_distribution'][4]}")
            for team, dr in sorted(debate_results.items(),
                                     key=lambda kv: -kv[1].level):
                if dr.level >= 2:
                    print(f"   • {team:<14} L{dr.level} spread={dr.spread_pp:.1f}pp "
                          f"→ {dr.consensus_pp:.2f}pp (mult={dr.confidence_multiplier})")
        
        # 输出
        result = {
            "method": "Queen-led Swarm (Hierarchical 3-layer + Byzantine 2/3 + Weighted Consensus)",
            "n_agents": len(outputs),
            "layers": {
                "strategic": len(self.strategic_agents),
                "tactical": len(self.tactical_agents),
                "execution": len(self.execution_agents),
            },
            "swarm_predictions": {team: round(p, 2) for team, p in 
                                    sorted(adjusted.items(), key=lambda x: -x[1])},
            "agent_outputs": [
                {
                    "agent": o.agent_name,
                    "layer": o.layer,
                    "role": o.role,
                    "confidence": o.confidence,
                    "rationale": o.rationale,
                    "evidence_count": len(o.evidence),
                    "n_teams_analyzed": len(o.probability_dist or {}),
                    "score": agent_scores.get(o.agent_name, {}),
                }
                for o in outputs
            ],
            "consensus_for_top_teams": {
                team: cons for team, cons in 
                sorted(consensus.items(), key=lambda x: -(x[1]["consensus"] or 0))[:8]
            },
            "debate_protocol": {
                "summary": debate_summary,
                "team_results": {
                    team: {
                        "level": dr.level,
                        "label": dr.label,
                        "spread_pp": dr.spread_pp,
                        "consensus_pp": dr.consensus_pp,
                        "confidence_multiplier": dr.confidence_multiplier,
                        "raw_estimates": dr.raw_estimates,
                        "arbiter": dr.arbiter_choice,
                        "rounds_played": dr.rounds_played,
                        "discarded": dr.discarded_agents,
                        "rationale": dr.rationale,
                    }
                    for team, dr in debate_results.items()
                },
            },
            "llm_health": llm_health,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        return result


def print_swarm_report(result: Dict[str, Any], top_n: int = 12):
    """打印 Swarm 报告"""
    print("\n" + "=" * 90)
    print(f"🏆 Queen-led Swarm 最终预测（{result['n_agents']} 个 Agent 共识）")
    print("=" * 90)
    
    print(f"\n方法: {result['method']}")
    print(f"层级: 战略 {result['layers']['strategic']} + 战术 {result['layers']['tactical']} + 执行 {result['layers']['execution']}")
    
    print(f"\n📊 Swarm 综合预测 Top {top_n}:")
    print(f"  {'Rank':<5} {'Team':<14} {'Probability':<12}")
    print("  " + "-" * 35)
    for i, (team, prob) in enumerate(list(result["swarm_predictions"].items())[:top_n], 1):
        print(f"  {i:<5} {team:<14} {prob:>5.2f}%")
    
    print(f"\n🤖 Agent 评分 Top 5:")
    sorted_agents = sorted(result["agent_outputs"], 
                            key=lambda x: -x.get("score", {}).get("total", 0))
    for a in sorted_agents[:5]:
        s = a.get("score", {})
        print(f"  {a['agent']:<14} [{a['layer']:<10}] Grade {s.get('grade', '?')} ({s.get('total', 0):.0f}/100) - {a['role']}")
    
    print(f"\n👑 Queen 高级共识（Top 8）:")
    for team, cons in result["consensus_for_top_teams"].items():
        if cons["consensus"] is not None:
            byz = "✓" if cons.get("byzantine_pass") else "✗"
            print(f"  {team:<14} 共识 {cons['consensus']:>6.2f}, std={cons['std']:>5.2f}, "
                  f"agreement {cons.get('agreement_pct', 0):>5.1f}%, Byzantine {byz}")


def main():
    swarm = QueenSwarm()
    result = swarm.run(verbose=True)
    print_swarm_report(result)
    
    # 保存
    out_path = DATA_OUTPUTS / "swarm_consensus.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 已保存: {out_path.name}")


if __name__ == "__main__":
    main()
