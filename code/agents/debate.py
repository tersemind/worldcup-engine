"""
四级辩论协议（参考报告 2.3.3）
====================================

当 Agent 间分歧超过阈值时，触发分级降级流程。
单位：pp（百分点）—— 报告原文示例 "估计 58% vs 52%，分歧度 6%" 即 6pp。

四级动作：
  Level 1  分歧 < 15pp     直接加权平均，置信度不调整
  Level 2  15-30pp          仲裁裁决（RiskPerception Agent），置信度 ×0.8
  Level 3  30-50pp          3 轮辩论模拟，输出标 "high_uncertainty"，置信度 ×0.5
  Level 4  > 50pp           "冲突"，暂停定量，置信度 ×0.2，标 "conflict"

仲裁逻辑：
  - 取 Critic 评分较高的一方观点为基准
  - 弱化或丢弃低评分方
  - 拓宽置信区间
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict

sys.path.insert(0, str(Path(__file__).parent))
from agents.base import AgentOutput, Critic


# ============ 阈值（参考报告 2.3.3 原文）============
THRESHOLDS = {
    "L1_max": 15.0,   # < 15pp
    "L2_max": 30.0,   # 15-30pp
    "L3_max": 50.0,   # 30-50pp
    # > 50pp → L4
}

CONFIDENCE_PENALTY = {
    1: 1.00,
    2: 0.80,
    3: 0.50,
    4: 0.20,
}

LEVEL_LABEL = {
    1: "mild",
    2: "moderate",
    3: "high",
    4: "extreme_conflict",
}


# ============ 数据结构 ============
@dataclass
class DebateResult:
    team: str
    level: int                           # 1-4
    label: str
    raw_estimates: Dict[str, float]      # agent_name → estimate (pp)
    consensus_pp: float
    std_pp: float
    spread_pp: float                     # max-min
    confidence_multiplier: float         # 0.2-1.0
    arbiter_choice: Optional[str] = None # 在 L2/L3 由谁裁决
    rounds_played: int = 0               # L3 实际轮数
    rationale: str = ""
    contributing_agents: List[str] = field(default_factory=list)
    discarded_agents: List[str] = field(default_factory=list)


# ============ 全局开关：是否启用 LLM 辩论 ============
USE_LLM_DEBATE = True   # 设 False 强制走数值（用于测试 / 离线）


# ============ 核心：分歧检测 ============
def detect_level(estimates: Dict[str, float]) -> int:
    """
    按 max-min spread 决定 Level
    """
    if len(estimates) < 2:
        return 1
    values = list(estimates.values())
    spread = max(values) - min(values)
    if spread < THRESHOLDS["L1_max"]:
        return 1
    if spread < THRESHOLDS["L2_max"]:
        return 2
    if spread < THRESHOLDS["L3_max"]:
        return 3
    return 4


# ============ 4 级处理 ============
class DebateEngine:
    """完整四级辩论协议"""

    def __init__(self):
        self.critic = Critic()

    def resolve(self, team: str, outputs: List[AgentOutput]) -> DebateResult:
        """
        给一个球队和涉及它的 Agent 输出列表，返回辩论结果
        
        Args:
            team: 球队名
            outputs: 至少 2 个含该 team 估计的 AgentOutput
        """
        # 抽取该 team 的估计
        estimates: Dict[str, float] = {}
        agent_map: Dict[str, AgentOutput] = {}
        for out in outputs:
            est = None
            if out.point_estimate is not None and not out.probability_dist:
                est = out.point_estimate
            elif out.probability_dist and team in out.probability_dist:
                est = out.probability_dist[team]
            if est is not None:
                estimates[out.agent_name] = est
                agent_map[out.agent_name] = out

        if len(estimates) < 2:
            # 无可辩论 → 直接 L1
            consensus = next(iter(estimates.values())) if estimates else 0.0
            return DebateResult(
                team=team, level=1, label=LEVEL_LABEL[1],
                raw_estimates=estimates, consensus_pp=consensus,
                std_pp=0.0, spread_pp=0.0,
                confidence_multiplier=1.0,
                contributing_agents=list(estimates.keys()),
                rationale="单一 Agent 估计，无辩论",
            )

        level = detect_level(estimates)
        values = list(estimates.values())
        spread = max(values) - min(values)
        mean = sum(values) / len(values)
        std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5

        # 分级处理
        if level == 1:
            return self._level1(team, estimates, agent_map, mean, std, spread)
        elif level == 2:
            return self._level2(team, estimates, agent_map, mean, std, spread)
        elif level == 3:
            return self._level3(team, estimates, agent_map, mean, std, spread)
        else:
            return self._level4(team, estimates, agent_map, mean, std, spread)

    # ============ L1 加权平均 ============
    def _level1(self, team, estimates, agent_map, mean, std, spread):
        # 用每个 Agent 的 confidence × layer_weight 作权重
        total_w, weighted = 0.0, 0.0
        for name, est in estimates.items():
            out = agent_map[name]
            w = out.confidence * out.weight
            weighted += est * w
            total_w += w
        consensus = weighted / total_w if total_w > 0 else mean

        return DebateResult(
            team=team, level=1, label=LEVEL_LABEL[1],
            raw_estimates=estimates, consensus_pp=round(consensus, 3),
            std_pp=round(std, 3), spread_pp=round(spread, 3),
            confidence_multiplier=CONFIDENCE_PENALTY[1],
            contributing_agents=list(estimates.keys()),
            rationale=f"L1 轻度分歧 ({spread:.1f}pp)，加权平均聚合",
        )

    # ============ L2 仲裁裁决 ============
    def _try_llm_debate(self, team, estimates, agent_map, level, spread):
        """尝试 LLM 多轮辩论；失败返回 None"""
        if not USE_LLM_DEBATE:
            return None
        from agents.llm_debate import run_llm_debate
        sorted_est = sorted(estimates.items(), key=lambda kv: kv[1])
        if len(sorted_est) < 2:
            return None
        low_name, low_est = sorted_est[0]
        high_name, high_est = sorted_est[-1]
        pro = agent_map[high_name]
        con = agent_map[low_name]
        return run_llm_debate(team, level, pro, con, spread)

    def _level2(self, team, estimates, agent_map, mean, std, spread):
        """
        优先 LLM 辩论（正方/反方/仲裁各 1 次 = 3 LLM 调用）；
        LLM 失败回退：选 Critic 评分最高的 Agent 观点为基准（70%）。
        """
        # 先尝试 LLM
        llm_result = self._try_llm_debate(team, estimates, agent_map, 2, spread)
        if llm_result:
            return DebateResult(
                team=team, level=2, label=LEVEL_LABEL[2],
                raw_estimates=estimates,
                consensus_pp=round(llm_result["consensus_pp"], 3),
                std_pp=round(std, 3), spread_pp=round(spread, 3),
                confidence_multiplier=CONFIDENCE_PENALTY[2],
                arbiter_choice="LLM_arbiter",
                rounds_played=1,
                contributing_agents=list(estimates.keys()),
                rationale=(
                    f"L2 LLM 辩论（{llm_result['n_llm_calls']} 次 LLM 调用）"
                    f"resolved={llm_result['resolved']} "
                    f"verdict: {llm_result['arbiter_verdict'][:100]}"
                ),
            )

        # === Fallback：原数值仲裁 ===
        # Critic 评分
        scores = {}
        for name, out in agent_map.items():
            sc = self.critic.score(out, consensus=mean)
            scores[name] = sc["total"]

        # 优先 RiskPerception 兼任（参考报告 2.3.3 原文）
        arbiter = None
        if "RiskPerception" in scores:
            arbiter = "RiskPerception"
        else:
            arbiter = max(scores, key=scores.get)

        arbiter_est = estimates[arbiter]
        # 其他 Agent 加权平均
        others = {n: e for n, e in estimates.items() if n != arbiter}
        others_mean = sum(others.values()) / len(others) if others else arbiter_est

        # 70% 仲裁 + 30% 其他
        consensus = 0.7 * arbiter_est + 0.3 * others_mean

        return DebateResult(
            team=team, level=2, label=LEVEL_LABEL[2],
            raw_estimates=estimates, consensus_pp=round(consensus, 3),
            std_pp=round(std, 3), spread_pp=round(spread, 3),
            confidence_multiplier=CONFIDENCE_PENALTY[2],
            arbiter_choice=arbiter,
            contributing_agents=list(estimates.keys()),
            rationale=(
                f"L2 中度分歧 ({spread:.1f}pp)，由 {arbiter} 仲裁 "
                f"(70% 权重)，置信度 ×0.8"
            ),
        )

    # ============ L3 三轮辩论 ============
    def _level3(self, team, estimates, agent_map, mean, std, spread):
        """
        优先 LLM 三轮辩论（最多 9 次 LLM 调用）；
        LLM 失败回退：Critic 评分 + 剔除劣方
        """
        # 先尝试 LLM
        llm_result = self._try_llm_debate(team, estimates, agent_map, 3, spread)
        if llm_result:
            return DebateResult(
                team=team, level=3, label=LEVEL_LABEL[3],
                raw_estimates=estimates,
                consensus_pp=round(llm_result["consensus_pp"], 3),
                std_pp=round(std, 3), spread_pp=round(spread, 3),
                confidence_multiplier=CONFIDENCE_PENALTY[3],
                arbiter_choice="LLM_arbiter",
                rounds_played=len(llm_result["rounds"]) // 3,
                contributing_agents=list(estimates.keys()),
                rationale=(
                    f"L3 LLM 三轮辩论（{llm_result['n_llm_calls']} 次 LLM）"
                    f"resolved={llm_result['resolved']} "
                    f"verdict: {llm_result['arbiter_verdict'][:100]}"
                ),
            )

        # === Fallback：原 Critic 剔除 ===
        rounds_played = 1
        # Round 2: 给所有 Agent 打分
        scores = {
            name: self.critic.score(out, consensus=mean)["total"]
            for name, out in agent_map.items()
        }
        rounds_played = 2

        # Round 3: 剔除评分最低方
        loser = min(scores, key=scores.get)
        discarded = [loser]
        rounds_played = 3
        remaining = {n: e for n, e in estimates.items() if n not in discarded}

        if len(remaining) >= 2:
            new_spread = max(remaining.values()) - min(remaining.values())
            new_mean = sum(remaining.values()) / len(remaining)
            new_agent_map = {n: agent_map[n] for n in remaining}

            # 若剔除后落到 L2 以下，走仲裁
            if new_spread < THRESHOLDS["L2_max"]:
                sub_result = self._level2(team, remaining, new_agent_map,
                                            new_mean, std, new_spread)
            else:
                # 仍 L3 → 加权平均剩余 Agent
                total_w, weighted = 0.0, 0.0
                for name, est in remaining.items():
                    out = new_agent_map[name]
                    w = out.confidence * out.weight
                    weighted += est * w
                    total_w += w
                consensus = weighted / total_w if total_w > 0 else new_mean
                sub_result = DebateResult(
                    team=team, level=3, label=LEVEL_LABEL[3],
                    raw_estimates=estimates, consensus_pp=round(consensus, 3),
                    std_pp=round(std, 3), spread_pp=round(spread, 3),
                    confidence_multiplier=CONFIDENCE_PENALTY[3],
                    rounds_played=rounds_played, discarded_agents=discarded,
                    contributing_agents=list(remaining.keys()),
                    rationale=f"L3 高度分歧 ({spread:.1f}pp)，剔除 {loser} 后仍 L3",
                )
        else:
            # 只剩 1 个 Agent → 取该 Agent 估计
            only = next(iter(remaining))
            sub_result = DebateResult(
                team=team, level=3, label=LEVEL_LABEL[3],
                raw_estimates=estimates, consensus_pp=round(remaining[only], 3),
                std_pp=round(std, 3), spread_pp=round(spread, 3),
                confidence_multiplier=CONFIDENCE_PENALTY[3],
                rounds_played=rounds_played, discarded_agents=discarded,
                contributing_agents=[only],
                rationale=f"L3 剔除 {loser} 后仅剩 {only}",
            )

        # 强制标 L3 + 降权 0.5
        sub_result.level = 3
        sub_result.label = LEVEL_LABEL[3]
        sub_result.confidence_multiplier = CONFIDENCE_PENALTY[3]
        sub_result.rounds_played = rounds_played
        sub_result.discarded_agents = discarded
        if sub_result.arbiter_choice:
            sub_result.rationale = (
                f"L3 高度分歧 ({spread:.1f}pp)，3 轮辩论剔除 {loser}（Critic 评分 "
                f"{scores[loser]:.0f}），剩余压到 L2 由 {sub_result.arbiter_choice} 仲裁，"
                f"置信度 ×0.5"
            )
        return sub_result

    # ============ L4 极端冲突 ============
    def _level4(self, team, estimates, agent_map, mean, std, spread):
        """
        优先 LLM 生成定性冲突报告；失败回退到中位数
        """
        llm_result = self._try_llm_debate(team, estimates, agent_map, 4, spread)
        if llm_result:
            return DebateResult(
                team=team, level=4, label=LEVEL_LABEL[4],
                raw_estimates=estimates,
                consensus_pp=round(llm_result["consensus_pp"], 3),
                std_pp=round(std, 3), spread_pp=round(spread, 3),
                confidence_multiplier=CONFIDENCE_PENALTY[4],
                contributing_agents=list(estimates.keys()),
                rationale=(
                    f"L4 LLM 冲突报告 ({llm_result['n_llm_calls']} 次)，"
                    f"定性判断: {llm_result['arbiter_verdict'][:120]}"
                ),
            )

        # === Fallback：中位数 ===
        sorted_vals = sorted(estimates.values())
        n = len(sorted_vals)
        median = sorted_vals[n // 2] if n % 2 == 1 else (
            (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
        )
        return DebateResult(
            team=team, level=4, label=LEVEL_LABEL[4],
            raw_estimates=estimates, consensus_pp=round(median, 3),
            std_pp=round(std, 3), spread_pp=round(spread, 3),
            confidence_multiplier=CONFIDENCE_PENALTY[4],
            contributing_agents=list(estimates.keys()),
            rationale=(
                f"L4 极端冲突 ({spread:.1f}pp > 50pp)，"
                f"暂停定量，返回中位数 {median:.2f}pp，置信度 ×0.2，"
                f"建议人工审查"
            ),
        )


# ============ 批量辩论 ============
def batch_debate(team_outputs: Dict[str, List[AgentOutput]]) -> Dict[str, DebateResult]:
    """
    对一批球队跑辩论，返回 {team: DebateResult}
    
    Args:
        team_outputs: {team_name: [AgentOutput, ...]}
    """
    eng = DebateEngine()
    return {team: eng.resolve(team, outs) for team, outs in team_outputs.items()}


def summarize(results: Dict[str, DebateResult]) -> Dict[str, Any]:
    """统计四级分布"""
    n_total = len(results)
    levels = {1: 0, 2: 0, 3: 0, 4: 0}
    for r in results.values():
        levels[r.level] += 1
    return {
        "n_teams": n_total,
        "level_distribution": levels,
        "level_pct": {k: round(v / n_total * 100, 1) if n_total else 0
                       for k, v in levels.items()},
        "n_conflicts": levels[4],
        "n_high_uncertainty": levels[3],
        "avg_confidence_multiplier": round(
            sum(r.confidence_multiplier for r in results.values()) / max(1, n_total), 3
        ),
    }
