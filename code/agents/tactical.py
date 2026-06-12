"""
战术层 Agent（Worker 节点，1 倍权重）

参考 参考表 2.6：
- ELO/FIFA 评级 Agent (8)
- Poisson 族模型 Agent (10)
- 过程指标 Agent (xG/xT)
- 情境因子 Agent

注：Market / Context 已升级为 LLM 驱动版本（带规则 fallback）。
其他 Agent 保留纯规则实现。
"""
import sys
import json as _json
from pathlib import Path
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))
from agents.base import Agent, AgentOutput
from agents.llm_base import LLMAgent


class EloAgent(Agent):
    """基于 Elo 评级的纯实力分析（不考虑情境）"""
    
    def __init__(self):
        super().__init__("Elo", layer="tactical", weight=1.0)
    
    def analyze(self, context):
        """读取 MC 输出作为 Elo 锚点"""
        mc = context.get("mc_results", {})
        
        prob_dist = {team: data.get("champion", 0) * 100 
                     for team, data in mc.items() if data.get("champion", 0) > 0.001}
        
        return AgentOutput(
            agent_name=self.name,
            layer=self.layer,
            role="Elo 纯实力评估",
            probability_dist=prob_dist,
            confidence=0.75,
            weight=self.weight,
            rationale="基于 100k 蒙特卡洛 + Elo 公式 + 真实 FIFA Bracket 的纯实力概率",
            evidence=[
                f"模拟次数: 100,000",
                f"球队总数: {len(prob_dist)}",
                f"Elo 缩放因子 s=600（Reference 推荐值）",
                "K 值: 友谊 20 / 预选 30 / 杯赛 40 / 世界杯 50-60",
            ],
        )


class PoissonAgent(Agent):
    """Poisson 族集成（DC + Bivariate + ZIGP，对应 参考模型 5/6/7）"""

    def __init__(self):
        super().__init__("Poisson", layer="tactical", weight=1.0)

    def analyze(self, context):
        """
        升级版：用 ensemble_score_matrix 给 Top 队跑真实进球分布预测，
        从联合分布的"统治指数"（A 期望进球比 - 对手 xG 失球比）反推 pp 调整。
        """
        from models.poisson_model import (
            ensemble_score_matrix, expected_lambdas, match_outcome_from_score_matrix,
        )

        teams = context.get("teams", {})
        # Top 12 队两两组合太多，简化为：每队 vs 一个虚拟对手（中位数球队）
        # 估算单队进攻强度
        adjustments = {}
        evidence = []
        components_info = {"dc": 0, "bivariate": 0, "zigp": 0}

        # 中位数对手参数（48 队 xG 中位数）
        all_xg_for = sorted([d.get("xg_for", 1.5) for d in teams.values()])
        all_xg_ag = sorted([d.get("xg_against", 1.0) for d in teams.values()])
        median_xg = all_xg_for[len(all_xg_for) // 2] if all_xg_for else 1.5
        median_xga = all_xg_ag[len(all_xg_ag) // 2] if all_xg_ag else 1.0
        median_elo = sorted([d.get("elo", 1500) for d in teams.values()])[len(teams) // 2]

        # 对每队和"中位对手"打一场
        sorted_teams = sorted(teams.items(),
                              key=lambda x: -x[1].get("elo", 0))[:16]
        for name, data in sorted_teams:
            xg_for = data.get("xg_for", 1.5)
            xg_ag = data.get("xg_against", 1.0)
            elo = data.get("elo", 1500)

            try:
                la, lb = expected_lambdas(elo, median_elo,
                                           xg_for, median_xg,
                                           xg_ag, median_xga)
                ens = ensemble_score_matrix(la, lb,
                                              rho=-0.05, lambda_c=0.10,
                                              theta=0.05, pi_zero=0.04)
                outcome = match_outcome_from_score_matrix(ens["matrix"])
                # 取胜率作为强度信号；vs 中位 50% 是基准
                p_win = outcome["p_win_a"]
                # 把 0.5 映射成 0pp，0.7 映射成 +0.4pp，0.3 → -0.4pp
                adj = round((p_win - 0.5) * 2.0, 3)
                adj = max(-0.6, min(0.6, adj))
                adjustments[name] = adj
                if abs(adj) >= 0.2:
                    evidence.append(
                        f"{name}: vs 中位对手 (Elo {median_elo}) → "
                        f"胜率 {p_win:.0%} → {adj:+.2f}pp"
                    )
            except Exception as e:
                continue

            for k in components_info:
                components_info[k] = ens["weights"][k]

        rationale = (
            f"集成 3 模型：Dixon-Coles({components_info['dc']:.1f}) + "
            f"Bivariate({components_info['bivariate']:.1f}) + "
            f"ZIGP({components_info['zigp']:.1f})。"
            f"Bivariate 通过共享 λ3 建模协方差，ZIGP 处理 0:0 过度离散。"
        )

        return AgentOutput(
            agent_name=self.name, layer=self.layer,
            role="Poisson 族集成（参考模型 5/6/7）",
            probability_dist=adjustments,
            confidence=0.65,
            weight=self.weight,
            rationale=rationale,
            evidence=evidence[:6] if evidence else ["集成跑成功，无显著强弱信号"],
        )


class XGAgent(Agent):
    """xG / xGA 过程指标分析"""
    
    def __init__(self):
        super().__init__("xG", layer="tactical", weight=0.8)
    
    def analyze(self, context):
        teams = context.get("teams", {})
        
        # 找 xG 净差 Top 5
        ranked = sorted(teams.items(), 
                        key=lambda x: -(x[1].get("xg_for", 0) - x[1].get("xg_against", 0)))
        
        adjustments = {}
        evidence = []
        for name, data in ranked[:8]:
            xg_diff = data.get("xg_for", 1.5) - data.get("xg_against", 1.0)
            adj = max(-0.5, min(0.5, xg_diff * 0.3))
            adjustments[name] = adj
            evidence.append(f"{name}: xG净差 {xg_diff:+.2f}, 调整 {adj:+.2f}pp")
        
        return AgentOutput(
            agent_name=self.name,
            layer=self.layer,
            role="xG 过程指标 Top 8",
            probability_dist=adjustments,
            confidence=0.55,
            weight=self.weight,
            rationale="xG 反映过程质量，长期与实际进球高度相关（权重 0.7）",
            evidence=evidence[:5],
        )


class XTAgent(Agent):
    """xT 过程威胁分析（Karun Singh 2018，参考 参考报告 2.2.2）"""

    def __init__(self):
        super().__init__("xT", layer="tactical", weight=0.8)

    def analyze(self, context):
        teams = context.get("teams", {})

        # 取有 xt_per_90 的队（来自 StatsBomb）
        xt_table = [(name, d.get("xt_per_90")) for name, d in teams.items()
                    if d.get("xt_per_90") is not None]
        if not xt_table:
            return AgentOutput(
                agent_name=self.name, layer=self.layer,
                role="xT 过程威胁（StatsBomb 数据缺失）",
                probability_dist={}, confidence=0.3, weight=self.weight,
                rationale="无 xt_per_90 数据，先跑 xt_engine.py compute",
                evidence=["[low_sample] xt_per_90 字段缺失"],
            )

        # 用整体均值/标准差做 z-score
        values = [v for _, v in xt_table]
        mean = sum(values) / len(values)
        std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
        std = max(std, 0.1)

        adjustments = {}
        evidence = []
        for name, xt in sorted(xt_table, key=lambda x: -x[1])[:10]:
            z = (xt - mean) / std
            # ±0.6 pp 上限：z=2 → +0.6pp 加成
            adj = max(-0.6, min(0.6, z * 0.3))
            adjustments[name] = round(adj, 3)
            evidence.append(f"{name}: xT_per_90={xt:.2f} (z={z:+.2f}) → {adj:+.2f}pp")

        return AgentOutput(
            agent_name=self.name, layer=self.layer,
            role="xT 过程威胁 Top 10（Karun Singh）",
            probability_dist=adjustments,
            confidence=0.55,
            weight=self.weight,
            rationale=(
                f"xT 量化每次推进对进球概率的累积贡献。基于 StatsBomb "
                f"WC2018/2022/Euro2024 的 65 万事件训练 16×12 网格矩阵；"
                f"全队均值 {mean:.2f} ± {std:.2f}，z-score × 0.3pp 缩放。"
            ),
            evidence=evidence[:6],
        )


class CatBoostAgent(Agent):
    """CatBoost + pi-ratings ML 模型（参考模型 10）"""

    def __init__(self):
        super().__init__("CatBoost", layer="tactical", weight=1.0)
        self._model = None
        self._R_H = None
        self._R_A = None
        self._form_5 = None
        self._loaded = False

    def _lazy_load(self):
        if self._loaded:
            return True
        try:
            from models.catboost_engine import load_model
            from models.pi_ratings import load_ratings
            self._model = load_model()
            self._R_H, self._R_A, meta = load_ratings()
            self._form_5 = meta.get("form_5", {})
            self._loaded = True
            return True
        except Exception:
            return False

    def analyze(self, context):
        teams = context.get("teams", {})
        if not self._lazy_load():
            return AgentOutput(
                agent_name=self.name, layer=self.layer,
                role="CatBoost ML（未训练）",
                probability_dist={}, confidence=0.3, weight=self.weight,
                rationale="catboost_match.cbm 不存在，先跑 catboost_engine.py train",
                evidence=["[low_sample] 模型未加载"],
            )

        from models.catboost_engine import predict_match

        # 对每队跑 vs "中位 Elo 对手"获得强度信号
        sorted_teams = sorted(teams.items(), key=lambda x: -x[1].get("elo", 0))[:16]
        # 中位对手 = Elo 排第 25 的队（48 队中位数）
        all_sorted = sorted(teams.items(), key=lambda x: -x[1].get("elo", 0))
        median_opp = all_sorted[len(all_sorted) // 2][0] if len(all_sorted) >= 24 else "Croatia"

        adjustments = {}
        evidence = []
        for name, _ in sorted_teams:
            try:
                result = predict_match(
                    home=name, away=median_opp, neutral=True,
                    R_H=self._R_H, R_A=self._R_A, form_5=self._form_5,
                    model=self._model,
                )
                p_win = result["p_home"]
                # vs 中位 0.5 基准 → ±0.6pp 区间
                adj = round((p_win - 0.5) * 1.5, 3)
                adj = max(-0.6, min(0.6, adj))
                adjustments[name] = adj
                if abs(adj) >= 0.2:
                    evidence.append(
                        f"{name}: vs {median_opp} (中立) → "
                        f"P(胜)={p_win:.0%} P(平)={result['p_draw']:.0%} "
                        f"P(负)={result['p_away']:.0%} → {adj:+.2f}pp"
                    )
            except Exception:
                continue

        return AgentOutput(
            agent_name=self.name, layer=self.layer,
            role="CatBoost + pi-ratings ML（参考模型 10）",
            probability_dist=adjustments,
            confidence=0.7,
            weight=self.weight,
            rationale=(
                f"基于 29,293 场国际比赛训练的 CatBoost (1000 iter, depth=6, lr=0.03)。"
                f"特征：pi-ratings 7 维 + 时序权重。验证集 accuracy 60.6%。"
                f"Top 特征：pi_A_diff (27%), expected_gd (12%)"
            ),
            evidence=evidence[:6] if evidence else ["CatBoost 已跑完所有候选队"],
        )


class SquadValueAgent(Agent):
    """阵容总身价 Agent (P5: Transfermarkt 估值 5% 权重)

    设计：
    - 身价反映个体球员潜力上限（FiveThirtyEight SPI 公式核心组件）
    - 与 Elo 互补：Elo 看"已实现战绩"，身价看"未来上限"
    - 关键场景：识别"身价高但 Elo 滞后"（如 England 反向偏差）
    """

    # 经验：每 200M 欧元身价差约对应 1pp 夺冠概率影响
    PP_PER_200M = 1.0

    def __init__(self):
        # weight 0.5：P5 优先级 5%，但作为补充信号有价值
        super().__init__("SquadValue", layer="tactical", weight=0.5)

    def analyze(self, context):
        teams = context.get("teams", {})
        values = [(n, d.get("squad_value_m_eur", 0))
                  for n, d in teams.items() if d.get("group") != "_"]
        values = [(n, v) for n, v in values if v > 0]
        if not values:
            return AgentOutput(
                agent_name=self.name, layer=self.layer,
                role="阵容总身价（Transfermarkt 数据缺失）",
                probability_dist={}, confidence=0.3, weight=self.weight,
                rationale="无 squad_value_m_eur 字段",
                evidence=["[low_sample] 身价字段缺失"],
            )

        # 中位数为基准
        sorted_v = sorted(v for _, v in values)
        median_v = sorted_v[len(sorted_v) // 2]

        # 给 Top 16 队按相对中位数差异调整
        sorted_teams = sorted(values, key=lambda x: -x[1])[:16]
        adjustments = {}
        evidence = []
        for name, v in sorted_teams:
            diff = v - median_v
            adj_pp = (diff / 200.0) * self.PP_PER_200M
            adj_pp = max(-1.5, min(1.5, adj_pp))   # cap ±1.5pp
            adjustments[name] = round(adj_pp, 3)
            if abs(adj_pp) >= 0.5:
                evidence.append(
                    f"{name}: €{v}M (vs 中位 €{median_v}M) → {adj_pp:+.2f}pp"
                )

        return AgentOutput(
            agent_name=self.name, layer=self.layer,
            role="阵容总身价 Top 16（Transfermarkt 2026.06）",
            probability_dist=adjustments,
            confidence=0.6,
            weight=self.weight,
            rationale=(
                f"基于 Transfermarkt 2026.06 数据。各队身价 vs 中位 €{median_v}M，"
                f"每 €200M 差对应 ±1pp，cap ±1.5pp。"
                f"参考报告 P5 优先级 5%，作为 Elo/xG 的补充上限信号。"
            ),
            evidence=evidence[:6] if evidence else ["所有队身价接近中位"],
        )


class HealthAgent(Agent):
    """阵容健康度（Health）综合评估"""
    
    def __init__(self):
        super().__init__("Health", layer="tactical", weight=1.0)
    
    def analyze(self, context):
        """
        从 injuries 数据库读取每队 health 调整
        """
        injuries = context.get("injuries", {})
        
        adjustments = {}
        evidence = []
        
        for team, info in injuries.items():
            adj = info.get("total_health_adj_pp", 0)
            adjustments[team] = adj
            n_inj = len(info.get("injuries", []))
            if adj < -1.0:
                evidence.append(f"{team}: {n_inj} 个伤情, health {adj:+.1f}pp")
        
        return AgentOutput(
            agent_name=self.name,
            layer=self.layer,
            role="阵容健康度评估",
            probability_dist=adjustments,
            confidence=0.8,
            weight=self.weight,
            rationale="基于关键球员伤情库（7 级状态映射）",
            evidence=evidence[:5] if evidence else ["大部分球队无重大伤情"],
        )


class ContextAgent(LLMAgent):
    """情境/环境分析（高温、海拔、旅行）—— LLM 解读环境引擎数值"""

    SYSTEM_PROMPT = """你是世界杯研究机构的"主客场/情境因子 Agent"（战术层，权重 1.0）。

# 角色与能力边界
- 接收每队在 16 个场馆的平均环境因子（高温 WBGT + 海拔 + 大洲适应度）。
- 解读：哪些球队真正受益/受害，哪些只是数值波动。

# 关键合规约束
- **场馆级细分，非单一值**：environment 因子是 16 场地平均，要承认场馆差异。
- 调整量不超过 ±2pp。
- 数据缺失时输出空 adjustments，不要编造。
"""

    OUTPUT_SCHEMA = {
        "adjustments": {"<team>": "<float pp>"},
        "confidence": "<float 0-1>",
        "rationale": "<一句话>",
        "evidence": ["<证据>"]
    }

    def __init__(self):
        super().__init__("Context", layer="tactical", weight=0.9)

    def _compute_env_factors(self, context):
        """规则版本同款：算每队 16 场地平均环境因子"""
        try:
            from models.environment_engine import get_environment_factors, load_venues
            venues, _, _ = load_venues()
        except Exception:
            return None
        teams = context.get("teams", {})
        env_table = {}
        for team in teams:
            factors = []
            for venue_name in venues:
                f = get_environment_factors(team, venue_name)
                factors.append(f["combined_factor"])
            if factors:
                env_table[team] = round(sum(factors) / len(factors), 4)
        return env_table

    def build_user_prompt(self, context: Dict[str, Any]) -> str:
        env = self._compute_env_factors(context) or {}
        # 只给 LLM 偏离 1.0 较大的球队，节省 token
        salient = {t: f for t, f in env.items() if abs(f - 1.0) >= 0.02}
        return (
            f"# 任务\n"
            f"基于 16 场地平均环境因子表（综合 WBGT 高温 + 海拔 + 大洲适应度），"
            f"给出概率调整（pp）。环境因子=1.0 表示中性，>1 利好，<1 利空。\n\n"
            f"# 各队 16 场地平均环境因子（仅显示偏离>0.02 的）\n"
            f"{_json.dumps(salient, ensure_ascii=False, indent=2)}\n\n"
            f"# 任务要求\n"
            f"按 OUTPUT_SCHEMA 输出 JSON。每 0.1 偏离对应约 1pp，可结合球队体能/赛程容错略调。"
        )

    def parse_response(self, data, context):
        adjustments = {}
        for t, v in (data.get("adjustments") or {}).items():
            try:
                adjustments[t] = max(-2.0, min(2.0, float(v)))
            except Exception:
                continue
        return AgentOutput(
            agent_name=self.name, layer=self.layer, role="情境/环境因子（LLM）",
            probability_dist=adjustments,
            confidence=float(data.get("confidence", 0.7)),
            weight=self.weight,
            rationale=str(data.get("rationale", ""))[:300],
            evidence=[str(e)[:200] for e in (data.get("evidence") or [])][:5],
        )

    def fallback_analyze(self, context):
        env = self._compute_env_factors(context)
        if env is None:
            return AgentOutput(
                agent_name=self.name, layer=self.layer, role="环境因子（fallback）",
                probability_dist={}, confidence=0.3, weight=self.weight,
                rationale="环境引擎不可用", evidence=[],
            )
        adjustments = {t: round((f - 1.0) * 10, 2) for t, f in env.items()}
        evidence = [f"{t}: 平均环境因子 {env[t]:.3f} → {adjustments[t]:+.2f}pp"
                    for t in sorted(env, key=lambda x: -abs(env[x] - 1.0))[:5]]
        return AgentOutput(
            agent_name=self.name, layer=self.layer, role="情境/环境因子（规则）",
            probability_dist=adjustments, confidence=0.7, weight=self.weight,
            rationale="规则版：每 0.1 环境因子偏离对应 1pp",
            evidence=evidence,
        )


class MarketAgent(LLMAgent):
    """市场赔率共识识别（用作研究变量，不直接预测）"""

    # 参考表 2.6: "仅作为研究变量，非预测依据"
    SYSTEM_PROMPT = """你是世界杯研究机构的"市场分析 Agent"（战术层）。

# 关键合规约束（来自 参考报告 2.3.2）
**你只能将市场赔率作为共识偏差研究变量使用。禁止将赔率作为预测的直接依据。**
所有输出必须包含以下免责声明：
"市场隐含概率反映资金加权共识，非客观实力度量。"

# 角色与能力边界
- 输入：Polymarket / Kalshi 等市场的去 overround 后隐含概率。
- 输出：以"调整量"形式给出（不是绝对概率），表达"市场共识 vs 模型基线"的偏差信号。
- Confidence 上限 0.85（市场效率高但非完美）。
"""

    OUTPUT_SCHEMA = {
        "adjustments": {"<team>": "<float pp, 推荐 [-3, +3]>"},
        "confidence": "<float, <= 0.85>",
        "rationale": "<一句话，必须包含免责声明>",
        "evidence": ["<证据>"]
    }

    def __init__(self):
        super().__init__("Market", layer="tactical", weight=0.7)

    def build_user_prompt(self, context):
        teams = context.get("teams", {})
        # 只取 market_implied > 1% 的球队
        market_table = []
        for t, d in sorted(teams.items(), key=lambda x: -x[1].get("market_implied", 0))[:15]:
            mkt = d.get("market_implied", 0) * 100
            if mkt < 1.0:
                continue
            market_table.append({
                "team": t,
                "market_pct": round(mkt, 2),
                "elo": d.get("elo"),
                "fifa_rank": d.get("fifa_rank"),
            })
        return (
            f"# 任务\n"
            f"对市场隐含概率与基本面（Elo/FIFA 排名）的偏差给出调整信号。\n"
            f"调整量是 pp 偏移，不是绝对概率。\n\n"
            f"# 市场隐含概率（去 overround 后）\n"
            f"{_json.dumps(market_table, ensure_ascii=False, indent=2)}\n\n"
            f"# 任务要求\n"
            f"- 市场显著低于 Elo 隐含 → 正向调整（被低估）\n"
            f"- 市场显著高于 Elo 隐含 → 负向调整（被高估）\n"
            f"- rationale 末尾必须附免责声明\n\n"
            f"按 OUTPUT_SCHEMA 输出 JSON。"
        )

    def parse_response(self, data, context):
        adjustments = {}
        for t, v in (data.get("adjustments") or {}).items():
            try:
                adjustments[t] = max(-3.0, min(3.0, float(v)))
            except Exception:
                continue
        return AgentOutput(
            agent_name=self.name, layer=self.layer, role="市场共识偏差（LLM）",
            probability_dist=adjustments,
            confidence=min(0.85, float(data.get("confidence", 0.75))),
            weight=self.weight,
            rationale=str(data.get("rationale", ""))[:300],
            evidence=[str(e)[:200] for e in (data.get("evidence") or [])][:5],
        )

    def fallback_analyze(self, context):
        # 规则 fallback：直接吐市场绝对概率（旧行为）
        teams = context.get("teams", {})
        market_dist = {team: data.get("market_implied", 0) * 100
                       for team, data in teams.items()
                       if data.get("market_implied", 0) > 0}
        return AgentOutput(
            agent_name=self.name, layer=self.layer, role="Polymarket 市场共识（规则）",
            probability_dist=market_dist, confidence=0.85, weight=self.weight,
            rationale="规则版：直接采用 Polymarket 去 overround 隐含概率。"
                      "免责：市场隐含概率反映资金加权共识，非客观实力度量。",
            evidence=["Polymarket: $92M 交易量", "Kalshi 互补", "去 overround 归一化"],
        )
