"""
战略层 Agent（Queen 节点，3 倍权重）—— LLM 驱动版

参考 参考报告 2.3.2 / 表 2.6：
- 宏观趋势 Agent  → "必须区分相关性与因果性"
- 赛制分析 Agent  → "必须量化不确定性"
- 风险感知 Agent  → "自动降级触发条件"

每个 Agent 通过 LLMAgent 基类调用 lingya。
LLM 不可用时自动 fallback 到原硬编码规则（与上一版本完全一致）。
"""
from __future__ import annotations
import sys
import json
from pathlib import Path
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).parent.parent))
from agents.base import Agent, AgentOutput
from agents.llm_base import LLMAgent


# ============ 通用工具：摘取 Top 球队让 prompt 精简 ============
def _top_teams_brief(teams: Dict[str, dict], n: int = 12) -> List[Dict[str, Any]]:
    """精简版 Top N 队信息，避免 prompt 过长"""
    sorted_t = sorted(teams.items(),
                      key=lambda x: -x[1].get("elo", 0))[:n]
    out = []
    for name, d in sorted_t:
        out.append({
            "team": name,
            "elo": d.get("elo"),
            "fifa_rank": d.get("fifa_rank"),
            "market_implied_pct": round(d.get("market_implied", 0) * 100, 2),
            "xg_diff": round(d.get("xg_for", 0) - d.get("xg_against", 0), 2),
        })
    return out


def _continent(team: str) -> str:
    european = {"Spain", "France", "Germany", "England", "Portugal", "Netherlands",
                "Italy", "Belgium", "Croatia", "Switzerland", "Sweden", "Norway",
                "Austria", "Czechia", "Poland", "Turkiye", "Denmark"}
    south_american = {"Brazil", "Argentina", "Uruguay", "Colombia", "Ecuador",
                      "Paraguay", "Chile", "Peru"}
    if team in european:
        return "Europe"
    if team in south_american:
        return "South America"
    return "Other"


# ============================================================
# 1. 宏观趋势 Agent
# ============================================================
class MacroTrendAgent(LLMAgent):
    """识别大赛历史模式、冠军年龄规律、卫冕魔咒"""

    SYSTEM_PROMPT = """你是世界杯研究机构的"宏观趋势 Agent"（战略层 Queen 节点）。

# 角色与能力边界
- 职责：识别大赛历史模式（冠军年龄规律、卫冕魔咒、东道主效应、大洲规律）。
- 你的输出会被加权（3.0×）后参与最终概率合成。

# 关键合规约束
- **必须区分相关性与因果性**：如"北美杯南美 100% 夺冠"是 N=3 的小样本，不能视为强因果。
- 卫冕魔咒：自 1962 巴西后 15 届无卫冕成功（基础事实，可作为先验）。
- 冠军年龄黄金窗：26.91 岁（基础事实）。
- 不得编造历史事件。证据必须基于公开可查的世界杯历史。
- 输出概率调整量（pp，相对蒙特卡洛基线），不是绝对概率。
"""

    OUTPUT_SCHEMA = {
        "adjustments": {
            "<team_name>": "<float, 单位pp，正数=利好，负数=利空，建议范围 -3 到 +2>"
        },
        "confidence": "<float 0-1>",
        "rationale": "<一句话总结，<=120字>",
        "evidence": ["<证据1>", "<证据2>", "..."]
    }

    def __init__(self):
        super().__init__("MacroTrend", layer="strategic", weight=1.0)

    def build_user_prompt(self, context: Dict[str, Any]) -> str:
        teams_brief = _top_teams_brief(context.get("teams", {}), n=10)
        defending_champ = "Argentina"  # 2022 冠军
        host_continent = "North America (USA/Canada/Mexico)"

        return (
            f"# 任务\n"
            f"分析 2026 年世界杯 Top 10 强队的宏观历史趋势调整。\n\n"
            f"# 关键事实\n"
            f"- 卫冕冠军：{defending_champ}（2022 卡塔尔夺冠）\n"
            f"- 主办大洲：{host_continent}（北美第 4 次办杯，前 3 次均南美夺冠：1970/1986/1994）\n\n"
            f"# 候选球队\n"
            f"{json.dumps(teams_brief, ensure_ascii=False, indent=2)}\n\n"
            f"# 任务要求\n"
            f"对每支球队给出宏观趋势调整量（pp）。重点考虑：\n"
            f"1. 卫冕魔咒（如适用）\n"
            f"2. 大洲赛场效应（北美举办+南美球队是否仍享加成？小样本要折扣）\n"
            f"3. 冠军年龄/周期规律\n\n"
            f"按 OUTPUT_SCHEMA 输出 JSON。"
        )

    def parse_response(self, data: Dict[str, Any], context: Dict[str, Any]) -> AgentOutput:
        adjustments = {}
        for team, val in (data.get("adjustments") or {}).items():
            try:
                adjustments[team] = float(val)
            except Exception:
                continue
        return AgentOutput(
            agent_name=self.name, layer=self.layer, role="历史宏观趋势分析（LLM）",
            probability_dist=adjustments,
            confidence=float(data.get("confidence", 0.6)),
            weight=self.weight,
            rationale=str(data.get("rationale", ""))[:300],
            evidence=[str(e)[:200] for e in (data.get("evidence") or [])][:8],
        )

    # ============ Fallback：完全等价的旧规则 ============
    def fallback_analyze(self, context: Dict[str, Any]) -> AgentOutput:
        teams = context.get("teams", {})
        defending_champion = "Argentina"
        defending_penalty = -2.0
        sorted_teams = sorted(teams.items(), key=lambda x: -x[1].get("elo", 0))[:8]
        adjustments = {}
        evidence = []
        for name, _ in sorted_teams:
            adj = 0.0
            if name == defending_champion:
                adj += defending_penalty
                evidence.append(f"{name}: 卫冕魔咒 -2.0pp")
            if _continent(name) == "South America":
                adj += 1.0
                evidence.append(f"{name}: 美洲赛场加成 +1.0pp")
            adjustments[name] = adj
        return AgentOutput(
            agent_name=self.name, layer=self.layer, role="历史宏观趋势分析（规则）",
            probability_dist=adjustments, confidence=0.65, weight=self.weight,
            rationale="规则版：卫冕魔咒 -2pp + 北美举办时南美 +1pp",
            evidence=evidence,
        )


# ============================================================
# 2. 赛制分析 Agent
# ============================================================
class FormatAnalysisAgent(LLMAgent):
    """48 队制赛制特殊性分析"""

    SYSTEM_PROMPT = """你是世界杯研究机构的"赛制分析 Agent"（战略层 Queen 节点）。

# 角色与能力边界
- 职责：分析 2026 首次启用 48 队制对各档次球队的影响。
- 这是首届新赛制，**没有历史回归数据**。

# 核心事实
- 48 队 → 12 组（每组 4 队）→ 32 进 → 16 → 8 → 4 → 决赛
- 8 个最佳第 3 名晋级 32 强（与之前 32 队制 16 强不同）
- 冠军需赢 8 场（旧制 7 场），多出 1/16 决赛
- 比赛日总数从 28 → 39，赛程更长

# 关键合规约束
- **必须量化不确定性**：因无历史样本，confidence 应 ≤ 0.6。
- 不得断言"某队必受冲击"，只给概率倾向。
- 调整量幅度建议在 [-1, +1] pp 内。
"""

    OUTPUT_SCHEMA = {
        "adjustments": {
            "<team_name>": "<float, pp, 推荐 [-1.0, +1.0]>"
        },
        "tier_logic": {
            "elite": "<对 Elo>2050 的球队的影响判断>",
            "upper_mid": "<对 Elo 1900-2050 的判断>",
            "mid": "<对 Elo 1750-1900 的判断>"
        },
        "confidence": "<float 0-1, 因无历史样本应 <= 0.6>",
        "rationale": "<一句话总结>",
        "evidence": ["<事实1>", "<事实2>"]
    }

    def __init__(self):
        super().__init__("FormatAnalysis", layer="strategic", weight=1.0)

    def build_user_prompt(self, context: Dict[str, Any]) -> str:
        teams_brief = _top_teams_brief(context.get("teams", {}), n=16)
        return (
            f"# 任务\n"
            f"评估 48 队制对 Top 16 强队夺冠概率的差异化影响。\n\n"
            f"# 候选球队（含 Elo 分档）\n"
            f"{json.dumps(teams_brief, ensure_ascii=False, indent=2)}\n\n"
            f"# 任务要求\n"
            f"按 [档次差异化] 思路给每队 pp 调整。重点关注：\n"
            f"- 阵容深度差的中上游球队（多打 1 场，疲劳累积）\n"
            f"- 中游球队是否受益于 [最佳第 3 名] 机制\n"
            f"- 强队是否真不受影响（注意：英格兰 2018 多踢 1 场点球大战导致后程衰减）\n\n"
            f"按 OUTPUT_SCHEMA 输出 JSON，confidence 不要超过 0.6。"
        )

    def parse_response(self, data: Dict[str, Any], context: Dict[str, Any]) -> AgentOutput:
        adjustments = {}
        for team, val in (data.get("adjustments") or {}).items():
            try:
                adjustments[team] = float(val)
            except Exception:
                continue
        tier = data.get("tier_logic") or {}
        evidence = [str(e)[:200] for e in (data.get("evidence") or [])][:6]
        if tier:
            evidence.append(f"档次逻辑: {json.dumps(tier, ensure_ascii=False)[:240]}")
        return AgentOutput(
            agent_name=self.name, layer=self.layer, role="48 队制冲击（LLM）",
            probability_dist=adjustments,
            confidence=min(0.6, float(data.get("confidence", 0.5))),
            weight=self.weight,
            rationale=str(data.get("rationale", ""))[:300],
            evidence=evidence,
        )

    def fallback_analyze(self, context: Dict[str, Any]) -> AgentOutput:
        teams = context.get("teams", {})
        adjustments = {}
        for name, data in teams.items():
            elo = data.get("elo", 1500)
            if elo > 2050:
                adj = 0.0
            elif elo > 1900:
                adj = -0.3
            elif elo > 1750:
                adj = +0.2
            else:
                adj = -0.1
            adjustments[name] = adj
        return AgentOutput(
            agent_name=self.name, layer=self.layer, role="48 队制冲击（规则）",
            probability_dist=adjustments, confidence=0.55, weight=self.weight,
            rationale="规则版：Elo 分档差异化（中上游 -0.3，中游 +0.2）",
            evidence=["新增 1/16 决赛", "夺冠总场次 +1", "8 个最佳第 3 名机制",
                      "首届赛制无历史回归数据"],
        )


# ============================================================
# 3. 风险感知 Agent
# ============================================================
class RiskPerceptionAgent(LLMAgent):
    """监控模型漂移、数据异常、黑天鹅事件"""

    SYSTEM_PROMPT = """你是世界杯研究机构的"风险感知 Agent"（战略层 Queen 节点 + 仲裁角色）。

# 角色与能力边界
- 职责：识别可能导致夺冠概率下行的风险信号（伤病潮、模型漂移、数据异常）。
- 同时担任 Agent 间辩论的"仲裁方"，确保保守原则。

# 关键合规约束
- **自动降级触发条件**：
  - 关键球员伤病 health_adj_pp ≤ -3 → 风险信号
  - 市场隐含 vs Elo 隐含分歧 > 4pp → 模型分歧风险
- 风险信号只下调，不上调（防止过度乐观）。
- 调整量幅度 ≤ 实际伤病影响的 50%（其余已被市场定价）。
"""

    OUTPUT_SCHEMA = {
        "adjustments": {
            "<team_name>": "<float, pp, 仅负数或0>"
        },
        "risk_signals": [
            {"team": "<name>", "type": "<injury/model_divergence/data_anomaly>",
             "severity": "<low/mid/high>", "note": "<说明>"}
        ],
        "confidence": "<float 0-1>",
        "rationale": "<一句话总结>",
        "evidence": ["<证据>"]
    }

    def __init__(self):
        super().__init__("RiskPerception", layer="strategic", weight=1.0)

    def build_user_prompt(self, context: Dict[str, Any]) -> str:
        teams = context.get("teams", {})
        injuries = context.get("injuries", {})

        # 只把有显著健康问题的队列出来，节省 prompt
        injury_brief = []
        for team, info in injuries.items():
            adj = info.get("total_health_adj_pp", 0)
            if adj <= -1.0:
                injury_brief.append({
                    "team": team,
                    "health_adj_pp": adj,
                    "n_injuries": len(info.get("injuries", [])),
                    "key_players": [i.get("player") for i in info.get("injuries", [])[:3]],
                })

        # 顶级队的市场 vs Elo 分歧
        divergence = []
        sorted_t = sorted(teams.items(), key=lambda x: -x[1].get("elo", 0))[:8]
        for name, d in sorted_t:
            mkt = d.get("market_implied", 0) * 100
            # 简易 Elo 锚点：Elo 排名前 1 → 18%, 前 2 → 13% ...（占位，真实应读 mc）
            divergence.append({
                "team": name,
                "market_pct": round(mkt, 2),
                "elo": d.get("elo"),
            })

        return (
            f"# 任务\n"
            f"识别下行风险并给出概率调整（仅下调）。\n\n"
            f"# 显著伤病数据\n"
            f"{json.dumps(injury_brief, ensure_ascii=False, indent=2)}\n\n"
            f"# Top 8 市场 vs Elo 概览（用于检测分歧）\n"
            f"{json.dumps(divergence, ensure_ascii=False, indent=2)}\n\n"
            f"# 任务要求\n"
            f"1. 对 health_adj_pp ≤ -3 的队给二次下调（不要超过其原值的 50%）\n"
            f"2. 标注模型分歧风险（不一定下调，但需 risk_signals 记录）\n"
            f"3. 严禁上调任何球队\n\n"
            f"按 OUTPUT_SCHEMA 输出 JSON。"
        )

    def parse_response(self, data: Dict[str, Any], context: Dict[str, Any]) -> AgentOutput:
        adjustments = {}
        for team, val in (data.get("adjustments") or {}).items():
            try:
                v = float(val)
                # 强制非正
                adjustments[team] = min(0.0, v)
            except Exception:
                continue
        signals = data.get("risk_signals") or []
        evidence = [str(e)[:200] for e in (data.get("evidence") or [])][:6]
        for s in signals[:5]:
            evidence.append(
                f"[{s.get('severity', '?')}] {s.get('team', '?')}: "
                f"{s.get('type', '?')} - {s.get('note', '')[:120]}"
            )
        return AgentOutput(
            agent_name=self.name, layer=self.layer, role="风险信号识别（LLM）",
            probability_dist=adjustments,
            confidence=float(data.get("confidence", 0.7)),
            weight=self.weight,
            rationale=str(data.get("rationale", ""))[:300],
            evidence=evidence if evidence else ["无显著风险信号"],
        )

    def fallback_analyze(self, context: Dict[str, Any]) -> AgentOutput:
        teams = context.get("teams", {})
        injuries = context.get("injuries", {})
        adjustments = {}
        evidence = []
        for name, _ in teams.items():
            adj = 0.0
            team_inj = injuries.get(name, {})
            health_adj = team_inj.get("total_health_adj_pp", 0)
            if health_adj < -3:
                adj += health_adj * 0.5
                evidence.append(f"{name}: 严重伤病潮 ({health_adj:+.1f}pp)")
            adjustments[name] = adj
        return AgentOutput(
            agent_name=self.name, layer=self.layer, role="风险信号识别（规则）",
            probability_dist=adjustments, confidence=0.7, weight=self.weight,
            rationale="规则版：health_adj_pp < -3 时叠加 50% 风险折扣",
            evidence=evidence if evidence else ["无显著风险信号"],
        )
