"""
执行层 Agent（Worker 节点，1 倍权重）

参考 参考表 2.6 + 第 3 章 3.x.5 / 3.x.6：
- Optimist Agent  → 夺冠路径与优势放大条件（4 情景论证）
- Pessimist Agent → 出局风险与劣势触发条件（4 情景论证）
- Pathway Agent   → 半区效应（结构化判定，保留规则）
- Psychology Agent → 心理动量 + 大赛 DNA（叙事推理）
- MarketBias Agent → 多模型共识 vs 市场偏差（保留规则）

LLM 化：Optimist / Pessimist / Psychology
保留规则：Pathway / MarketBias
"""
from __future__ import annotations
import sys
import json as _json
from pathlib import Path
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))
from agents.base import Agent, AgentOutput
from agents.llm_base import LLMAgent


# ============ 工具：摘取 Top N 队作为 LLM 输入 ============
def _top_teams_brief(context: Dict[str, Any], n: int = 8) -> list:
    teams = context.get("teams", {})
    mc = context.get("mc_results", {})
    injuries = context.get("injuries", {})
    sorted_t = sorted(teams.items(),
                      key=lambda x: -x[1].get("elo", 0))[:n]
    out = []
    for name, d in sorted_t:
        out.append({
            "team": name,
            "elo": d.get("elo"),
            "fifa_rank": d.get("fifa_rank"),
            "group": d.get("group"),
            "market_pct": round(d.get("market_implied", 0) * 100, 2),
            "mc_champion_pct": round(mc.get(name, {}).get("champion", 0) * 100, 2),
            "xg_diff": round(d.get("xg_for", 0) - d.get("xg_against", 0), 2),
            "health_adj_pp": injuries.get(name, {}).get("total_health_adj_pp", 0),
        })
    return out


def _scenarios_brief(context: Dict[str, Any], top_n: int = 8) -> dict:
    """提取 Bull/Base/Bear 三情景的 Top N 队差异"""
    sc = context.get("scenarios", {})
    if not sc:
        return {}
    base = sc.get("base", {})
    bull = sc.get("bull", {})
    bear = sc.get("bear", {})

    # 找 base Top N
    sorted_base = sorted(base.items(),
                         key=lambda x: -x[1].get("champion", 0))[:top_n]
    out = []
    for name, info in sorted_base:
        b = base.get(name, {}).get("champion", 0) * 100
        bl = bull.get(name, {}).get("champion", 0) * 100
        br = bear.get(name, {}).get("champion", 0) * 100
        out.append({
            "team": name,
            "base": round(b, 2),
            "bull": round(bl, 2),
            "bear": round(br, 2),
            "bull_delta": round(bl - b, 2),
            "bear_delta": round(br - b, 2),
        })
    return {"top_teams": out}


# ============================================================
# 1. Optimist Agent（LLM）
# ============================================================
class OptimistAgent(LLMAgent):
    """乐观 Agent：每队找放大优势的逻辑（Reference 3.x.5 体例）"""

    SYSTEM_PROMPT = """你是世界杯研究机构的"乐观 Agent"（Bull case advocate，执行层）。

# 角色与能力边界
- 职责：为顶级球队论证 [优势放大条件]，按 参考报告 3.x.5 体例输出。
- 你只论证利好情景，但必须给概率加权（≤1）的触发条件，不要无脑乐观。
- 你的输出会与 Pessimist Agent 形成辩论对抗，由 Critic 裁决。

# 关键合规约束
- 调整量为 pp（正数为主），单队范围 [-0.5, +1.5]。
- 必须给出 [触发条件] —— 即假设什么成立才会发生这个利好。
- 不得使用 [必中] [稳赢] 等确定性表述。
- 三情景数据缺失时输出空 adjustments + insufficient_data。
"""

    OUTPUT_SCHEMA = {
        "adjustments": {
            "<team>": "<float pp, 推荐 [0.0, +1.5]>"
        },
        "scenarios": [
            {
                "team": "<name>",
                "headline": "<10字以内利好情景标题>",
                "trigger": "<触发条件>",
                "delta_pp": "<float>"
            }
        ],
        "confidence": "<float 0-1, 推荐 0.4-0.5>",
        "rationale": "<一句话总结>"
    }

    def __init__(self):
        super().__init__("Optimist", layer="execution", weight=0.5)

    def build_user_prompt(self, context):
        teams = _top_teams_brief(context, n=8)
        scenarios = _scenarios_brief(context, top_n=8)
        return (
            f"# 任务\n"
            f"为 Top 8 强队论证 [优势放大条件]，按 参考报告 3.x.5 章节体例。\n\n"
            f"# Top 8 候选队基本面\n"
            f"{_json.dumps(teams, ensure_ascii=False, indent=2)}\n\n"
            f"# 三情景对比（Bull vs Base vs Bear）\n"
            f"{_json.dumps(scenarios, ensure_ascii=False, indent=2)}\n\n"
            f"# 任务要求\n"
            f"1. 优先参考 Bull-Base 正向差异 (bull_delta) 给调整量\n"
            f"2. 每队最多写 1 个最强利好情景（headline + trigger）\n"
            f"3. 调整量最大 +1.5pp，加总不要超过 +5pp\n"
            f"4. 不要为已经 mc_champion_pct > 15% 的球队再过度加成\n\n"
            f"按 OUTPUT_SCHEMA 输出 JSON。"
        )

    def parse_response(self, data, context):
        adjustments = {}
        for t, v in (data.get("adjustments") or {}).items():
            try:
                adjustments[t] = max(-0.5, min(1.5, float(v)))
            except Exception:
                continue
        scenarios = data.get("scenarios") or []
        evidence = []
        for s in scenarios[:6]:
            evidence.append(
                f"{s.get('team', '?')}: {s.get('headline', '')} | "
                f"触发: {s.get('trigger', '')[:120]} ({float(s.get('delta_pp', 0)):+.2f}pp)"
            )
        return AgentOutput(
            agent_name=self.name, layer=self.layer,
            role="乐观侧 Bull Case 论证（LLM）",
            probability_dist=adjustments,
            confidence=min(0.55, float(data.get("confidence", 0.45))),
            weight=self.weight,
            rationale=str(data.get("rationale", ""))[:300],
            evidence=evidence,
        )

    def fallback_analyze(self, context):
        scenarios = context.get("scenarios", {})
        bull = scenarios.get("bull", {})
        base = scenarios.get("base", {})
        adjustments = {}
        evidence = []
        if bull and base:
            for team in bull:
                if team in base:
                    bull_p = bull[team].get("champion", 0) * 100
                    base_p = base[team].get("champion", 0) * 100
                    delta = bull_p - base_p
                    if delta > 0.3:
                        adjustments[team] = delta * 0.3
                        evidence.append(
                            f"{team}: Bull {bull_p:.1f}% vs Base {base_p:.1f}% "
                            f"(+{delta:.1f}pp)"
                        )
        return AgentOutput(
            agent_name=self.name, layer=self.layer,
            role="乐观侧（规则）",
            probability_dist=adjustments, confidence=0.4, weight=self.weight,
            rationale="规则版：Bull-Base 差额 × 30% 权重",
            evidence=evidence[:5] if evidence else ["三情景未跑，跳过"],
        )


# ============================================================
# 2. Pessimist Agent（LLM）
# ============================================================
class PessimistAgent(LLMAgent):
    """悲观 Agent：每队找下行风险逻辑（Reference 3.x.6 体例）"""

    SYSTEM_PROMPT = """你是世界杯研究机构的"悲观 Agent"（Bear case advocate，执行层）。

# 角色与能力边界
- 职责：为顶级球队论证 [出局风险与劣势触发条件]，按 参考报告 3.x.6 体例。
- 五大风险维度（Reference 推荐）：
    D1: 关键球员伤病/状态
    D2: 防线不稳/进攻效率
    D3: 心理压力/大赛包袱
    D4: 阵容深度不足
    D5: 历史魔咒（卫冕、东道主、年龄等）
- 你的输出会与 Optimist 辩论对抗。

# 关键合规约束
- 调整量为 pp（仅负数），单队范围 [-2.0, 0]。
- 必须给出 [触发条件]——假设什么发生才会触发该风险。
- 不得编造未公开的伤病或丑闻。
- 健康数据已知时优先引用 health_adj_pp。
"""

    OUTPUT_SCHEMA = {
        "adjustments": {
            "<team>": "<float pp, 仅负数>"
        },
        "risks": [
            {
                "team": "<name>",
                "dimension": "<D1-D5>",
                "headline": "<风险标题>",
                "trigger": "<触发条件>",
                "delta_pp": "<float, 负数>"
            }
        ],
        "confidence": "<float 0-1>",
        "rationale": "<一句话>"
    }

    def __init__(self):
        super().__init__("Pessimist", layer="execution", weight=0.5)

    def build_user_prompt(self, context):
        teams = _top_teams_brief(context, n=8)
        scenarios = _scenarios_brief(context, top_n=8)
        return (
            f"# 任务\n"
            f"为 Top 8 强队论证 [出局风险与劣势触发条件]，按 参考报告 3.x.6 体例。\n\n"
            f"# Top 8 候选队基本面（注意 health_adj_pp 已经反映已知伤病）\n"
            f"{_json.dumps(teams, ensure_ascii=False, indent=2)}\n\n"
            f"# 三情景对比\n"
            f"{_json.dumps(scenarios, ensure_ascii=False, indent=2)}\n\n"
            f"# 任务要求\n"
            f"1. 优先参考 Bear-Base 负向差异 (bear_delta) 给调整量\n"
            f"2. 每队 1-2 个最强风险点，覆盖 D1-D5 中的不同维度\n"
            f"3. 调整量为负，单队不超过 -2.0pp\n"
            f"4. health_adj_pp ≤ -3 的队应额外加重风险\n\n"
            f"按 OUTPUT_SCHEMA 输出 JSON。"
        )

    def parse_response(self, data, context):
        adjustments = {}
        for t, v in (data.get("adjustments") or {}).items():
            try:
                # 强制非正
                adjustments[t] = max(-2.0, min(0.0, float(v)))
            except Exception:
                continue
        risks = data.get("risks") or []
        evidence = []
        for r in risks[:8]:
            evidence.append(
                f"{r.get('team', '?')} [{r.get('dimension', '?')}]: "
                f"{r.get('headline', '')} | 触发: {r.get('trigger', '')[:120]} "
                f"({float(r.get('delta_pp', 0)):+.2f}pp)"
            )
        return AgentOutput(
            agent_name=self.name, layer=self.layer,
            role="悲观侧 Bear Case 论证（LLM）",
            probability_dist=adjustments,
            # 悲观侧自信度容易过高，钳制到 0.55
            confidence=min(0.55, float(data.get("confidence", 0.45))),
            weight=self.weight,
            rationale=str(data.get("rationale", ""))[:300],
            evidence=evidence,
        )

    def fallback_analyze(self, context):
        scenarios = context.get("scenarios", {})
        bear = scenarios.get("bear", {})
        base = scenarios.get("base", {})
        adjustments = {}
        evidence = []
        if bear and base:
            for team in bear:
                if team in base:
                    bear_p = bear[team].get("champion", 0) * 100
                    base_p = base[team].get("champion", 0) * 100
                    delta = bear_p - base_p
                    if delta < -0.3:
                        adjustments[team] = delta * 0.3
                        evidence.append(
                            f"{team}: Bear {bear_p:.1f}% vs Base {base_p:.1f}% "
                            f"({delta:+.1f}pp)"
                        )
        return AgentOutput(
            agent_name=self.name, layer=self.layer,
            role="悲观侧（规则）",
            probability_dist=adjustments, confidence=0.4, weight=self.weight,
            rationale="规则版：Bear-Base 差额 × 30% 权重",
            evidence=evidence[:5] if evidence else ["三情景未跑，跳过"],
        )


# ============================================================
# 3. Pathway Agent（保留规则——纯结构化判定）
# ============================================================
class PathwayAgent(Agent):
    """淘汰赛路径分析（保留规则版，结构化判定 LLM 价值低）"""

    def __init__(self):
        super().__init__("Pathway", layer="execution", weight=0.6)

    def analyze(self, context):
        teams = context.get("teams", {})
        upper_half_groups = ["E", "F", "H", "I"]
        adjustments = {}
        evidence = []
        for team, data in teams.items():
            group = data.get("group")
            elo = data.get("elo", 1500)
            if group in upper_half_groups and elo > 2000:
                adj = -0.5
                evidence.append(f"{team}({group}): 上半区死亡半区 -0.5pp")
            elif group not in upper_half_groups and group != "_" and elo > 1900:
                adj = +0.3
                evidence.append(f"{team}({group}): 下半区路径红利 +0.3pp")
            else:
                adj = 0.0
            adjustments[team] = adj
        return AgentOutput(
            agent_name=self.name, layer=self.layer,
            role="半区效应 + 路径红利识别",
            probability_dist=adjustments, confidence=0.65, weight=self.weight,
            rationale="上半区集中 Spain/France/Germany/Netherlands 4 强队",
            evidence=evidence[:5],
        )


# ============================================================
# 4. Psychology Agent（LLM）
# ============================================================
class PsychologyAgent(LLMAgent):
    """心理动量 + 大赛 DNA（叙事推理 → LLM 化）"""

    SYSTEM_PROMPT = """你是世界杯研究机构的"心理动量 + 近期表现 Agent"（执行层）。

# 角色与能力边界
- 职责：基于公开赛绩、教练叙事、关键球员心态，判断球队心理向量。
- **重点审查 Elo 评级与近期实际表现的偏差**（这是模型 vs 市场分歧的常见来源）。
- 你的输出主观性强，confidence ≤ 0.5 是合理的。

# 心理因子参考清单
正向（典型 +0.3 到 +0.6 pp）：
  - 教练 + 核心球员形成的 [赢球文化]（如阿根廷三冠王心态）
  - 近期大赛冠军延续动能（西班牙 2024 欧洲杯）
  - 核心球员 [最后一舞] 叙事
负向（典型 -0.3 到 -0.8 pp）：
  - 长期无冠 + 关键时刻心理脆弱（英格兰 60 年）
  - 教练经验不足（如新晋国家队主帅）
  - 历史魔咒（葡萄牙从未进世界杯决赛）

# 【新增】Elo 衰减审查维度（典型 -0.5 到 -1.5 pp）
某些队的高 Elo 高度依赖**单次爆发事件**（如美洲杯亚军、单届欧洲杯），
但**近 12 个月表现已明显回落**。需检查：
  - Colombia: Elo 1982 主要来自 2024 美洲杯亚军 + 28 场不败纪录，
    但 2025-2026 预选赛仅以第 3 名险胜出线，6-3 险胜委内瑞拉显示防守崩盘，
    核心 J. Rodriguez 已 34 岁 → 建议 -1.5pp
  - Ecuador: 类似情况，Elo 1938 偏高 vs 实际状态 → 建议 -1.0pp
  - 历史亚军后衰退队普遍 → -0.8 至 -1.5pp
对应这类"Elo 滞后"的队，应给负向调整以校正模型偏高。

# 关键合规约束
- 调整量范围 [-1.5, +1.0] pp。
- 不得编造负面叙事——必须能引用公开事实。
- 仅对 Top 14 队评估，其他队留空。
- **强制审查 Colombia / Ecuador 等"Elo 高但市场低"的队**
"""

    OUTPUT_SCHEMA = {
        "adjustments": {
            "<team>": "<float pp, [-1.0, +1.0]>"
        },
        "narratives": [
            {
                "team": "<name>",
                "polarity": "<positive/negative>",
                "story": "<一句话叙事>",
                "delta_pp": "<float>"
            }
        ],
        "confidence": "<float 0-1, <= 0.5>",
        "rationale": "<一句话>"
    }

    def __init__(self):
        # weight 提升到 0.9：心理因子要能对 Elo 滞后做有效校正
        super().__init__("Psychology", layer="execution", weight=0.9)

    def build_user_prompt(self, context):
        teams = _top_teams_brief(context, n=14)
        # 计算 Elo vs Market 的偏差，提示 LLM 关注
        teams_dict = context.get("teams", {})
        elo_market_div = []
        for t, d in teams_dict.items():
            elo = d.get("elo", 0)
            mkt = d.get("market_implied", 0) * 100
            if elo >= 1900 and mkt > 0:
                # 粗略 Elo 隐含夺冠概率（用 Elo Top 排名估算）
                elo_implied = max(0, (elo - 1700) / 50)  # 简易估算
                div = elo_implied - mkt
                if div >= 2.5:   # 模型 Elo 显著高于市场 → 可能高估
                    elo_market_div.append({
                        "team": t, "elo": elo,
                        "market_pct": round(mkt, 1),
                        "elo_implied_pct_approx": round(elo_implied, 1),
                        "divergence_pp": round(div, 1),
                    })
        return (
            f"# 任务\n"
            f"基于公开赛绩与人物叙事，给 Top 14 强队的心理向量做 pp 调整。\n\n"
            f"# Top 14 候选队\n"
            f"{_json.dumps(teams, ensure_ascii=False, indent=2)}\n\n"
            f"# 【重点审查】Elo 高于市场的疑似 Elo 滞后队\n"
            f"{_json.dumps(elo_market_div, ensure_ascii=False, indent=2)}\n\n"
            f"# 任务要求\n"
            f"- 每队至多 1 条最强叙事（正/负）\n"
            f"- 调整量 [-1.5, +1.0]\n"
            f"- 必须能援引公开事实\n"
            f"- 对 Elo 滞后队（如 Colombia/Ecuador）应主动给负向调整\n"
            f"- confidence 不超过 0.5\n\n"
            f"按 OUTPUT_SCHEMA 输出 JSON。"
        )

    def parse_response(self, data, context):
        adjustments = {}
        for t, v in (data.get("adjustments") or {}).items():
            try:
                adjustments[t] = max(-1.5, min(1.0, float(v)))
            except Exception:
                continue
        narratives = data.get("narratives") or []
        evidence = []
        for n in narratives[:8]:
            polarity_mark = "+" if n.get("polarity") == "positive" else "-"
            evidence.append(
                f"{n.get('team', '?')} [{polarity_mark}]: {n.get('story', '')[:140]} "
                f"({float(n.get('delta_pp', 0)):+.2f}pp)"
            )
        return AgentOutput(
            agent_name=self.name, layer=self.layer,
            role="心理动量 + 大赛 DNA（LLM）",
            probability_dist=adjustments,
            confidence=min(0.5, float(data.get("confidence", 0.4))),
            weight=self.weight,
            rationale=str(data.get("rationale", ""))[:300],
            evidence=evidence,
        )

    # 心理叙事文本（仅用于 evidence 展示）
    PSYCH_RATIONALES = {
        "Spain":       "32 场不败 + 2024 欧洲杯冠军 + 年轻核心 (Yamal/Pedri/Gavi)",
        "France":      "德尚谢幕激励 + 阵容深度 + 近 4 届 2 决赛 1 冠 1 亚",
        "Germany":     "纳格尔斯曼复苏 + 2024 欧洲杯主场 4 强 + Musiala/Wirtz 黄金中场",
        "Mexico":      "东道主主场加成 + 阿兹特克魔咒",
        "USA":         "东道主主场 + 年轻一代 (Pulisic/McKennie)",
        "Argentina":   "卫冕魔咒 (1962 后 15 届无卫冕) + Messi 38 老化 + 缺淘汰赛 X 因子",
        "England":     "60 年无冠 + 2020 欧洲杯决赛点球失利 + Saka 伤情",
        "Brazil":      "安切洛蒂大赛执教样本不足 (12 场国家队)",
        "Portugal":    "队史从未进入世界杯决赛 + C 罗时代尾声",
        "Netherlands": "2010 决赛后大赛点球出局 2 次 + 核心 Timber/Simons 伤",
        "Belgium":     "黄金一代退役 + 新核心尚未建立",
        "Colombia":    "Elo 1982 依赖 2024 美洲杯亚军；2025 预选赛仅以第 3 险胜出线；J. Rodriguez 34 老化",
        "Ecuador":     "Elo 1938 含 2024 美洲杯余温；预选赛中游；缺世界级球星",
        "Norway":      "Elo 1914 单核 Haaland 依赖严重；国家队大赛经验极少",
        "Croatia":     "黄金一代 Modric 谢幕，无新替代",
        "Morocco":     "2022 4 强余温但 2024 美洲杯表现一般",
        "Switzerland": "无显著心理特质",
    }

    # swarm 通道对 Psychology 信号有缩放，部分球队需要放大才能等效 synth
    SWARM_AMPLIFY = {"Germany": 2.0}

    def fallback_analyze(self, context):
        """
        【单一真源设计】
        直接读 synthesizer.DEFAULT_ADJUSTMENTS['psych'] 作为 psych 数值，
        rationale 文本由本类 PSYCH_RATIONALES 提供，
        SWARM_AMPLIFY 表对特定球队应用 swarm 等效放大。

        这样 synth 和 swarm 永远使用同一份 psych 数据，无漂移可能。
        """
        try:
            from models.synthesizer import DEFAULT_ADJUSTMENTS
        except ImportError:
            DEFAULT_ADJUSTMENTS = {}

        adjustments = {}
        evidence = []
        for team, adj in DEFAULT_ADJUSTMENTS.items():
            base_pp = float(adj.get("psych", 0.0))
            amplified = base_pp * self.SWARM_AMPLIFY.get(team, 1.0)
            adjustments[team] = amplified
            rationale = self.PSYCH_RATIONALES.get(team, "(synth 表默认)")
            evidence.append(f"{team}: {rationale} ({amplified:+.1f}pp)")

        return AgentOutput(
            agent_name=self.name, layer=self.layer,
            role="心理动量（synth 表单一真源）",
            probability_dist=adjustments, confidence=0.4, weight=self.weight,
            rationale="数据源：synthesizer.DEFAULT_ADJUSTMENTS['psych']（保证 swarm/synth 一致）",
            evidence=evidence,
        )


# ============================================================
# 5. MarketBias Agent（保留规则——已有 LLM Market 在战术层）
# ============================================================
class MarketBiasAgent(Agent):
    """市场偏差识别（保留规则，避免与战术层 MarketAgent 重复 LLM）"""

    def __init__(self):
        super().__init__("MarketBias", layer="execution", weight=0.7)

    def analyze(self, context):
        external = context.get("external_predictions", {})
        teams = context.get("teams", {})
        adjustments = {}
        evidence = []
        models = ["engine", "report", "opta"]
        for team_name in teams:
            preds = []
            for m in models:
                p = external.get(m, {}).get("predictions", {}).get(team_name)
                if p is not None:
                    preds.append(p)
            if len(preds) >= 2:
                consensus = sum(preds) / len(preds)
                market = teams[team_name].get("market_implied", 0) * 100
                if market > 0:
                    bias = consensus - market
                    if abs(bias) >= 2.0:
                        adjustments[team_name] = bias * 0.2
                        if bias > 0:
                            evidence.append(
                                f"{team_name}: 共识 {consensus:.1f}% > 市场 "
                                f"{market:.1f}% (+{bias:.1f}pp 被低估)"
                            )
                        else:
                            evidence.append(
                                f"{team_name}: 共识 {consensus:.1f}% < 市场 "
                                f"{market:.1f}% ({bias:.1f}pp 被高估)"
                            )
        return AgentOutput(
            agent_name=self.name, layer=self.layer,
            role="市场共识偏差识别",
            probability_dist=adjustments, confidence=0.6, weight=self.weight,
            rationale="多模型共识 vs 市场赔率，识别系统性低估/高估",
            evidence=evidence[:5],
        )
