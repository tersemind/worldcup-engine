"""
LLM 多轮辩论引擎（参考报告 2.3.3 完整实现）
===============================================

按报告原文实现三角色辩论：
  - 正方（Pro）：捍卫较高估计
  - 反方（Con）：捍卫较低估计
  - 仲裁（Arbiter）：由 RiskPerception 兼任

L2 流程（15-30pp）：
  1. 正方陈述（1 次 LLM）
  2. 反方陈述（1 次 LLM）
  3. 仲裁裁决（1 次 LLM）
  → 共 3 次 LLM 调用

L3 流程（30-50pp）：
  Round 1: 正方 + 反方 + 仲裁评估（3 次）
  Round 2: 正方反驳 + 反方反驳 + 仲裁评估（3 次）
  Round 3: 正方终陈 + 反方终陈 + 仲裁终裁（3 次）
  → 共 9 次 LLM 调用
  → 若 Round 1/2 后仲裁宣布达成共识，提前结束

L4 流程（>50pp）：
  暂停定量，LLM 仅生成定性冲突报告（1 次）

每次失败自动 fallback 到数值辩论（debate.py）。
"""
from __future__ import annotations
import json
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import asdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from agents.base import AgentOutput
from agents.debate import DebateResult, LEVEL_LABEL, CONFIDENCE_PENALTY
from llm import get_default_client, LLMUnavailable
from llm.client import LLMClient

logger = logging.getLogger("worldcup.llm_debate")


# ============ Prompt 模板 ============
PRO_SYSTEM = """你是"正方 Agent"，在世界杯预测辩论中捍卫**较高的概率估计**。

## 角色任务
- 团队：{team}
- 你的初始估计：{pro_estimate:.2f}pp
- 对方估计：{con_estimate:.2f}pp
- 分歧度：{spread:.1f}pp（属于 Level {level}）

## 必须做的
1. 给出 2-3 条**具体证据**支撑你的高估计（基于公开数据/赛事/球员状态）
2. 评估对方观点的弱点
3. 第 N 轮陈述（共 {max_rounds} 轮）—— 越后面的轮次越要**精确，给新证据**

## 严格约束
- 你不能"软化立场"——必须坚持高估计直到被仲裁说服
- 证据必须能引用公开事实
- 不得编造伤病/事件
"""

CON_SYSTEM = """你是"反方 Agent"，在世界杯预测辩论中捍卫**较低的概率估计**。

## 角色任务
- 团队：{team}
- 你的初始估计：{con_estimate:.2f}pp
- 对方估计：{pro_estimate:.2f}pp
- 分歧度：{spread:.1f}pp（属于 Level {level}）

## 必须做的
1. 给出 2-3 条**具体证据**支撑你的低估计
2. 评估对方观点的弱点
3. 第 N 轮陈述（共 {max_rounds} 轮）

## 严格约束
- 坚持低估计立场
- 必须引用公开数据/赛事/球员状态
- 不得编造
"""

ARBITER_SYSTEM = """你是"仲裁 Agent"（由 RiskPerception 兼任），在多 Agent 辩论中做最终裁决。

## 仲裁原则（参考报告 2.3.3）
- **证据质量优于声量**：给出具体公开数据的一方更可信
- **保守优先**：分歧未消解时倾向更低估计（避免泡沫）
- **量化置信度**：必须给出 [0, 1] 的 confidence_in_resolution

## 任务
- 团队：{team}
- 正方估计：{pro_estimate:.2f}pp，论证摘要：{pro_summary}
- 反方估计：{con_estimate:.2f}pp，论证摘要：{con_summary}
- 当前轮：{round_num}/{max_rounds}

## 你的产出
1. 评估两方证据质量（pro_evidence_score / con_evidence_score 各 0-100）
2. 给出最终 consensus_pp（介于两端，向证据强方倾斜）
3. 判断 resolved=true/false：是否已可终结辩论
4. 一句话裁决理由
"""

OUTPUT_SCHEMA_AGENT = {
    "estimate_pp": "<float, 你坚持的估计>",
    "evidence": ["<证据1>", "<证据2>", "<证据3>"],
    "opponent_weakness": "<对方观点的核心弱点，<=80字>",
    "rationale": "<一句话总结，<=100字>"
}

OUTPUT_SCHEMA_ARBITER = {
    "pro_evidence_score": "<int 0-100>",
    "con_evidence_score": "<int 0-100>",
    "consensus_pp": "<float, 最终共识 [con, pro] 区间内>",
    "resolved": "<bool: 是否已达成共识>",
    "verdict": "<裁决理由 <=100字>"
}


# ============ 核心辩论引擎 ============
class LLMDebateEngine:
    """完整 Reference 三轮辩论"""

    def __init__(self, client: Optional[LLMClient] = None,
                  max_tokens: int = 6000):
        try:
            self.client = client or get_default_client()
            self.available = True
        except LLMUnavailable:
            self.client = None
            self.available = False
        self.max_tokens = max_tokens

    # ============ L2: 一轮辩论 ============
    def debate_l2(self, team: str,
                   pro: AgentOutput, con: AgentOutput,
                   spread: float) -> Dict[str, Any]:
        """3 次 LLM 调用：正方 → 反方 → 仲裁"""
        if not self.available:
            raise LLMUnavailable("LLM 不可用")

        pro_est = pro.point_estimate if pro.point_estimate is not None else (
            pro.probability_dist.get(team, 0) if pro.probability_dist else 0
        )
        con_est = con.point_estimate if con.point_estimate is not None else (
            con.probability_dist.get(team, 0) if con.probability_dist else 0
        )
        if pro_est < con_est:
            pro, con = con, pro
            pro_est, con_est = con_est, pro_est

        rounds = []

        # Round 1 正方
        pro_msg = self._agent_statement(
            "pro", team, pro_est, con_est, spread, level=2,
            round_num=1, max_rounds=1, agent_ctx=pro,
            opponent_ctx=con,
        )
        rounds.append({"role": "pro", "round": 1, **pro_msg})

        # Round 1 反方
        con_msg = self._agent_statement(
            "con", team, pro_est, con_est, spread, level=2,
            round_num=1, max_rounds=1, agent_ctx=con,
            opponent_ctx=pro, prior_statements=[pro_msg],
        )
        rounds.append({"role": "con", "round": 1, **con_msg})

        # 仲裁
        arbiter_msg = self._arbiter_judge(
            team, pro_est, con_est, pro_msg, con_msg,
            round_num=1, max_rounds=1,
        )
        rounds.append({"role": "arbiter", "round": 1, **arbiter_msg})

        return {
            "level": 2,
            "consensus_pp": float(arbiter_msg.get("consensus_pp", (pro_est + con_est) / 2)),
            "resolved": bool(arbiter_msg.get("resolved", True)),
            "n_llm_calls": 3,
            "rounds": rounds,
            "arbiter_verdict": arbiter_msg.get("verdict", ""),
        }

    # ============ L3: 三轮辩论 ============
    def debate_l3(self, team: str,
                   pro: AgentOutput, con: AgentOutput,
                   spread: float, max_rounds: int = 3) -> Dict[str, Any]:
        """最多 9 次 LLM 调用，仲裁说 resolved 则提前结束"""
        if not self.available:
            raise LLMUnavailable("LLM 不可用")

        pro_est = pro.point_estimate if pro.point_estimate is not None else (
            pro.probability_dist.get(team, 0) if pro.probability_dist else 0
        )
        con_est = con.point_estimate if con.point_estimate is not None else (
            con.probability_dist.get(team, 0) if con.probability_dist else 0
        )
        if pro_est < con_est:
            pro, con = con, pro
            pro_est, con_est = con_est, pro_est

        rounds_log: List[Dict[str, Any]] = []
        prior_pro: List[Dict[str, Any]] = []
        prior_con: List[Dict[str, Any]] = []
        n_calls = 0
        final_consensus = (pro_est + con_est) / 2
        last_arbiter: Dict[str, Any] = {}

        for round_num in range(1, max_rounds + 1):
            # 正方
            pro_msg = self._agent_statement(
                "pro", team, pro_est, con_est, spread, level=3,
                round_num=round_num, max_rounds=max_rounds,
                agent_ctx=pro, opponent_ctx=con,
                prior_statements=prior_con,
            )
            n_calls += 1
            prior_pro.append(pro_msg)
            rounds_log.append({"role": "pro", "round": round_num, **pro_msg})

            # 反方
            con_msg = self._agent_statement(
                "con", team, pro_est, con_est, spread, level=3,
                round_num=round_num, max_rounds=max_rounds,
                agent_ctx=con, opponent_ctx=pro,
                prior_statements=prior_pro,
            )
            n_calls += 1
            prior_con.append(con_msg)
            rounds_log.append({"role": "con", "round": round_num, **con_msg})

            # 仲裁
            arbiter_msg = self._arbiter_judge(
                team, pro_est, con_est, pro_msg, con_msg,
                round_num=round_num, max_rounds=max_rounds,
            )
            n_calls += 1
            last_arbiter = arbiter_msg
            rounds_log.append({"role": "arbiter", "round": round_num, **arbiter_msg})

            final_consensus = float(arbiter_msg.get("consensus_pp", final_consensus))
            if arbiter_msg.get("resolved", False):
                break

        return {
            "level": 3,
            "consensus_pp": final_consensus,
            "resolved": bool(last_arbiter.get("resolved", False)),
            "n_llm_calls": n_calls,
            "rounds": rounds_log,
            "arbiter_verdict": last_arbiter.get("verdict", ""),
        }

    # ============ L4: 暂停定量，仅出定性报告 ============
    def debate_l4(self, team: str,
                   pro: AgentOutput, con: AgentOutput,
                   spread: float) -> Dict[str, Any]:
        """1 次 LLM 调用生成冲突报告"""
        if not self.available:
            raise LLMUnavailable("LLM 不可用")

        pro_est = pro.point_estimate if pro.point_estimate is not None else (
            pro.probability_dist.get(team, 0) if pro.probability_dist else 0
        )
        con_est = con.point_estimate if con.point_estimate is not None else (
            con.probability_dist.get(team, 0) if con.probability_dist else 0
        )

        system = """你是 冲突报告生成器。两个核心 Agent 对一个球队的估计差异 >50pp，
按 参考报告 2.3.3 节，应**暂停定量预测、输出定性结论**。
"""
        user = f"""# 冲突场景
- 球队: {team}
- Agent A: {pro.agent_name} 估计 {pro_est:.1f}pp
- Agent B: {con.agent_name} 估计 {con_est:.1f}pp
- 分歧: {spread:.1f}pp

# 任务
按下面 schema 输出 JSON：
- qualitative_judgment: 一句话定性判断（如"高度不确定，可能受 X 影响"）
- recommended_action: "human_review" / "pause_update" / "fetch_more_data"
- possible_causes: 列出 2-3 个可能的分歧根因
- conservative_estimate_pp: 保守的中位数估计
"""
        try:
            text = self.client.chat(system=system, user=user,
                                       json_mode=True, max_tokens=2000,
                                       temperature=0.3)
            data = LLMClient.extract_json(text)
            return {
                "level": 4,
                "consensus_pp": float(data.get("conservative_estimate_pp",
                                                  (pro_est + con_est) / 2)),
                "resolved": False,
                "n_llm_calls": 1,
                "qualitative_report": data,
                "arbiter_verdict": data.get("qualitative_judgment", ""),
            }
        except Exception as e:
            raise LLMUnavailable(f"L4 LLM 失败: {e}")

    # ============ 内部：单 Agent 陈述 ============
    def _agent_statement(self, role: str, team: str,
                          pro_est: float, con_est: float, spread: float,
                          level: int, round_num: int, max_rounds: int,
                          agent_ctx: AgentOutput,
                          opponent_ctx: AgentOutput,
                          prior_statements: Optional[List[Dict]] = None
                          ) -> Dict[str, Any]:
        """单轮单方陈述（带历史上下文）"""
        sys_template = PRO_SYSTEM if role == "pro" else CON_SYSTEM
        system = sys_template.format(
            team=team, pro_estimate=pro_est, con_estimate=con_est,
            spread=spread, level=level, max_rounds=max_rounds,
        )
        system = system + "\n\n## 输出 JSON Schema\n" + json.dumps(
            OUTPUT_SCHEMA_AGENT, ensure_ascii=False, indent=2,
        )

        user_parts = [
            f"# 第 {round_num}/{max_rounds} 轮陈述",
            "",
            f"## 你的初始 Agent ({agent_ctx.agent_name}) 背景",
            f"- 角色: {agent_ctx.role}",
            f"- Rationale: {agent_ctx.rationale}",
            f"- 自带证据: {agent_ctx.evidence[:3]}",
            "",
            f"## 对方 Agent ({opponent_ctx.agent_name}) 摘要",
            f"- Rationale: {opponent_ctx.rationale}",
            f"- 证据数: {len(opponent_ctx.evidence)}",
        ]
        if prior_statements:
            user_parts.append("\n## 对方上一轮陈述")
            for s in prior_statements[-1:]:
                user_parts.append(json.dumps({
                    "estimate_pp": s.get("estimate_pp"),
                    "evidence": s.get("evidence", [])[:3],
                    "rationale": s.get("rationale", ""),
                }, ensure_ascii=False))
        user_parts.append("\n按 OUTPUT_SCHEMA 输出 JSON。")
        user = "\n".join(user_parts)

        text = self.client.chat(system=system, user=user, json_mode=True,
                                   temperature=0.4, max_tokens=self.max_tokens)
        return LLMClient.extract_json(text)

    # ============ 内部：仲裁裁决 ============
    def _arbiter_judge(self, team: str,
                        pro_est: float, con_est: float,
                        pro_msg: Dict, con_msg: Dict,
                        round_num: int, max_rounds: int) -> Dict[str, Any]:
        system = ARBITER_SYSTEM.format(
            team=team, pro_estimate=pro_est, con_estimate=con_est,
            pro_summary=str(pro_msg.get("rationale", ""))[:120],
            con_summary=str(con_msg.get("rationale", ""))[:120],
            round_num=round_num, max_rounds=max_rounds,
        )
        system = system + "\n\n## 输出 JSON Schema\n" + json.dumps(
            OUTPUT_SCHEMA_ARBITER, ensure_ascii=False, indent=2,
        )
        user = f"""# 第 {round_num}/{max_rounds} 轮仲裁

## 正方完整陈述
{json.dumps(pro_msg, ensure_ascii=False, indent=2)}

## 反方完整陈述
{json.dumps(con_msg, ensure_ascii=False, indent=2)}

按 OUTPUT_SCHEMA 输出 JSON。"""
        text = self.client.chat(system=system, user=user, json_mode=True,
                                   temperature=0.2, max_tokens=self.max_tokens)
        return LLMClient.extract_json(text)


# ============ 统一入口：兼容数值 fallback ============
def run_llm_debate(team: str, level: int,
                    pro: AgentOutput, con: AgentOutput,
                    spread: float,
                    engine: Optional[LLMDebateEngine] = None,
                    ) -> Optional[Dict[str, Any]]:
    """
    跑 LLM 辩论。
    
    Returns:
        如果成功：完整 LLM 辩论结果 dict
        如果 LLM 不可用：None（调用方应回退到数值辩论）
    """
    engine = engine or LLMDebateEngine()
    if not engine.available:
        return None

    try:
        if level == 2:
            return engine.debate_l2(team, pro, con, spread)
        elif level == 3:
            return engine.debate_l3(team, pro, con, spread, max_rounds=3)
        elif level == 4:
            return engine.debate_l4(team, pro, con, spread)
        else:
            return None  # L1 不需要 LLM
    except LLMUnavailable as e:
        logger.warning(f"LLM 辩论失败 ({team} L{level}): {e}")
        return None
    except Exception as e:
        logger.exception(f"LLM 辩论异常 ({team} L{level})")
        return None
