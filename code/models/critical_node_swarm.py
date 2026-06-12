"""
关键节点 5-Agent 微调器
===========================
对方案 B 选出的"最可能剧本"中**胶着场**（胜率 35-65%）做 LLM 路径级博弈。

5 个专门的"路径级 Agent"（区别于球队级 17-Agent Swarm）：

  1. TacticalCounterAgent   战术相克分析（控球 vs 反击、高位逼抢相克等）
  2. PlayerFormAgent        关键球员近期状态（受伤复出、状态低迷）
  3. CoachMatchupAgent      教练博弈历史（同教练 vs 此战术体系胜率）
  4. PsychologyAgent2       淘汰赛压力学（卫冕魔咒、东道主优势、关键场心态）
  5. HeadToHeadAgent        历史交锋数据（近 10 年同样对位结果）

聚合方式：Byzantine 多数票
  - 5 Agent 各输出: {推荐胜方, 修正pp (-5 ~ +5), 理由}
  - 若 ≥3 Agent 推荐同一胜方 → 取这些 Agent 修正pp均值
  - 若分歧（无 3-Agent 多数）→ 不修正（标记为"分歧"，保留 MC 原值）

修正 pp 应用：
  - 例如 MC 给 Spain 50.5% vs Brazil 49.5%（胶着）
  - 5 Agent 多数推 Spain，平均修正 +1.5pp → Spain 52% vs Brazil 48%
  - 单 Agent 修正限幅 ±5pp，避免幻觉过度
"""
from __future__ import annotations
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))
from llm import get_default_client, LLMUnavailable
from llm.client import LLMClient

logger = logging.getLogger("worldcup.critical_node")


# ============ Agent Prompt 定义 ============
AGENT_PROMPTS = {
    "tactical_counter": {
        "name": "战术相克 Agent",
        "system": """你是足球战术分析师，专门分析两队战术体系的相克性。

# 任务
判断 A 队和 B 队的战术体系是否存在"风格相克"关系，给出对结果的修正建议。

# 输出 JSON
{
  "predicted_winner": "<A 队名或 B 队名>",
  "adjustment_pp": <数字, 范围 -5 到 +5，正数表示推荐胜方的胜率应上调>,
  "rationale": "<不超过60字的战术相克逻辑>"
}

# 约束
- 必须基于历史数据（如 PPDA、控球率、xG 转化等）
- 修正幅度保守，无明确相克关系时填 0
- 严格 JSON 输出""",
    },
    "player_form": {
        "name": "球员状态 Agent",
        "system": """你是球员状态追踪师，专注于关键球员近 3 月的状态和伤病情况。

# 任务
判断哪一方的关键球员状态/出场可用性更优，给出对结果的修正建议。

# 输出 JSON
{
  "predicted_winner": "<A 队名或 B 队名>",
  "adjustment_pp": <-5 到 +5>,
  "rationale": "<不超过60字，提到具体球员名>"
}

# 约束
- 提及具体球员（如姆巴佩、贝林厄姆、维尼修斯）
- 不编造伤病，仅引用公开信息
- 严格 JSON 输出""",
    },
    "coach_matchup": {
        "name": "教练博弈 Agent",
        "system": """你是教练战术博弈分析师，分析两队主教练的执教风格和大赛经验。

# 任务
判断哪一方主教练在此类对阵中更具优势，给出对结果的修正建议。

# 输出 JSON（必须严格返回，绝不留空，不可拒答）
{
  "predicted_winner": "<A 队名或 B 队名>",
  "adjustment_pp": <-5 到 +5>,
  "rationale": "<不超过60字，提及教练名或描述差异>"
}

# 重要：处理「不熟悉的教练」
- 若你**不确定教练姓名**或**信息不足**：predicted_winner 仍必须选一方（比如按球队的整体大赛积淀），adjustment_pp 填 0，rationale 写 "教练信息有限，无明显偏向"
- 严禁返回空字符串、null 或拒答
- 严格 JSON 输出""",
    },
    "psychology": {
        "name": "淘汰赛心态 Agent",
        "system": """你是足球心理学家，专门分析淘汰赛压力下球队的心态优势/劣势。

# 任务
判断哪一方在此场淘汰赛压力下心态更稳，给出对结果的修正建议。

# 考虑因素
- 卫冕冠军魔咒（2022 阿根廷）
- 东道主优势/压力（2026 美/加/墨）
- 球队历史心理障碍（点球大战、关键场失利）
- 球员平均年龄和大赛经验

# 输出 JSON
{
  "predicted_winner": "<A 队名或 B 队名>",
  "adjustment_pp": <-5 到 +5>,
  "rationale": "<不超过60字>"
}""",
    },
    "head_to_head": {
        "name": "历史交锋 Agent",
        "system": """你是足球历史数据分析师，专注于两队近 10 年的直接交锋数据。

# 任务
基于历史对阵记录判断结果倾向，给出对结果的修正建议。

# 输出 JSON（必须严格返回，绝不留空，不可拒答）
{
  "predicted_winner": "<A 队名或 B 队名>",
  "adjustment_pp": <-5 到 +5>,
  "rationale": "<不超过60字>"
}

# 重要：处理「无交锋数据」
- 若两队**近 10 年没有交锋记录**或**样本极少**（非常常见，尤其跨大洲对阵如 Iran vs Paraguay）：
  - predicted_winner 仍必须选一方（按整体实力，例如 Elo 更高的一方）
  - adjustment_pp 填 0
  - rationale 写 "近 10 年无交锋记录，按整体实力默认 X"
- 严禁返回空字符串、null 或拒答
- 严禁编造比分
- 严格 JSON 输出""",
    },
}


# ============ 单 Agent 调用 ============
def run_single_agent(agent_key: str, team_a: str, team_b: str,
                      mc_p_win_a: float, team_a_info: Dict, team_b_info: Dict,
                      client: LLMClient) -> Optional[Dict]:
    """跑单个 Agent，返回 {predicted_winner, adjustment_pp, rationale} 或 None"""
    agent = AGENT_PROMPTS[agent_key]
    
    user = f"""# 对阵
{team_a} (Elo {team_a_info.get('elo')}, xG {team_a_info.get('xg_for')}/{team_a_info.get('xg_against')}) 
  vs 
{team_b} (Elo {team_b_info.get('elo')}, xG {team_b_info.get('xg_for')}/{team_b_info.get('xg_against')})

# 量化模型预测
- MC 给 {team_a} 胜率: {mc_p_win_a*100:.1f}%
- MC 给 {team_b} 胜率: {(1-mc_p_win_a)*100:.1f}%
- 这是一场"胶着场"，请基于你的专业维度给出修正建议

# 任务
按你专长的维度（{agent['name']}）分析，给出 JSON 输出。"""
    
    # 最多重试 2 次（空返回是典型偶发问题）
    last_err = None
    data = None
    for attempt in range(2):
        try:
            # 第二次重试用更高 temperature 增加多样性，避免 LLM 卡在"拒答"模式
            temp = 0.3 if attempt == 0 else 0.7
            text = client.chat(system=agent["system"], user=user,
                                json_mode=True, max_tokens=500, temperature=temp)
            data = LLMClient.extract_json(text)
            if "predicted_winner" in data and "adjustment_pp" in data:
                break
            last_err = f"字段缺失: {data}"
        except Exception as e:
            last_err = str(e)
            data = None
    if not data or "predicted_winner" not in data or "adjustment_pp" not in data:
        logger.warning(f"{agent_key} 重试 2 次仍失败: {last_err}")
        return None
    
    try:
        # 修正幅度 clip
        try:
            adj = float(data["adjustment_pp"])
        except (TypeError, ValueError):
            adj = 0.0
        adj = max(-5.0, min(5.0, adj))
        # 校验 predicted_winner 合法性
        winner = data["predicted_winner"].strip()
        if winner not in (team_a, team_b):
            # LLM 可能给中文名/缩写，做容错
            if team_a in winner or winner in team_a:
                winner = team_a
            elif team_b in winner or winner in team_b:
                winner = team_b
            else:
                logger.warning(f"{agent_key} winner 不识别: {winner!r}")
                return None
        return {
            "agent": agent_key,
            "agent_name": agent["name"],
            "predicted_winner": winner,
            "adjustment_pp": adj,
            "rationale": data.get("rationale", "")[:120],
        }
    except Exception as e:
        logger.warning(f"{agent_key} 调用失败: {e}")
        return None


# ============ 5-Agent Byzantine 聚合 ============
def run_critical_node_swarm(team_a: str, team_b: str,
                              mc_p_win_a: float,
                              team_a_info: Dict, team_b_info: Dict,
                              client: Optional[LLMClient] = None) -> Dict:
    """
    对一场胶着对阵跑 5-Agent + Byzantine 多数票聚合。
    
    Returns:
        {
            "applied": True/False,  ← 是否产生有效修正
            "final_adjustment_pp": <正负数, 加到 team_a 胜率上>,
            "consensus_winner": team_a 或 team_b 或 None,
            "majority_count": 3-5,  ← 多数票 Agent 数
            "agents": [...],  ← 各 Agent 详细输出
            "summary": "<人类可读总结>"
        }
    """
    if client is None:
        try:
            client = get_default_client()
        except LLMUnavailable as e:
            return {"applied": False, "error": f"LLM 不可用: {e}"}
    
    # 跑 5 Agent（顺序执行，避免 API 并发限制）
    agent_outputs = []
    for key in AGENT_PROMPTS.keys():
        out = run_single_agent(key, team_a, team_b, mc_p_win_a,
                                 team_a_info, team_b_info, client)
        if out:
            agent_outputs.append(out)
    
    if len(agent_outputs) < 3:
        return {
            "applied": False,
            "error": f"有效 Agent 输出不足 3 个（实际 {len(agent_outputs)}）",
            "agents": agent_outputs,
        }
    
    # Byzantine 多数票
    winner_votes = Counter(a["predicted_winner"] for a in agent_outputs)
    top_winner, top_count = winner_votes.most_common(1)[0]
    
    if top_count < 3:
        # 分歧，无 3-Agent 多数 → 不修正
        return {
            "applied": False,
            "reason": f"无 3-Agent 多数共识，分布 {dict(winner_votes)}",
            "consensus_winner": None,
            "majority_count": top_count,
            "agents": agent_outputs,
            "summary": f"⚠ Agent 分歧 ({dict(winner_votes)})，保留 MC 原值",
        }
    
    # 取多数派 Agent 的修正均值
    majority_adj = [a["adjustment_pp"] for a in agent_outputs
                     if a["predicted_winner"] == top_winner]
    avg_adj = sum(majority_adj) / len(majority_adj)
    
    # 应用方向：top_winner 是 team_a 则 +avg，是 team_b 则 -avg
    if top_winner == team_a:
        final_adj = avg_adj
    else:
        final_adj = -avg_adj
    
    summary_parts = [f"{top_count}/{len(agent_outputs)} Agent 推 {top_winner}",
                      f"平均修正 {avg_adj:+.1f}pp"]
    
    return {
        "applied": True,
        "final_adjustment_pp": round(final_adj, 2),
        "consensus_winner": top_winner,
        "majority_count": top_count,
        "total_agents": len(agent_outputs),
        "agents": agent_outputs,
        "summary": " · ".join(summary_parts),
    }


# ============ CLI 自测 ============
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--p", type=float, default=0.50, help="MC 给 A 队的胜率（0-1）")
    args = ap.parse_args()
    
    from utils.io import load_teams
    teams = load_teams()
    if args.a not in teams or args.b not in teams:
        print(f"unknown team: {args.a if args.a not in teams else args.b}")
        sys.exit(1)
    
    print(f"=== 关键节点 5-Agent 微调测试 ===")
    print(f"对阵: {args.a} vs {args.b}")
    print(f"MC 原值: {args.a} {args.p*100:.1f}% vs {args.b} {(1-args.p)*100:.1f}%")
    print()
    
    result = run_critical_node_swarm(args.a, args.b, args.p,
                                       teams[args.a], teams[args.b])
    print(json.dumps(result, ensure_ascii=False, indent=2))
