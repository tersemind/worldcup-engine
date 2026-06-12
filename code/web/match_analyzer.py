"""
LLM 单场分析器（深度分析报告体例 体例）

输出严格按 参考报告 3.7.8 节"葡萄牙小组赛对阵深度分析"标准：
  - 战术对位（对方风格 vs 我方应对）
  - 关键 X 因子（核心球员/定位球/特殊战术）
  - 环境/海拔/温度影响（如适用）
  - 胜率/净胜球区间确认或修正
  - 数据源标注
"""
from __future__ import annotations
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from llm import get_default_client, LLMUnavailable
from llm.client import LLMClient

logger = logging.getLogger("worldcup.match_analyzer")


SYSTEM_PROMPT = """你是世界杯单场比赛分析师，严格按 参考报告 3.7.8 节"葡萄牙小组赛对阵深度分析"格式输出。

# 必须包含的 5 个维度
1. tactical_matchup: 战术对位（对方风格 vs 我方应对，3-5 句话）
2. x_factor: 关键 X 因子（最多 2 个，如某球员、定位球、特殊战术）
3. environment: 环境影响（高温/海拔/旅行疲劳；无显著则填"中性"）
4. confidence_check: 对量化模型给出的胜率/净胜球**确认或修正**
5. risk_signals: 2-3 条可能改变比赛走向的风险点

# 严格约束
- 不得编造伤病或事件
- 引用公开赛绩、教练叙事、球员状态
- 输出必须为合法 JSON
- 中文输出，简洁专业
"""

OUTPUT_SCHEMA = {
    "tactical_matchup": "<3-5 句战术对位描述>",
    "x_factor": [
        {"name": "<球员名/战术名>", "impact": "<影响描述>"}
    ],
    "environment": "<环境影响 / '中性'>",
    "confidence_check": {
        "model_winner": "<量化模型预测的赢方>",
        "model_win_rate": "<量化预测胜率>",
        "llm_agree": "<bool>",
        "llm_adjustment": "<如不同意，给出修正后的胜率区间>",
        "rationale": "<修正理由 <=80字>"
    },
    "risk_signals": ["<风险1>", "<风险2>", "<风险3>"]
}


def analyze_match_with_llm(team_a: str, team_b: str,
                             quick_pred: Dict[str, Any],
                             team_a_info: Dict, team_b_info: Dict,
                             client: Optional[LLMClient] = None
                             ) -> Dict[str, Any]:
    """
    用 LLM 分析单场比赛
    
    Args:
        team_a, team_b: 球队名
        quick_pred: tournament_api._quick_match_preview 的输出
        team_a_info, team_b_info: teams.json 中的球队字典
    """
    try:
        c = client or get_default_client()
    except LLMUnavailable as e:
        return {"available": False, "error": f"LLM 不可用: {e}"}

    # 拼 user prompt
    user = f"""# 比赛
{team_a} vs {team_b}

# 量化模型预测
- 胜方: {quick_pred.get('predicted_winner')}
- 胜率: {quick_pred.get('winner_confidence_pct')}%
- 期望进球: {team_a} {quick_pred.get('lambda_a')}, {team_b} {quick_pred.get('lambda_b')}
- Top 3 比分: {[s['score'] + ' (' + str(s['prob_pct']) + '%)' for s in quick_pred.get('top_scorelines', [])]}

# {team_a} 数据
- Elo: {team_a_info.get('elo')}
- xG_for/against: {team_a_info.get('xg_for')}/{team_a_info.get('xg_against')}
- xT_per_90: {team_a_info.get('xt_per_90')}
- 身价: €{team_a_info.get('squad_value_m_eur')}M
- 大洲: {team_a_info.get('confederation')}

# {team_b} 数据
- Elo: {team_b_info.get('elo')}
- xG_for/against: {team_b_info.get('xg_for')}/{team_b_info.get('xg_against')}
- xT_per_90: {team_b_info.get('xt_per_90')}
- 身价: €{team_b_info.get('squad_value_m_eur')}M
- 大洲: {team_b_info.get('confederation')}

# 任务
按 深度分析报告体例体例分析，输出 5 维度 JSON。

# 输出 JSON Schema
{json.dumps(OUTPUT_SCHEMA, ensure_ascii=False, indent=2)}
"""

    try:
        text = c.chat(system=SYSTEM_PROMPT, user=user,
                       json_mode=True, max_tokens=4000, temperature=0.3)
        data = LLMClient.extract_json(text)
        data["available"] = True
        data["model"] = c.model
        return data
    except Exception as e:
        logger.exception(f"LLM 分析失败 {team_a} vs {team_b}")
        return {"available": False, "error": str(e)}


# CLI 自测
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    args = ap.parse_args()

    from web.tournament_api import _quick_match_preview, _load_teams
    teams = _load_teams()
    if args.a not in teams or args.b not in teams:
        print(f"unknown teams: {args.a if args.a not in teams else args.b}")
        sys.exit(1)

    pred = _quick_match_preview(args.a, args.b)
    print("量化预测:")
    print(f"  {pred['predicted_winner']} 胜率 {pred['winner_confidence_pct']}%")
    print(f"  最可能比分: {pred['top_scorelines'][0]['score']}")
    print()
    print("LLM 分析中...")
    analysis = analyze_match_with_llm(args.a, args.b, pred,
                                        teams[args.a], teams[args.b])
    print(json.dumps(analysis, ensure_ascii=False, indent=2))
