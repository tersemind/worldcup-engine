"""
爆冷 LLM 归因引擎
==================
对规则粗筛出的"候选爆冷场"做 LLM 终审：综合模型预测、Kalshi 赔率、最新新闻，
输出 5 个标签之一 + 关键证据 + 置信度。

5 个标签（对齐 参考规范 §3.6.10 偏差归因体例）：
  🚀 真爆冷潜力 (true_upset)     — LLM 找到独立证据支持弱队
  🤔 信息差 (info_gap)             — 市场未充分定价已公开信息
  ⚖ 模型保守 (model_too_aggressive)— 模型 + 市场都对，规则误触发
  ❌ 市场对 (market_correct)       — LLM 同意市场，模型可能高估
  ⚠ 数据噪声 (data_noise)          — Kalshi 流动性低/异常

输入：
  - (team_a, team_b, date)
  - 模型胜率三元组、Kalshi 胜率三元组、偏差 pp
  - 双方近 3 月战绩、xG、伤病（从 teams.json）
  - 可选：WebSearch 拉的最新新闻

输出 JSON：
  {
    "label": "true_upset",
    "label_text": "🚀 真爆冷潜力",
    "confidence": "中",
    "key_evidence": ["...","...","..."],
    "rationale": "...",
    "suggested_action": "..."
  }

使用 WebSearch 的策略（结合 LLM）：
  1. 跑批模式：默认不联网（避免限流），用 teams.json 数据
  2. 单场深度分析：API 触发 use_search=True 时联网拉 2 个查询
"""
from __future__ import annotations
import json
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

sys.path.insert(0, str(Path(__file__).parent.parent))
from llm import get_default_client, LLMUnavailable
from llm.client import LLMClient

logger = logging.getLogger("worldcup.upset_llm")

ROOT = Path(__file__).parent.parent.parent


# ============ 标签定义 ============
LABELS = {
    "true_upset": {"icon": "🚀", "text": "真爆冷潜力", "color": "#4ade80"},
    "info_gap": {"icon": "🤔", "text": "信息差", "color": "#fbbf24"},
    "model_too_aggressive": {"icon": "⚖", "text": "模型保守", "color": "#888"},
    "market_correct": {"icon": "❌", "text": "市场对", "color": "#fb923c"},
    "data_noise": {"icon": "⚠", "text": "数据噪声", "color": "#f87171"},
}


SYSTEM_PROMPT = """你是足球预测市场分析师，专注于识别"模型 vs 市场"分歧的真实原因。

# 任务
针对一场比赛，综合以下信息判断"模型给弱队的胜率高于市场"是真爆冷信号，还是误报。

# 输出 5 个标签之一（必选其一）
- "true_upset"          : 你找到独立证据支持弱队（如战术相克、强队伤病、状态低迷）
- "info_gap"            : 市场未充分定价已公开信息（教练换人、阵容轮换、心态变化）
- "model_too_aggressive": 模型对弱队估计过高，市场是对的，但模型也不算离谱
- "market_correct"      : 市场明显对，模型高估弱队（建议跟市场走）
- "data_noise"          : Kalshi 流动性低或赔率异常，数据不可信

# 关键考虑
- 强弱悬殊场（弱队 Elo 比强队低 200+）通常市场是对的
- 战术相克（如反击型 vs 高位逼抢）可以产生真爆冷
- 强队连续大赛后疲劳 / 关键球员伤停 / 教练调整都是 info_gap
- 模型如果在多场比赛都高估弱队，可能是系统性偏差（标 model_too_aggressive）

# 输出 JSON（必须严格返回，无解释文字）
{
  "label": "<5个之一>",
  "confidence": "高/中/低",
  "key_evidence": ["证据1", "证据2", "证据3"],
  "rationale": "<不超过80字的核心理由>",
  "suggested_action": "<跟模型/跟市场/观望，不超过30字>"
}"""


def build_context(team_a: str, team_b: str, date: str,
                   model: Dict, kalshi: Dict, delta_pp: Dict,
                   weak_team: str,
                   team_a_info: Dict, team_b_info: Dict,
                   extra_news: Optional[List[str]] = None) -> str:
    """构造给 LLM 的用户提示（结构化数据 + 可选新闻）"""
    # 弱队识别
    weak_info = team_a_info if weak_team == team_a else team_b_info
    strong_info = team_b_info if weak_team == team_a else team_a_info
    strong_team = team_b if weak_team == team_a else team_a
    
    parts = [
        f"# 比赛",
        f"{team_a} vs {team_b}  ({date})",
        "",
        f"# 模型预测（Elo + Poisson 集成）",
        f"- {team_a} 胜: {model['p_win_a']*100:.0f}%",
        f"- 平局: {model['p_draw']*100:.0f}%",
        f"- {team_b} 胜: {model['p_win_b']*100:.0f}%",
        "",
        f"# Kalshi 真实市场赔率",
        f"- {team_a} 胜: {kalshi['p_win_a']*100:.0f}% (24h vol ${kalshi.get('volume_24h', 0):.0f})",
        f"- 平局: {kalshi['p_draw']*100:.0f}%",
        f"- {team_b} 胜: {kalshi['p_win_b']*100:.0f}%",
        "",
        f"# 偏差分析",
        f"- 关注弱队: {weak_team}",
        f"- 模型给 {weak_team}: {(model['p_win_a'] if weak_team==team_a else model['p_win_b'])*100:.0f}%",
        f"- 市场给 {weak_team}: {(kalshi['p_win_a'] if weak_team==team_a else kalshi['p_win_b'])*100:.0f}%",
        f"- 差距: 模型相对市场上调 {abs((delta_pp['p_win_a'] if weak_team==team_a else delta_pp['p_win_b'])):.1f}pp",
        "",
        f"# {weak_team}（弱队）基础数据",
        f"- Elo: {weak_info.get('elo')}",
        f"- xG_for / xG_against: {weak_info.get('xg_for')} / {weak_info.get('xg_against')}",
        f"- xT/90: {weak_info.get('xt_per_90', '?')}",
        f"- 身价: €{weak_info.get('squad_value_m_eur', '?')}M",
        f"- 大洲: {weak_info.get('confederation', '?')}",
        f"- 市场夺冠概率（综合）: {weak_info.get('market_implied', 0)*100:.1f}%",
        "",
        f"# {strong_team}（强队）基础数据",
        f"- Elo: {strong_info.get('elo')}",
        f"- xG_for / xG_against: {strong_info.get('xg_for')} / {strong_info.get('xg_against')}",
        f"- 身价: €{strong_info.get('squad_value_m_eur', '?')}M",
        f"- 市场夺冠概率: {strong_info.get('market_implied', 0)*100:.1f}%",
    ]
    
    if extra_news:
        parts.append("")
        parts.append(f"# 实时新闻摘要（最近 7 天）")
        for i, n in enumerate(extra_news[:5], 1):
            parts.append(f"- {n}")
    
    parts.extend([
        "",
        f"# 你的任务",
        f"基于以上数据，判断「模型给 {weak_team} 胜率 vs 市场偏差」属于哪个标签。",
        f"严格 JSON 输出（5 个字段：label, confidence, key_evidence, rationale, suggested_action）。",
    ])
    
    return "\n".join(parts)


def analyze_upset(team_a: str, team_b: str, date: str,
                   model: Dict, kalshi: Dict, delta_pp: Dict,
                   weak_team: str,
                   team_a_info: Dict, team_b_info: Dict,
                   extra_news: Optional[List[str]] = None,
                   client: Optional[LLMClient] = None
                   ) -> Dict[str, Any]:
    """主入口：调 LLM 跑一次归因"""
    try:
        c = client or get_default_client()
    except LLMUnavailable as e:
        return {"error": f"LLM 不可用: {e}", "fallback": "rules_only"}
    
    user_msg = build_context(team_a, team_b, date, model, kalshi, delta_pp,
                              weak_team, team_a_info, team_b_info, extra_news)
    
    # 重试 2 次，覆盖偶发空返回（LLM 路由层不稳）
    data = None
    last_err = None
    for attempt in range(2):
        try:
            temp = 0.3 if attempt == 0 else 0.7
            text = c.chat(system=SYSTEM_PROMPT, user=user_msg,
                           json_mode=True, max_tokens=600, temperature=temp)
            data = LLMClient.extract_json(text)
            if data and "label" in data:
                break
            last_err = f"字段缺失: {data}"
        except Exception as e:
            last_err = str(e)
            data = None
    
    if not data or "label" not in data:
        logger.warning(f"LLM 重试 2 次仍失败 {team_a} vs {team_b}: {last_err}")
        return {"error": f"重试 2 次仍失败: {last_err}"}
    
    # 校验 label
    label = data.get("label", "").strip()
    if label not in LABELS:
        # 容错：找最接近的
        for k in LABELS:
            if k in label.lower() or label.lower() in k:
                label = k
                break
        else:
            label = "data_noise"  # 实在不识别就归为噪声
    
    meta = LABELS[label]
    return {
        "label": label,
        "label_text": f"{meta['icon']} {meta['text']}",
        "label_color": meta["color"],
        "confidence": data.get("confidence", "中"),
        "key_evidence": data.get("key_evidence", []),
        "rationale": data.get("rationale", "")[:200],
        "suggested_action": data.get("suggested_action", "")[:80],
        "_used_search": extra_news is not None,
    }


# ============ CLI 自测 ============
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    args = ap.parse_args()
    
    # 加载数据
    teams = json.load(open(ROOT / "data" / "raw" / "teams.json"))["teams"]
    mb = json.load(open(ROOT / "data" / "outputs" / "match_bias.json"))
    
    # 找该场
    match = None
    for m in mb["matches"]:
        if (m["team_a"] == args.a and m["team_b"] == args.b) or \
           (m["team_a"] == args.b and m["team_b"] == args.a):
            match = m
            break
    if not match:
        print(f"❌ 未找到 {args.a} vs {args.b} 的偏差记录")
        sys.exit(1)
    
    # 识别弱队
    if match["kalshi"]["p_win_a"] < match["kalshi"]["p_win_b"]:
        weak = match["team_a"]
    else:
        weak = match["team_b"]
    
    print(f"=== {match['team_a']} vs {match['team_b']} ({match['date']}) ===")
    print(f"模型: {match['model']['p_win_a']*100:.0f}/{match['model']['p_draw']*100:.0f}/{match['model']['p_win_b']*100:.0f}")
    print(f"市场: {match['kalshi']['p_win_a']*100:.0f}/{match['kalshi']['p_draw']*100:.0f}/{match['kalshi']['p_win_b']*100:.0f}")
    print(f"弱队: {weak}")
    print()
    print("跑 LLM 分析...")
    
    result = analyze_upset(
        match["team_a"], match["team_b"], match["date"],
        match["model"], match["kalshi"], match["delta_pp"],
        weak,
        teams[match["team_a"]], teams[match["team_b"]],
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
