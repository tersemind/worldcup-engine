"""
单场偏差检测器（72 场赛程级）
=================================
对比"我们的集成模型" vs "Kalshi 真实市场赔率"，识别每场的偏差与爆冷信号。

数据来源：
  - 模型：tournament_api._quick_match_preview() 输出（Elo + Poisson 集成）
  - 市场：data/outputs/kalshi_match_odds.json（Kalshi 公共 API 实时抓取）

输出：
  - data/outputs/match_bias.json
    格式: {
      "matches": [
        {
          "date", "team_a", "team_b",
          "model": {p_win_a, p_draw, p_win_b},
          "kalshi": {p_win_a, p_draw, p_win_b, overround_pct, data_quality},
          "delta": {
            "p_win_a_pp": +5.2,  # 模型相对市场的偏差
            "p_draw_pp": -1.0,
            "p_win_b_pp": -4.2,
            "max_abs_pp": 5.2
          },
          "level": "L1/L2/L3/L4",
          "upset_signal": {
            "detected": True/False,
            "team": "Iran",         # 弱队名字
            "model_pct": 38,        # 模型给弱队的胜率
            "market_pct": 22,       # 市场给弱队的胜率
            "delta_pp": +16,        # 弱队被市场低估的幅度
            "level": "L3"
          }
        }
      ],
      "summary": {
        "n_compared", "n_upset_signals",
        "top_upsets": [...]   # 爆冷信号 Top N
      }
    }

偏差级别（沿用 market_bias_detector.py 标准）：
  - L1 < 3pp
  - L2 3-6pp
  - L3 6-10pp
  - L4 >= 10pp

爆冷信号判定：
  - 弱队（市场胜率 < 50% 的一方）的胜率被模型显著高估（model - market >= 5pp）
  - 即"市场觉得它没机会，但模型说它有机会" → 潜在爆冷
"""
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

ROOT = Path(__file__).parent.parent.parent
KALSHI_PATH = ROOT / "data" / "outputs" / "kalshi_match_odds.json"
OUT_PATH = ROOT / "data" / "outputs" / "match_bias.json"


def _classify(abs_max_pp: float) -> str:
    if abs_max_pp >= 10:
        return "L4"
    if abs_max_pp >= 6:
        return "L3"
    if abs_max_pp >= 3:
        return "L2"
    return "L1"


def _level_text(level: str) -> str:
    return {"L1": "轻度", "L2": "中度", "L3": "高度", "L4": "极端"}.get(level, "?")


def detect_match_bias(save: bool = True) -> Dict[str, Any]:
    """主入口：对所有 Kalshi 有覆盖的场次跑偏差检测"""
    # 加载 Kalshi 数据
    if not KALSHI_PATH.exists():
        return {"error": f"{KALSHI_PATH} 不存在，请先跑 fetch_kalshi_match_odds.py"}
    kalshi = json.load(open(KALSHI_PATH))
    
    # 调模型预测（避免循环导入：lazy import）
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from web.tournament_api import _quick_match_preview
    
    results = []
    upset_signals = []
    
    for k_match in kalshi.get("matches", []):
        team_a = k_match["team_a"]
        team_b = k_match["team_b"]
        date = k_match["date"]
        
        # 数据质量过滤：overround 异常大的场次跳过
        overround = k_match.get("raw_overround_pct", 0)
        if abs(overround) >= 8:
            continue  # 数据不完整
        
        # 模型预测
        try:
            model = _quick_match_preview(team_a, team_b)
            if "error" in model:
                continue
        except Exception:
            continue
        
        # 三维偏差（模型 - 市场，单位 pp）
        delta_a = (model["p_win_a"] - k_match["kalshi_p_win_a"]) * 100
        delta_d = (model["p_draw"]  - k_match["kalshi_p_draw"]) * 100
        delta_b = (model["p_win_b"] - k_match["kalshi_p_win_b"]) * 100
        max_abs = max(abs(delta_a), abs(delta_d), abs(delta_b))
        level = _classify(max_abs)
        
        entry = {
            "date": date,
            "event_ticker": k_match.get("event_ticker"),
            "team_a": team_a,
            "team_b": team_b,
            "model": {
                "p_win_a": round(model["p_win_a"], 4),
                "p_draw":  round(model["p_draw"], 4),
                "p_win_b": round(model["p_win_b"], 4),
                "predicted_winner": model["predicted_winner"],
            },
            "kalshi": {
                "p_win_a": k_match["kalshi_p_win_a"],
                "p_draw":  k_match["kalshi_p_draw"],
                "p_win_b": k_match["kalshi_p_win_b"],
                "overround_pct": overround,
                "volume_24h": k_match.get("total_volume_24h", 0),
            },
            "delta_pp": {
                "p_win_a": round(delta_a, 2),
                "p_draw":  round(delta_d, 2),
                "p_win_b": round(delta_b, 2),
                "max_abs": round(max_abs, 2),
            },
            "level": level,
            "level_text": _level_text(level),
        }
        
        # ===== 爆冷信号（A 定义：模型负方视角）=====
        # 爆冷 = "模型预测的负方胜率明显大于市场给该负方的胜率"
        # 三个条件全部满足：
        #   1. 模型负方（model 胜率较低者）的模型胜率 ≥ 25%（模型给了实质机会）
        #   2. 市场给该队胜率 ≥ 5%（市场赔率有效，不是 0% 无成交）
        #   3. 模型 - 市场 ≥ 8pp（差距实质性）
        # 不要求"模型胜方 == 市场胜方"——允许模型完全反市场
        if model["p_win_a"] < model["p_win_b"]:
            model_loser = team_a
            model_loser_pct = model["p_win_a"] * 100
            market_loser_pct = k_match["kalshi_p_win_a"] * 100
            loser_delta = delta_a  # 已是 (model - market) * 100
        else:
            model_loser = team_b
            model_loser_pct = model["p_win_b"] * 100
            market_loser_pct = k_match["kalshi_p_win_b"] * 100
            loser_delta = delta_b
        
        UPSET_MIN_MODEL_PCT = 25    # 模型给负方 ≥25%
        UPSET_MIN_MARKET_PCT = 5     # 市场赔率有效
        UPSET_MIN_DELTA_PP = 8       # 模型 - 市场 ≥ 8pp
        
        is_upset = (
            model_loser_pct >= UPSET_MIN_MODEL_PCT
            and market_loser_pct >= UPSET_MIN_MARKET_PCT
            and loser_delta >= UPSET_MIN_DELTA_PP
        )
        
        # 模型胜方一致性（仅作展示，不影响触发）
        if model["p_win_a"] > model["p_win_b"]:
            model_winner = team_a
        else:
            model_winner = team_b
        market_winner = team_a if k_match["kalshi_p_win_a"] > k_match["kalshi_p_win_b"] else team_b
        winners_consistent = (model_winner == market_winner)
        
        if is_upset:
            upset_level = _classify(abs(loser_delta))
            upset = {
                "detected": True,
                "team": model_loser,                # 模型负方
                "opponent": team_b if model_loser == team_a else team_a,
                "date": date,
                "model_pct": round(model_loser_pct, 1),
                "market_pct": round(market_loser_pct, 1),
                "delta_pp": round(loser_delta, 2),
                "level": upset_level,
                "winners_consistent": winners_consistent,  # 模型胜方与市场胜方是否一致
            }
            entry["upset_signal"] = upset
            upset_signals.append({**upset, "match": f"{team_a} vs {team_b}"})
        else:
            reasons = []
            if model_loser_pct < UPSET_MIN_MODEL_PCT:
                reasons.append(f"模型给负方仅 {model_loser_pct:.0f}%（需≥{UPSET_MIN_MODEL_PCT}%）")
            if market_loser_pct < UPSET_MIN_MARKET_PCT:
                reasons.append(f"市场赔率无效（仅 {market_loser_pct:.0f}%）")
            if loser_delta < UPSET_MIN_DELTA_PP:
                reasons.append(f"差距不足（{loser_delta:.1f}pp < {UPSET_MIN_DELTA_PP}pp）")
            entry["upset_signal"] = {
                "detected": False,
                "model_loser": model_loser,
                "model_loser_pct": round(model_loser_pct, 1),
                "market_loser_pct": round(market_loser_pct, 1),
                "loser_delta_pp": round(loser_delta, 2),
                "reason": "; ".join(reasons) if reasons else "—",
            }
        
        results.append(entry)
    
    # 按偏差倒序
    results.sort(key=lambda e: -e["delta_pp"]["max_abs"])
    
    # 爆冷信号按 delta 倒序
    upset_signals.sort(key=lambda u: -u["delta_pp"])
    
    output = {
        "_fetched_at": datetime.now().isoformat(timespec="seconds"),
        "_method": "分级规范 偏差分级 · Kalshi 公共 API 实时赔率 · Elo+Poisson 集成模型",
        "n_compared": len(results),
        "n_upset_signals": len(upset_signals),
        "summary_by_level": {
            "L1": sum(1 for r in results if r["level"] == "L1"),
            "L2": sum(1 for r in results if r["level"] == "L2"),
            "L3": sum(1 for r in results if r["level"] == "L3"),
            "L4": sum(1 for r in results if r["level"] == "L4"),
        },
        "top_upsets": upset_signals[:10],
        "matches": results,
    }
    
    if save:
        OUT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    return output


# ===== CLI 自测 =====
if __name__ == "__main__":
    print("=== 72 场赛程级单场偏差检测 ===\n")
    result = detect_match_bias()
    if "error" in result:
        print(f"❌ {result['error']}")
        sys.exit(1)
    
    print(f"已对比场次: {result['n_compared']}")
    print(f"爆冷信号: {result['n_upset_signals']}")
    print(f"按级别分布: {result['summary_by_level']}")
    print()
    
    print("=== Top 10 偏差最大 ===")
    print(f"{'#':<3} {'对阵':<32} {'方向':<10} {'模型':<24} {'市场':<24} {'级别'}")
    print("-" * 110)
    for i, r in enumerate(result['matches'][:10], 1):
        m = r['model']; k = r['kalshi']; d = r['delta_pp']
        model_s = f"{m['p_win_a']*100:>2.0f}/{m['p_draw']*100:>2.0f}/{m['p_win_b']*100:>2.0f}"
        kalshi_s = f"{k['p_win_a']*100:>2.0f}/{k['p_draw']*100:>2.0f}/{k['p_win_b']*100:>2.0f}"
        direction = f"max {d['max_abs']:+.1f}pp"
        print(f"{i:<3} {r['team_a'] + ' vs ' + r['team_b']:<32} {direction:<10} {model_s:<24} {kalshi_s:<24} {r['level']}")
    
    print()
    print("=== 爆冷信号 Top 10（弱队被市场低估）===")
    print(f"{'#':<3} {'弱队':<14} {'对手':<14} {'模型%':>7} {'市场%':>7} {'Δpp':>7} {'级别':<5} {'日期'}")
    print("-" * 80)
    for i, u in enumerate(result['top_upsets'], 1):
        print(f"{i:<3} {u['team']:<14} {u['opponent']:<14} {u['model_pct']:>6.1f}% {u['market_pct']:>6.1f}% "
              f"{u['delta_pp']:>+6.2f}  {u['level']:<5} {u['date']}")
