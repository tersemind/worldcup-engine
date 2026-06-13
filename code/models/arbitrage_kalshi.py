"""
Kalshi 单场套利信号生成器
==========================

复用 arbitrage.py 的 Kelly/EV/Grade 函数，差异在于：
- 数据源：match_bias.json（已含模型 P(H/D/A) vs Kalshi P(H/D/A) + delta_pp）
- 每场比赛产 3 个候选信号（主胜/平/客胜），各自独立计算
- 入口深链：Kalshi event page (kalshi.com/events/<event_ticker>)

输出: data/outputs/arbitrage_kalshi_signals.json

容错：完全吞异常（feedback_no_break_existing），任一场失败不影响其他场。
"""
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.io import DATA_OUTPUTS, save_output
from models.arbitrage import (
    kelly_fraction, implied_to_decimal_odds, expected_value,
    edge_score, confidence_grade,
)


# 球队中文名（与前端 zh() 一致即可，这里仅给 logging 用，前端会再 zh 一次）
def _build_signal(event_ticker: str, date: str, team_a: str, team_b: str,
                  side: str, side_label: str, model_p: float, market_p: float,
                  bankroll: float, min_edge_pp: float) -> Optional[dict]:
    """构造一条 Kalshi 单场套利信号；不满足阈值返回 None。

    Args:
        side: "home" | "draw" | "away" — 用于前端排版
        side_label: 显示文本，如 "Austria 胜" / "平" / "Jordan 胜"
    """
    if market_p <= 0 or model_p <= 0:
        return None

    edge = edge_score(model_p, market_p)
    # min_edge_pp=0 时不过滤，全量返回（用于"全部场次单场赔率"展示）
    # 信号 type 仍按 |edge| 与默认 BUY/SELL 阈值（3pp）判定
    if min_edge_pp > 0 and abs(edge) < min_edge_pp:
        return None

    decimal_odds = implied_to_decimal_odds(market_p)
    ev = expected_value(model_p, decimal_odds)
    kelly = kelly_fraction(model_p, decimal_odds, kelly_multiplier=0.25)
    grade = confidence_grade(edge, model_p)
    # NEUTRAL：|edge| < 3pp（默认 BUY/SELL 阈值）
    if abs(edge) < 3.0:
        signal_type = "NEUTRAL"
    else:
        signal_type = "BUY" if edge > 0 else "SELL"

    return {
        "event_ticker": event_ticker,
        "date": date,
        "match": f"{team_a} vs {team_b}",
        "team_a": team_a,
        "team_b": team_b,
        "side": side,
        "side_label": side_label,
        "type": signal_type,
        "model_prob": round(model_p * 100, 2),
        "market_prob": round(market_p * 100, 2),
        "edge_pp": round(edge, 2),
        "decimal_odds": round(decimal_odds, 2),
        "ev_per_dollar": round(ev, 4),
        "ev_pct": round(ev * 100, 2),
        "kelly_quarter": round(kelly * 100, 2),
        "suggested_bet": round(bankroll * kelly, 2),
        "grade": grade,
        "market_venue": "Kalshi",
        "market_url": f"https://kalshi.com/events/{event_ticker}",
    }


def generate_kalshi_signals(bias_file: str = "match_bias.json",
                            bankroll: float = 10000,
                            min_edge_pp: float = 3.0,
                            future_only: bool = True) -> list:
    """从 match_bias.json 生成 Kalshi 单场套利信号。

    Args:
        bias_file: match_bias_detector 的输出（已对齐 model vs kalshi）
        bankroll: 总资金（计算 Kelly 仓位）
        min_edge_pp: 最小偏差阈值；单场赔率方差大，默认 3.0（vs 夺冠盘 1.0）
        future_only: 仅产未来场次的信号（已结束比赛无意义）
    """
    path = DATA_OUTPUTS / bias_file
    if not path.exists():
        return []

    try:
        bias = json.load(open(path))
    except Exception:
        return []

    matches = bias.get("matches") or []
    today = datetime.now().strftime("%Y-%m-%d")

    signals = []
    for m in matches:
        try:
            date = m.get("date") or ""
            if future_only and date < today:
                continue

            ev_ticker = m.get("event_ticker")
            team_a = m.get("team_a")
            team_b = m.get("team_b")
            model = m.get("model") or {}
            kalshi = m.get("kalshi") or {}

            if not (ev_ticker and team_a and team_b and model and kalshi):
                continue

            # H / D / A 三个选项各算一遍
            for side, side_label, mkey in [
                ("home", f"{team_a} 胜", "p_win_a"),
                ("draw", "平局", "p_draw"),
                ("away", f"{team_b} 胜", "p_win_b"),
            ]:
                model_p = model.get(mkey)
                market_p = kalshi.get(mkey)
                if model_p is None or market_p is None:
                    continue
                sig = _build_signal(ev_ticker, date, team_a, team_b,
                                    side, side_label, model_p, market_p,
                                    bankroll, min_edge_pp)
                if sig is not None:
                    signals.append(sig)
        except Exception:
            # 单场出错不影响其他场
            continue

    # 按 |edge| 降序
    signals.sort(key=lambda x: -abs(x["edge_pp"]))
    return signals


def main(bankroll: float = 10000):
    signals = generate_kalshi_signals(bankroll=bankroll)
    buys = [s for s in signals if s["type"] == "BUY"]
    sells = [s for s in signals if s["type"] == "SELL"]

    print("=" * 100)
    print(f"💰 Kalshi 单场套利信号扫描器")
    print(f"   假设总资金 ${bankroll:,.0f} · Quarter Kelly (1/4) · 仅产未来场次 · min_edge=3pp")
    print("=" * 100)

    print(f"\n🟢 BUY 信号（模型 > 市场）：{len(buys)} 个")
    print(f"  {'Grade':<6} {'日期':<11} {'比赛':<30} {'选项':<18} {'模型%':<7} {'市场%':<7} {'Edge':<8} {'赔率':<7} {'EV%':<8} {'下注':<10}")
    print("  " + "-" * 130)
    for s in buys[:15]:
        match_str = f"{s['team_a']} vs {s['team_b']}"[:28]
        print(f"  {s['grade']:<6} {s['date']:<11} {match_str:<30} {s['side_label']:<18} "
              f"{s['model_prob']:>5.1f}%  {s['market_prob']:>5.1f}%  "
              f"+{s['edge_pp']:>4.1f}pp  {s['decimal_odds']:>5.2f}x  "
              f"{s['ev_pct']:>+6.1f}%  ${s['suggested_bet']:>7,.0f}")

    print(f"\n🔴 AVOID 信号（模型 < 市场，市场高估）：{len(sells)} 个")
    for s in sells[:5]:
        match_str = f"{s['team_a']} vs {s['team_b']}"[:28]
        print(f"  {s['grade']:<6} {s['date']:<11} {match_str:<30} {s['side_label']:<18} "
              f"{s['model_prob']:>5.1f}% / {s['market_prob']:>5.1f}%  {s['edge_pp']:>+5.1f}pp")

    # 保存
    try:
        from pathlib import Path as _P
        bias_path = DATA_OUTPUTS / "match_bias.json"
        kalshi_path = DATA_OUTPUTS / "kalshi_match_odds.json"
        freshness = {
            "trigger": "manual",
            "refreshed_at": datetime.now().isoformat(timespec="seconds"),
            "bias_snapshot_ts": datetime.fromtimestamp(bias_path.stat().st_mtime).isoformat(timespec="seconds") if bias_path.exists() else None,
            "kalshi_snapshot_ts": datetime.fromtimestamp(kalshi_path.stat().st_mtime).isoformat(timespec="seconds") if kalshi_path.exists() else None,
        }

        save_output("arbitrage_kalshi_signals.json", {
            "bankroll": bankroll,
            "n_signals": len(signals),
            "n_buy": len(buys),
            "n_sell": len(sells),
            "signals": signals,
            "market": {
                "name": "FIFA World Cup 2026 — Single Game Winner (H/D/A)",
                "venue": "Kalshi",
                "series_ticker": "KXWCGAME",
                "series_url": "https://kalshi.com/markets/kxwcgame",
                "note": "单场胜平负盘：每场 3 个独立 Yes/No 合约；信号源自 match_bias（模型 vs Kalshi）",
            },
            "_freshness": freshness,
        })
        print(f"\n✅ 已保存到 data/outputs/arbitrage_kalshi_signals.json ({len(signals)} 信号)")
    except Exception as e:
        print(f"⚠ 保存失败: {type(e).__name__}: {e}")


if __name__ == "__main__":
    bankroll = float(sys.argv[1]) if len(sys.argv) > 1 else 10000
    main(bankroll)
