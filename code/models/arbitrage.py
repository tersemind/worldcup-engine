"""
Polymarket 套利信号生成器（#18）

核心逻辑：
- 模型概率 vs 市场概率 → 偏差 (edge)
- 用 Kelly 公式计算最优仓位
- 输出可执行下注建议（含止损线）

Kelly 公式：
  f* = (bp - q) / b
  其中 b = 赔率 - 1（净收益率）, p = 真实概率, q = 1-p

由于真实 p 不可知，使用模型概率作为近似：
  f* = (model_p × decimal_odds - 1) / (decimal_odds - 1)
  
保守版：仅取 Kelly 的 1/4（避免过度自信，行业惯例）
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.io import DATA_OUTPUTS, load_teams


def kelly_fraction(model_prob: float, decimal_odds: float, kelly_multiplier: float = 0.25) -> float:
    """
    计算 Kelly 仓位比例
    
    Args:
        model_prob: 模型估计的真实概率（0-1）
        decimal_odds: 十进制赔率（如 5.0 表示赢可拿本金 5 倍）
        kelly_multiplier: Kelly 折扣（0.25 = Quarter Kelly，行业稳健做法）
    
    Returns:
        建议下注比例（占总资金的百分比）
    """
    if model_prob <= 0 or decimal_odds <= 1:
        return 0.0
    
    b = decimal_odds - 1
    p = model_prob
    q = 1 - p
    
    full_kelly = (b * p - q) / b
    
    # 负 Kelly = 不应下注
    if full_kelly <= 0:
        return 0.0
    
    return full_kelly * kelly_multiplier


def implied_to_decimal_odds(implied_prob: float, vig: float = 0.0) -> float:
    """市场隐含概率 → 十进制赔率（含庄家抽水）"""
    if implied_prob <= 0:
        return float("inf")
    # decimal_odds = 1 / (implied_prob × (1+vig))
    return 1.0 / (implied_prob * (1 + vig))


def expected_value(model_prob: float, decimal_odds: float) -> float:
    """单注期望收益率（每 $1 注的预期 ROI）"""
    if decimal_odds <= 0:
        return 0.0
    # 赢的概率 × 净收益 - 输的概率 × 1
    return model_prob * (decimal_odds - 1) - (1 - model_prob) * 1.0


def edge_score(model_prob: float, market_prob: float) -> float:
    """偏差强度 (pp)"""
    return (model_prob - market_prob) * 100


def confidence_grade(edge_pp: float, model_prob: float) -> str:
    """根据偏差幅度和模型概率给出推荐等级"""
    if edge_pp >= 5.0:
        if model_prob >= 0.10:
            return "S"  # 强买入
        else:
            return "A"
    elif edge_pp >= 3.0:
        return "A" if model_prob >= 0.05 else "B"
    elif edge_pp >= 1.5:
        return "B"
    elif edge_pp >= 0.5:
        return "C"
    else:
        return "-"


def _load_polymarket_market_links() -> tuple:
    """读 external_predictions.json 拿每队的 Polymarket market slug，构造深链入口。

    Returns:
        (team -> market_url, event_url) — 全包 try/except，失败返 ({}, fallback_url)
    """
    fallback_event = "https://polymarket.com/event/world-cup-winner"
    try:
        from utils.io import DATA_RAW
        ext_path = DATA_RAW / "external_predictions.json"
        if not ext_path.exists():
            return {}, fallback_event
        ext = json.load(open(ext_path))
        poly = (ext.get("models") or {}).get("polymarket") or {}
        slugs = poly.get("market_slugs") or {}
        event_url = poly.get("event_url") or fallback_event
        return ({t: f"https://polymarket.com/market/{s}" for t, s in slugs.items() if s},
                event_url)
    except Exception:
        return {}, fallback_event


def generate_signals(synth_file: str = "synthesizer_report.json",
                     bankroll: float = 10000,
                     min_edge_pp: float = 1.0) -> list:
    """
    生成套利信号

    Args:
        synth_file: 综合预测输出文件
        bankroll: 总资金（计算建议下注金额）
        min_edge_pp: 最小偏差阈值（pp），低于不输出
    """
    path = DATA_OUTPUTS / synth_file
    with open(path, "r") as f:
        report = json.load(f)

    # 加载市场深链（吞异常，没有也不影响信号计算）
    team_market_urls, event_url = _load_polymarket_market_links()

    signals = []
    for team, data in report.items():
        model_p = data["final_probability"] / 100  # 转为小数
        market_p = data["market_implied"] / 100
        
        if market_p <= 0 or model_p <= 0:
            continue
        
        edge = edge_score(model_p, market_p)
        decimal_odds = implied_to_decimal_odds(market_p)
        ev = expected_value(model_p, decimal_odds)
        kelly = kelly_fraction(model_p, decimal_odds, kelly_multiplier=0.25)
        grade = confidence_grade(edge, model_p)
        
        if abs(edge) < min_edge_pp:
            continue
        
        signal_type = "BUY" if edge > 0 else "SELL"
        
        signals.append({
            "team": team,
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
            "ci_lower": data.get("ci_lower"),
            "ci_upper": data.get("ci_upper"),
            # 市场上下文 & 深链入口
            "market_name": "2026 FIFA World Cup Winner",
            "market_venue": "Polymarket",
            "market_url": team_market_urls.get(team, event_url),
            "event_url": event_url,
        })
    
    # 按偏差幅度绝对值排序
    signals.sort(key=lambda x: -abs(x["edge_pp"]))
    return signals


def print_signals(signals: list, bankroll: float = 10000):
    """打印套利信号报告"""
    print("=" * 100)
    print(f"💰 Polymarket 套利信号扫描器（基于模型 vs 市场偏差）")
    print(f"   假设总资金 ${bankroll:,.0f} · Quarter Kelly (1/4)")
    print("=" * 100)
    
    buy_signals = [s for s in signals if s["type"] == "BUY"]
    sell_signals = [s for s in signals if s["type"] == "SELL"]
    
    # 买入信号
    print(f"\n🟢 BUY 信号（模型 > 市场，做多）：{len(buy_signals)} 个")
    print(f"  {'Grade':<6} {'Team':<14} {'模型%':<7} {'市场%':<7} {'Edge':<8} {'赔率':<7} {'EV%':<7} {'Kelly%':<8} {'建议下注':<12}")
    print("  " + "-" * 90)
    for s in buy_signals[:15]:
        print(f"  {s['grade']:<6} {s['team']:<14} {s['model_prob']:>5.1f}%  {s['market_prob']:>5.1f}%  "
              f"+{s['edge_pp']:>4.1f}pp  {s['decimal_odds']:>5.1f}x  {s['ev_pct']:>+5.1f}%  "
              f"{s['kelly_quarter']:>5.2f}%   ${s['suggested_bet']:>8,.0f}")
    
    # 卖出/避开信号
    print(f"\n🔴 AVOID 信号（模型 < 市场，市场高估）：{len(sell_signals)} 个")
    print(f"  {'Grade':<6} {'Team':<14} {'模型%':<7} {'市场%':<7} {'Edge':<8} {'赔率':<7} {'EV%':<7}")
    print("  " + "-" * 70)
    for s in sell_signals[:10]:
        print(f"  {s['grade']:<6} {s['team']:<14} {s['model_prob']:>5.1f}%  {s['market_prob']:>5.1f}%  "
              f"{s['edge_pp']:>+5.1f}pp  {s['decimal_odds']:>5.1f}x  {s['ev_pct']:>+5.1f}%")
    
    # 资金配置建议
    print(f"\n💎 总资金配置建议：")
    total_buy = sum(s["suggested_bet"] for s in buy_signals if s["grade"] in ["S", "A"])
    s_count = sum(1 for s in buy_signals if s["grade"] == "S")
    a_count = sum(1 for s in buy_signals if s["grade"] == "A")
    print(f"  S 级强买入: {s_count} 个")
    print(f"  A 级买入:   {a_count} 个")
    print(f"  推荐总仓位: ${total_buy:,.0f} ({total_buy/bankroll*100:.1f}% of bankroll)")
    print(f"  保留现金:   ${bankroll - total_buy:,.0f} ({(bankroll-total_buy)/bankroll*100:.1f}%)")
    
    # 风险提示
    print(f"\n⚠️  风险提示：")
    print(f"  1. 所有信号基于 100k 蒙特卡洛 + 综合调整，置信度档次：低（<40%）")
    print(f"  2. Quarter Kelly 已是稳健折扣，但仍建议设置止损线（-30%）")
    print(f"  3. 决赛单场赛制随机性高，单注不超过总仓位 30%")
    print(f"  4. 所有偏差因子（伤病/路径/心理）都可能在赛前 48 小时变化")
    print(f"  5. 95% 置信区间显示真实概率可能浮动 ±3pp，下注前确认 CI 不跨越市场价")
    
    print("=" * 100)


def main(bankroll: float = 10000):
    signals = generate_signals(bankroll=bankroll)
    print_signals(signals, bankroll)

    # 保存（与 scheduler._arbitrage_fast_hook 字段结构保持一致，供 web 展示 freshness）
    from utils.io import save_output, DATA_OUTPUTS, DATA_RAW
    from datetime import datetime

    synth_path = DATA_OUTPUTS / "synthesizer_report.json"
    teams_path = DATA_RAW / "teams.json"
    freshness = {
        "trigger": "cascade",  # 走完整 cascade 流水线，区分于 fast hook
        "refreshed_at": datetime.now().isoformat(timespec="seconds"),
        "model_snapshot_ts": datetime.fromtimestamp(synth_path.stat().st_mtime).isoformat(timespec="seconds") if synth_path.exists() else None,
        "market_snapshot_ts": datetime.fromtimestamp(teams_path.stat().st_mtime).isoformat(timespec="seconds") if teams_path.exists() else None,
    }

    save_output("arbitrage_signals.json", {
        "bankroll": bankroll,
        "n_signals": len(signals),
        "signals": signals,
        "market": {
            "name": "2026 FIFA World Cup Winner",
            "venue": "Polymarket",
            "event_url": "https://polymarket.com/event/world-cup-winner",
            "note": "夺冠盘：每队为独立的 Yes/No 合约（'Will <Team> win the 2026 FIFA World Cup?'）",
        },
        "_freshness": freshness,
    })
    print(f"\n✅ 已保存到 data/outputs/arbitrage_signals.json")


if __name__ == "__main__":
    bankroll = float(sys.argv[1]) if len(sys.argv) > 1 else 10000
    main(bankroll)
