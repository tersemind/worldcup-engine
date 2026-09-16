"""
Kalshi 单场赔率抓取器
=====================
免费、无 key、公开 API：
  https://api.elections.kalshi.com/trade-api/v2/markets?series_ticker=KXWCGAME

数据流：
  1. 抓 KXWCGAME 系列所有 markets（每场 3 个市场：主胜 / 平 / 客胜）
  2. 按 event_ticker 聚合：event_ticker → {team_a, team_b, date, p_a, p_d, p_b}
  3. 球队名标准化（Kalshi 三字母代码 → FIFA 长名）
  4. 写入 data/outputs/kalshi_match_odds.json

注意：
  - Kalshi last_price 单位是美元（每份合约结算 $1.00），等价于 0-1 的隐含概率
  - 三选项理论上和为 1，但实际可能 sum > 1（overround）或 sum < 1，需要归一化
  - 部分场次可能只有 2 个选项（无平），按 0 处理
"""
import json
import sys
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

ROOT = Path(__file__).parent.parent.parent
OUT_PATH = ROOT / "data" / "outputs" / "kalshi_match_odds.json"

KALSHI_URL = "https://api.elections.kalshi.com/trade-api/v2/markets?series_ticker=KXWCGAME&limit=1000"

# Kalshi 三字母代码 → FIFA 球队名（teams.json 使用的名）
KALSHI_TO_FIFA = {
    "BRA": "Brazil", "MAR": "Morocco", "MOR": "Morocco",   # MAR/MOR 都见过
    "FRA": "France", "SEN": "Senegal", "NED": "Netherlands", "JPN": "Japan",
    "SWE": "Sweden", "TUN": "Tunisia", "ESP": "Spain",
    "URY": "Uruguay", "URG": "Uruguay",
    "CPV": "Cape Verde", "SAU": "Saudi Arabia", "ARG": "Argentina", "ALG": "Algeria",
    "JOR": "Jordan", "AUT": "Austria", "POR": "Portugal", "COL": "Colombia",
    "UZB": "Uzbekistan", "COD": "DR Congo", "ENG": "England", "CRO": "Croatia",
    "GHA": "Ghana", "PAN": "Panama", "MEX": "Mexico", "RSA": "South Africa",
    "KOR": "Korea Republic", "CZE": "Czechia", "SUI": "Switzerland", "CAN": "Canada",
    "BIH": "Bosnia", "QAT": "Qatar", "USA": "USA", "TUR": "Turkiye",
    "AUS": "Australia", "PAR": "Paraguay", "GER": "Germany", "CUR": "Curacao",
    "ECU": "Ecuador", "CIV": "Cote d'Ivoire", "BEL": "Belgium", "EGY": "Egypt",
    "IRN": "Iran", "NZL": "New Zealand", "HAI": "Haiti", "SCO": "Scotland",
    "NOR": "Norway", "IRQ": "Iraq",
}


def parse_event_ticker(event_ticker: str) -> Dict[str, str]:
    """
    解析 KXWCGAME-26JUN13BRAMA → {date: 2026-06-13, code_a: BRA, code_b: MAR}
    
    格式假定：KXWCGAME-YYMONDD<3code><3code>
    例如 KXWCGAME-26JUN27CODUZB → 2026-06-27, COD vs UZB
    """
    s = event_ticker.replace("KXWCGAME-", "")
    if len(s) < 11:
        return {}
    year_part = "20" + s[:2]      # 26 → 2026
    month_str = s[2:5].upper()    # JUN
    day = s[5:7]                  # 27
    rest = s[7:]                  # CODUZB
    if len(rest) != 6:
        return {}
    code_a = rest[:3]
    code_b = rest[3:]
    
    months = {"JAN":"01","FEB":"02","MAR":"03","APR":"04","MAY":"05","JUN":"06",
              "JUL":"07","AUG":"08","SEP":"09","OCT":"10","NOV":"11","DEC":"12"}
    month_num = months.get(month_str, "06")
    date = f"{year_part}-{month_num}-{day}"
    
    return {"date": date, "code_a": code_a, "code_b": code_b}


def fetch_kalshi() -> Dict[str, Any]:
    """从 Kalshi 公共 API 抓全部 KXWCGAME 市场"""
    req = urllib.request.Request(KALSHI_URL, headers={"User-Agent": "worldcup-engine/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def build():
    raw = fetch_kalshi()
    markets = raw.get("markets", [])
    
    def to_float(v):
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None
    
    # 按 event_ticker 聚合（只保留 active/trading 市场，过滤已结算 finalized）
    by_event = {}
    for m in markets:
        ev = m.get("event_ticker")
        if not ev:
            continue
        status = (m.get("status") or "").lower()
        if status not in ("active", "open", "trading", ""):
            continue
        by_event.setdefault(ev, []).append({
            "ticker": m.get("ticker"),
            "yes_sub_title": m.get("yes_sub_title"),  # 例如 "Brazil" / "Tie"
            "last_price": to_float(m.get("last_price_dollars")),  # 字符串 "0.3100" → float
            "yes_bid": to_float(m.get("yes_bid_dollars")),
            "yes_ask": to_float(m.get("yes_ask_dollars")),
            "volume_24h": to_float(m.get("volume_24h_fp")),
            "open_interest": to_float(m.get("open_interest_fp")),
            "close_time": m.get("close_time"),
        })
    
    # 解析每场比赛
    matches = []
    for ev, mks in by_event.items():
        parsed = parse_event_ticker(ev)
        if not parsed:
            continue
        date = parsed["date"]
        code_a = parsed["code_a"]
        code_b = parsed["code_b"]
        team_a = KALSHI_TO_FIFA.get(code_a)
        team_b = KALSHI_TO_FIFA.get(code_b)
        if not team_a or not team_b:
            continue
        
        # 提取三个选项的价格
        p_a = p_b = p_tie = None
        total_vol = 0
        for m in mks:
            sub = (m.get("yes_sub_title") or "").strip()
            price = m.get("last_price")
            vol = m.get("volume_24h") or 0
            total_vol += vol
            if price is None:
                continue
            sub_lc = sub.lower()
            # Kalshi 新格式: "Reg Time: France" / "Reg Time: Tie"
            # 先提取冒号后的核心名称
            core = sub_lc.split(":")[-1].strip() if ":" in sub_lc else sub_lc
            # 匹配 sub_title
            if core in ("tie", "(平局)", "draw"):
                p_tie = price
            elif team_a.lower() == core:
                p_a = price
            elif team_b.lower() == core:
                p_b = price
            # 兜底：用三字母代码匹配
            elif core.startswith(code_a.lower()) or sub_lc.startswith(code_a.lower()):
                p_a = price
            elif core.startswith(code_b.lower()) or sub_lc.startswith(code_b.lower()):
                p_b = price
        
        if p_a is None and p_b is None:
            continue
        
        # 归一化
        p_a = p_a or 0
        p_b = p_b or 0
        p_tie = p_tie or 0
        total = p_a + p_b + p_tie
        if total > 0:
            p_a /= total
            p_b /= total
            p_tie /= total
        
        matches.append({
            "event_ticker": ev,
            "date": date,
            "team_a": team_a,
            "team_b": team_b,
            "kalshi_p_win_a": round(p_a, 4),
            "kalshi_p_draw": round(p_tie, 4),
            "kalshi_p_win_b": round(p_b, 4),
            "raw_overround_pct": round((total - 1) * 100, 1),
            "total_volume_24h": round(total_vol, 0),
            # 深链入口：Kalshi 事件页（浏览器侧由前端 router 接管）
            "kalshi_url": f"https://kalshi.com/events/{ev}",
            "series_ticker": "KXWCGAME",
            "venue": "Kalshi",
        })
    
    matches.sort(key=lambda m: m["date"])
    
    output = {
        "_source": "Kalshi public API (no auth required)",
        "_endpoint": KALSHI_URL,
        "_fetched_at": datetime.now().isoformat(timespec="seconds"),
        "market": {
            "name": "FIFA World Cup 2026 — Single Game Winner (H/D/A)",
            "venue": "Kalshi",
            "series_ticker": "KXWCGAME",
            "series_url": "https://kalshi.com/markets/kxwcgame",
            "note": "单场胜平负盘：每场比赛 3 个 Yes/No 合约（主胜/平/客胜）",
        },
        "n_matches": len(matches),
        "matches": matches,
    }
    OUT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"✅ 写入 {OUT_PATH}")
    print(f"   {len(matches)} 场单场赔率数据")
    print()
    print("=== 摘要 ===")
    for m in matches:
        print(f"  [{m['date']}] {m['team_a']:12s} {m['kalshi_p_win_a']*100:>4.0f}%  vs  "
              f"{m['kalshi_p_win_b']*100:>4.0f}% {m['team_b']:12s}  (平 {m['kalshi_p_draw']*100:>4.0f}%, "
              f"overround={m['raw_overround_pct']:+.1f}%)")


if __name__ == "__main__":
    build()
