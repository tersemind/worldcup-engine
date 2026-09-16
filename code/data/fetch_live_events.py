"""
赛中事件流抓取（ESPN scoreboard，免费无 key）
=============================================

逻辑:
  1. 抓今天 + 昨天 + 明天的 ESPN scoreboard（避免时区错过）
  2. 只对 state=in（正在进行）或 state=post（刚结束 < 2h）的场次拉事件
  3. 写入 data/raw/live_events.json
  4. 没 live 场次时秒退（无 quota 消耗）

输出格式:
  {
    "_metadata": {...},
    "live_matches": [
      {
        "match_key": "USA vs Paraguay @ 2026-06-12",
        "state": "in" / "post",
        "clock": "65'",
        "score": "1-0",
        "team_a": "USA", "team_b": "Paraguay",
        "events": [
          {"minute": "9'", "type": "Goal", "team": "USA",
           "player": "Pulisic", "detail": ""},
          ...
        ],
        "updated_at": "..."
      }
    ]
  }

调用频率建议: scheduler 每 5min 跑一次。无 live 场次时秒退。

下游消费: code/agents/in_match.py / models/bayesian_updater.py 可消费这个文件做赛中预测刷新。
"""
import sys
import json
import re
import requests
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.io import DATA_RAW

warnings.filterwarnings("ignore")

ESPN_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard"
OUT_PATH = DATA_RAW / "live_events.json"

# 复用 schedule fetcher 里的队名映射
ESPN_TO_TEAM = {
    "United States": "USA",
    "South Korea": "Korea Republic",
    "Korea Rep": "Korea Republic",
    "Türkiye": "Turkiye",
    "Turkey": "Turkiye",
    "Bosnia-Herzegovina": "Bosnia",
    "Bosnia and Herzegovina": "Bosnia",
    "Ivory Coast": "Cote d'Ivoire",
    "Côte d'Ivoire": "Cote d'Ivoire",
    "Curaçao": "Curacao",
    "Cabo Verde": "Cape Verde",
    "DR Congo": "DR Congo",
    "Congo DR": "DR Congo",
    "Democratic Republic of the Congo": "DR Congo",
}


def _norm_team(name: str) -> str:
    return ESPN_TO_TEAM.get(name, name)


def _fetch_day(date_str: str, timeout: int = 15) -> List[Dict]:
    """抓单日比赛"""
    r = requests.get(ESPN_URL, params={"dates": date_str, "limit": 50}, timeout=timeout)
    r.raise_for_status()
    return r.json().get("events", [])


def _parse_event(comp: Dict, e: Dict) -> Dict:
    """从 ESPN competition + event 字段提取关键信息"""
    state_obj = e.get("status", {}).get("type", {})
    state = state_obj.get("state", "")  # pre/in/post
    
    competitors = comp.get("competitors") or []
    home = away = None
    home_score = away_score = 0
    for c in competitors:
        team = c.get("team", {})
        name = team.get("displayName") or team.get("name") or ""
        norm = _norm_team(name)
        try:
            score = int(c.get("score", 0))
        except (ValueError, TypeError):
            score = 0
        if c.get("homeAway") == "home":
            home, home_score = norm, score
        elif c.get("homeAway") == "away":
            away, away_score = norm, score
    
    # 备用 shortName 解析
    if not home or not away:
        m = re.match(r"^(.+?)\s+(?:at|vs\.?)\s+(.+)$", e.get("shortName", ""))
        if m:
            away = away or _norm_team(m.group(1).strip())
            home = home or _norm_team(m.group(2).strip())
    
    # 事件流
    events = []
    for ev in comp.get("details", []) or []:
        clock = ev.get("clock", {}).get("displayValue", "")
        etype_obj = ev.get("type", {})
        etype = etype_obj.get("text", "")
        team_id = ev.get("team", {}).get("id")
        # 把 team_id 映射回 home/away
        ev_team = None
        for c in competitors:
            if c.get("team", {}).get("id") == team_id:
                ev_team = _norm_team(c.get("team", {}).get("displayName", ""))
                break
        
        athletes = ev.get("athletesInvolved") or []
        player = athletes[0].get("displayName", "") if athletes else ""
        
        events.append({
            "minute": clock,
            "type": etype,
            "team": ev_team or "?",
            "player": player,
            "scoring_play": ev.get("scoringPlay", False),
        })
    
    # 比赛日期 — 用 ESPN 给的 UTC 日期（生成 match_key 用）
    iso = e.get("date", "")
    try:
        dt_utc = datetime.strptime(iso, "%Y-%m-%dT%H:%MZ")
        # 用 EDT 估算本地日期（赛事大多在北美东部时区）
        date_local = (dt_utc - timedelta(hours=4)).strftime("%Y-%m-%d")
    except ValueError:
        date_local = ""
    
    match_key = f"{home} vs {away} @ {date_local}" if home and away else f"unknown @ {date_local}"
    
    return {
        "match_key": match_key,
        "espn_id":   e.get("id"),
        "state":     state,
        "state_desc": state_obj.get("description", ""),
        "clock":     e.get("status", {}).get("displayClock", ""),
        "period":    e.get("status", {}).get("period"),
        "score":     f"{home_score}-{away_score}",
        "score_a":   home_score,
        "score_b":   away_score,
        "team_a":    home,
        "team_b":    away,
        "events":    events,
    }


def main():
    now = datetime.now()
    # 抓昨天/今天/明天，避免时区跨日漏掉
    dates_to_check = [
        (now - timedelta(days=1)).strftime("%Y%m%d"),
        now.strftime("%Y%m%d"),
        (now + timedelta(days=1)).strftime("%Y%m%d"),
    ]
    
    all_events = []
    for d in dates_to_check:
        try:
            all_events.extend(_fetch_day(d))
        except Exception as e:
            print(f"  ⚠ 抓 {d} 失败: {e}")
    
    # 解析所有比赛 + 筛 in 和最近 post
    matches = []
    live_count = 0
    recent_post_count = 0
    
    for e in all_events:
        comp = (e.get("competitions") or [{}])[0]
        parsed = _parse_event(comp, e)
        if not parsed["team_a"] or not parsed["team_b"]:
            continue
        
        state = parsed["state"]
        # 关注的: 正在进行 (in) 或 < 3h 内结束 (post)
        if state == "in":
            matches.append(parsed)
            live_count += 1
        elif state == "post":
            # 看比赛结束时间
            iso = e.get("date", "")
            try:
                dt_utc = datetime.strptime(iso, "%Y-%m-%dT%H:%MZ")
                # 比赛开始后 2.5h 算"刚结束"
                end_est = dt_utc + timedelta(hours=2, minutes=30)
                age_hours = (datetime.utcnow() - end_est).total_seconds() / 3600
                # 窗口扩到 -3h ~ +48h：fetch_group_results 用这份数据补 group_results.json，
                # 必须保留过去 2 天的 post 比赛（cascade 间隔 + 夜间停机都可能 >3h）
                if -3 <= age_hours <= 48:  # 刚结束 ±3h，已结束最多 48h
                    parsed["age_hours"] = round(age_hours, 1)
                    matches.append(parsed)
                    recent_post_count += 1
            except ValueError:
                pass
    
    out = {
        "_metadata": {
            "updated_at": now.strftime("%Y-%m-%d %H:%M:%S"),
            "source": "ESPN scoreboard (no key required)",
            "n_live": live_count,
            "n_recent_post": recent_post_count,
            "dates_checked": dates_to_check,
        },
        "live_matches": matches,
    }
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    
    print(f"=== Live Events Fetch ===")
    print(f"日期窗口: {dates_to_check}")
    print(f"正在进行: {live_count}  ·  刚结束: {recent_post_count}")
    
    if not matches:
        print(f"  ✓ 无 live 场次，写入空集合")
        return
    
    print(f"\n场次:")
    for m in matches:
        st = m["state"]
        emoji = "🔴" if st == "in" else "✓"
        clock = m.get("clock", "") or m.get("state_desc", "")
        print(f"  {emoji} {m['team_a']} {m['score']} {m['team_b']:18s} [{st} {clock}] events={len(m['events'])}")
        for ev in m["events"][:5]:
            tag = "⚽" if ev["scoring_play"] else ev["type"][:1]
            print(f"      [{ev['minute']:>4s}] {tag} {ev['type']:18s} {ev['team']:8s} {ev['player']}")
        if len(m["events"]) > 5:
            print(f"      ... 共 {len(m['events'])} 事件")


if __name__ == "__main__":
    main()
