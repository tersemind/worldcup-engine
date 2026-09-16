"""
拉取已结束小组赛比分
====================
合并四类来源到 data/outputs/group_results.json：
  1. ESPN 历史 scoreboard API：遍历小组赛所有日期抓取 state=post 的结果（自动、兜底）
  2. data/raw/live_events.json 中 state=post 的刚结束比赛（自动）
  3. KNOWN_RESULTS 硬编码（人工兜底）
  4. 已有 group_results.json 历史结果（持久化合并）

幂等：重复跑只会增量补齐缺失场次。

输出格式：
{
  "last_updated": "...",
  "n_results": 3,
  "results": {
    "Mexico vs South Africa @ 2026-06-11": {"date":..., "team_a":..., "team_b":..., "score_a":..., "score_b":..., "score_str":..., "winner":..., "source":...},
    ...
  }
}
"""
import json
import re
import time
import urllib.request
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

ROOT = Path(__file__).parent.parent.parent
OUT_PATH = ROOT / "data" / "outputs" / "group_results.json"
LIVE_EVENTS_PATH = ROOT / "data" / "raw" / "live_events.json"
SCHEDULE_PATH = ROOT / "data" / "raw" / "group_schedule.json"

# ESPN scoreboard API（世界杯）
ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard"


# === 已知比赛结果（人工兜底；ESPN 数据有时延迟时用）===
# 格式：(date, team_a, team_b) → (score_a, score_b, source)
KNOWN_RESULTS = {
    # 2026-06-11
    ("2026-06-11", "Mexico", "South Africa"): (2, 0, "ESPN/FIFA"),
    ("2026-06-11", "Korea Republic", "Czechia"): (2, 1, "FIFA/ESPN"),
}


def make_key(m):
    return f"{m['team_a']} vs {m['team_b']} @ {m['date']}"


def _build_entry(date, team_a, team_b, score_a, score_b, source):
    return {
        "date": date,
        "team_a": team_a,
        "team_b": team_b,
        "score_a": score_a,
        "score_b": score_b,
        "score_str": f"{score_a}-{score_b}",
        "winner": team_a if score_a > score_b else (team_b if score_b > score_a else "DRAW"),
        "source": source,
    }


def _espn_name_to_schedule_name(name):
    """ESPN 队名 → 赛程队名（尽量对齐，不匹配的返回 None 让调用方过滤）。"""
    mapping = {
        "USA": "USA",
        "United States": "USA",
        "Mexico": "Mexico",
        "Canada": "Canada",
        "Brazil": "Brazil",
        "Argentina": "Argentina",
        "Germany": "Germany",
        "France": "France",
        "Spain": "Spain",
        "England": "England",
        "Netherlands": "Netherlands",
        "Portugal": "Portugal",
        "Italy": "Italy",
        "Belgium": "Belgium",
        "Uruguay": "Uruguay",
        "Croatia": "Croatia",
        "Denmark": "Denmark",
        "Switzerland": "Switzerland",
        "Colombia": "Colombia",
        "Chile": "Chile",
        "Ecuador": "Ecuador",
        "Paraguay": "Paraguay",
        "Peru": "Peru",
        "Venezuela": "Venezuela",
        "Bolivia": "Bolivia",
        "Jamaica": "Jamaica",
        "Honduras": "Honduras",
        "Costa Rica": "Costa Rica",
        "Panama": "Panama",
        "Guatemala": "Guatemala",
        "El Salvador": "El Salvador",
        "Morocco": "Morocco",
        "Nigeria": "Nigeria",
        "Senegal": "Senegal",
        "Egypt": "Egypt",
        "Tunisia": "Tunisia",
        "Algeria": "Algeria",
        "Cameroon": "Cameroon",
        "Ghana": "Ghana",
        "Ivory Coast": "Cote d'Ivoire",
        "Côte d'Ivoire": "Cote d'Ivoire",
        "South Africa": "South Africa",
        "DR Congo": "DR Congo",
        "Mali": "Mali",
        "Burkina Faso": "Burkina Faso",
        "Guinea": "Guinea",
        "Zambia": "Zambia",
        "Japan": "Japan",
        "Korea Republic": "Korea Republic",
        "South Korea": "Korea Republic",
        "Iran": "Iran",
        "Australia": "Australia",
        "Saudi Arabia": "Saudi Arabia",
        "Qatar": "Qatar",
        "Iraq": "Iraq",
        "Uzbekistan": "Uzbekistan",
        "Jordan": "Jordan",
        "United Arab Emirates": "United Arab Emirates",
        "Oman": "Oman",
        "Bahrain": "Bahrain",
        "China": "China",
        "New Zealand": "New Zealand",
        "Czechia": "Czechia",
        "Czech Republic": "Czechia",
        "Poland": "Poland",
        "Sweden": "Sweden",
        "Norway": "Norway",
        "Ukraine": "Ukraine",
        "Serbia": "Serbia",
        "Slovenia": "Slovenia",
        "Slovakia": "Slovakia",
        "Romania": "Romania",
        "Bulgaria": "Bulgaria",
        "Hungary": "Hungary",
        "Turkey": "Turkiye",
        "Türkiye": "Turkiye",
        "Russia": "Russia",
        "Scotland": "Scotland",
        "Wales": "Wales",
        "Northern Ireland": "Northern Ireland",
        "Republic of Ireland": "Republic of Ireland",
        "Greece": "Greece",
        "Finland": "Finland",
        "Iceland": "Iceland",
        "North Macedonia": "North Macedonia",
        "Albania": "Albania",
        "Bosnia and Herzegovina": "Bosnia and Herzegovina",
        "Montenegro": "Montenegro",
        "Moldova": "Moldova",
        "Armenia": "Armenia",
        "Azerbaijan": "Azerbaijan",
        "Georgia": "Georgia",
        "Kazakhstan": "Kazakhstan",
        "Belarus": "Belarus",
        "Estonia": "Estonia",
        "Latvia": "Latvia",
        "Lithuania": "Lithuania",
        "Luxembourg": "Luxembourg",
        "Malta": "Malta",
        "Cyprus": "Cyprus",
        "Kosovo": "Kosovo",
        "Gibraltar": "Gibraltar",
        "Faroe Islands": "Faroe Islands",
        "Andorra": "Andorra",
        "San Marino": "San Marino",
        "Liechtenstein": "Liechtenstein",
        "Haiti": "Haiti",
        "Curacao": "Curacao",
        "Curaçao": "Curacao",
        "Trinidad and Tobago": "Trinidad and Tobago",
    }
    return mapping.get(name)


def _load_schedule_matches():
    try:
        d = json.load(open(SCHEDULE_PATH))
        return d.get("matches", []) or []
    except Exception as e:
        print(f"  ⚠ group_schedule.json 读取失败: {e}")
        return []


def _load_live_post_results():
    """从 live_events.json 提取 state=post（已结束）比赛。

    返回 dict: { (date, team_a, team_b): (score_a, score_b, source) }
    其中 (team_a, team_b) 用 ESPN home/away 顺序；调用方需做主客对齐。

    失败时返回空 dict（不抛异常，feedback_no_break_existing）。
    """
    out = {}
    try:
        if not LIVE_EVENTS_PATH.exists():
            return out
        d = json.load(open(LIVE_EVENTS_PATH))
        for m in d.get("live_matches", []) or []:
            if m.get("state") != "post":
                continue
            mk = m.get("match_key", "")
            # match_key 形如 "USA vs Paraguay @ 2026-06-12"
            if " @ " not in mk:
                continue
            date = mk.split(" @ ")[-1].strip()
            ta = m.get("team_a")
            tb = m.get("team_b")
            sa = m.get("score_a")
            sb = m.get("score_b")
            if not (ta and tb and date) or sa is None or sb is None:
                continue
            try:
                sa = int(sa); sb = int(sb)
            except (TypeError, ValueError):
                continue
            out[(date, ta, tb)] = (sa, sb, "ESPN-live")
    except Exception as e:
        print(f"  ⚠ live_events 读取失败（已忽略）: {e}")
    return out


def _fetch_espn_scoreboard(date_str):
    """date_str: YYYYMMDD。返回该日所有 FIFA World Cup 赛事列表（原始 ESPN event 字典）。"""
    url = f"{ESPN_SCOREBOARD_URL}?dates={date_str}"
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("events", []) or []
    except Exception as e:
        print(f"  ⚠ ESPN scoreboard {date_str} 拉取失败: {e}")
        return []


def _parse_espn_event(event):
    """从 ESPN event 提取 (home_name, away_name, home_score, away_score, status_state)。"""
    comp = event.get("competitions", [{}])[0]
    competitors = comp.get("competitors", [])
    if len(competitors) != 2:
        return None

    home = away = None
    for c in competitors:
        team = c.get("team", {})
        name = team.get("displayName") or team.get("shortDisplayName") or team.get("abbreviation")
        if c.get("homeAway") == "home":
            home = name
            home_score = c.get("score")
        else:
            away = name
            away_score = c.get("score")
    status = event.get("status", {}).get("type", {}).get("state", "").lower()
    if not home or not away:
        return None
    try:
        home_score = int(home_score)
        away_score = int(away_score)
    except (TypeError, ValueError):
        return None
    return home, away, home_score, away_score, status


def _load_historical_espn_results(schedule_dates):
    """遍历小组赛日期，从 ESPN scoreboard 抓取所有 post 状态比赛。

    ESPN 与本地赛程可能存在时区/日期偏差（如 00:00 EDT 的比赛在 ESPN
    被归到次日）。因此这里按"队对"索引，返回：
      { frozenset({team_a, team_b}): {"date": espn_date, "home": home, "away": away,
                                      "home_score": hs, "away_score": aw, "source": src} }
    后续匹配时按赛程主客对齐分数。
    """
    out = {}
    if not schedule_dates:
        return out

    print(f"\n📡 开始从 ESPN scoreboard 回填 {len(schedule_dates)} 个小组赛日期...")
    for date in sorted(schedule_dates):
        date_compact = date.replace("-", "")
        events = _fetch_espn_scoreboard(date_compact)
        if not events:
            continue
        for ev in events:
            parsed = _parse_espn_event(ev)
            if not parsed:
                continue
            home_raw, away_raw, hs, aw, status = parsed
            if status != "post":
                continue
            home = _espn_name_to_schedule_name(home_raw)
            away = _espn_name_to_schedule_name(away_raw)
            if not home or not away:
                continue
            pair = frozenset({home, away})
            # 若同一队对在多个日期出现（理论上小组赛不会），保留已有的即可
            if pair in out:
                continue
            out[pair] = {
                "date": date,
                "home": home,
                "away": away,
                "home_score": hs,
                "away_score": aw,
                "source": "ESPN-scoreboard",
            }
        # 轻量限速，避免被 API 限流
        time.sleep(0.15)

    print(f"   ESPN scoreboard 共抓取到 {len(out)} 场已结束小组赛")
    return out


def _normalize_result_for_schedule(m, candidates):
    """给定赛程 match m 与多个候选结果源，返回按赛程主客对齐的 entry。

    candidates 是混合 dict，可能包含：
      - 旧格式（按日期索引）: { (date, team_a, team_b): (score_a, score_b, source) }
      - 新格式（按队对索引）: { frozenset({team_a, team_b}): {"date", "home", "away", "home_score", "away_score", "source"} }
    """
    date = m["date"]
    sched_a = m["team_a"]
    sched_b = m["team_b"]

    # 1) 优先精确按日期+队名匹配（兼容旧 live_events / KNOWN_RESULTS）
    if (date, sched_a, sched_b) in candidates:
        sa, sb, src = candidates[(date, sched_a, sched_b)]
        return _build_entry(date, sched_a, sched_b, sa, sb, src)
    if (date, sched_b, sched_a) in candidates:
        sa_rev, sb_rev, src = candidates[(date, sched_b, sched_a)]
        return _build_entry(date, sched_a, sched_b, sb_rev, sa_rev, src)

    # 2) 按队对匹配（处理 ESPN 日期偏差）
    pair = frozenset({sched_a, sched_b})
    if pair in candidates and isinstance(candidates[pair], dict):
        ev = candidates[pair]
        if sched_a == ev["home"] and sched_b == ev["away"]:
            sa, sb = ev["home_score"], ev["away_score"]
        elif sched_a == ev["away"] and sched_b == ev["home"]:
            sa, sb = ev["away_score"], ev["home_score"]
        else:
            return None
        return _build_entry(date, sched_a, sched_b, sa, sb, ev["source"])

    return None


def build():
    # 加载赛程
    matches = _load_schedule_matches()
    schedule_dates = sorted({m["date"] for m in matches})

    # 加载已有结果（若有）— 历史持久化
    existing = {}
    if OUT_PATH.exists():
        existing = json.load(open(OUT_PATH)).get("results", {})
    results = dict(existing)

    # ── 来源 1：ESPN scoreboard 历史回填（新增）
    historical = _load_historical_espn_results(schedule_dates)

    # ── 来源 2：ESPN live_events.json 中 state=post（自动）
    live_post = _load_live_post_results()

    # 合并来源：live 最新，historical 兜底，known 最后
    all_candidates = {}
    all_candidates.update(historical)
    all_candidates.update(live_post)
    all_candidates.update({
        (date, ta, tb): (sa, sb, src)
        for (date, ta, tb), (sa, sb, src) in KNOWN_RESULTS.items()
    })

    n_new_hist = 0
    n_new_live = 0
    n_new_known = 0

    for m in matches:
        key = make_key(m)
        date = m["date"]
        sched_a = m["team_a"]
        sched_b = m["team_b"]

        chosen = _normalize_result_for_schedule(m, all_candidates)
        if chosen is None:
            continue

        if results.get(key) != chosen:
            src = chosen["source"]
            if src == "ESPN-scoreboard":
                n_new_hist += 1
            elif src == "ESPN-live":
                n_new_live += 1
            else:
                n_new_known += 1
            results[key] = chosen

    output = {
        "last_updated": datetime.now().isoformat(timespec="seconds"),
        "n_results": len(results),
        "results": results,
    }
    OUT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2))

    print(f"\n✅ 写入 {OUT_PATH}")
    print(f"   已结束场次: {len(results)} / 72")
    print(f"   本次新增/更新: scoreboard={n_new_hist}, live={n_new_live}, known={n_new_known}")
    if results:
        print(f"\n   已有结果:")
        for k, v in sorted(results.items(), key=lambda x: x[1]["date"]):
            print(f"     [{v['date']}] {v['team_a']} {v['score_str']} {v['team_b']} (来源: {v['source']})")


if __name__ == "__main__":
    build()
