#!/usr/bin/env python3
"""
自动从 ESPN 抓取世界杯淘汰赛实际结果，更新到 knockout_results.json。
由 launchd 每 10 分钟调度一次。
"""
from __future__ import annotations

import json
import ssl
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = BASE_DIR / "data" / "outputs"
RESULTS_PATH = OUTPUT_DIR / "knockout_results.json"
ACTUAL_MATCHES_PATH = OUTPUT_DIR / "actual_knockout_matches.json"
GROUP_SCHEDULE_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "group_schedule.json"

# worldcup-engine API endpoint（本机）
API_BASE = "http://localhost:8088"

# ESPN 队名标准化映射（worldcup-engine API 队名 -> ESPN displayName）
# 用于处理 ESPN 与本地命名差异
ESPN_NAME_NORMALIZE = {
    "Korea Republic": ["Korea Republic", "South Korea"],
    "Czechia": ["Czechia", "Czech Republic"],
    "Bosnia": ["Bosnia", "Bosnia and Herzegovina", "Bosnia-Herzegovina", "Bosnia & Herzegovina"],
    "USA": ["USA", "United States"],
    "Turkiye": ["Turkiye", "Turkey", "Türkiye"],
    "Cape Verde": ["Cape Verde", "Cape Verde Islands"],
    "Cote d'Ivoire": ["Cote d'Ivoire", "Côte d'Ivoire", "Ivory Coast"],
    "Curacao": ["Curacao", "Curaçao"],
    "DR Congo": ["DR Congo", "Congo DR", "Congo"],
    "Uzbekistan": ["Uzbekistan"],
    "Iran": ["Iran"],
}

# 反向索引：任意 ESPN 名称 -> 本地标准名称
_ESPN_TO_LOCAL: dict[str, str] = {}
for local_name, espn_names in ESPN_NAME_NORMALIZE.items():
    for name in espn_names:
        _ESPN_TO_LOCAL[name] = local_name


def http_get_json(url: str, timeout: int = 20) -> dict:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_knockout_fixtures() -> dict[int, dict]:
    """从 /api/r32 获取 1/16 决赛对阵表，建立 match_id -> {team_a, team_b, date}。"""
    data = http_get_json(f"{API_BASE}/api/r32", timeout=20)
    fixtures = {}
    for m in data.get("matches", []):
        mid = m.get("match_id")
        if mid is None:
            continue
        fixtures[int(mid)] = {
            "date": m.get("date", ""),
            "team_a": m.get("team_a", ""),
            "team_b": m.get("team_b", ""),
        }
    return fixtures


def normalize_team(name: str) -> str:
    """把 ESPN 队名转换为本地标准队名。"""
    return _ESPN_TO_LOCAL.get(name, name)


def _load_group_match_pairs() -> set[frozenset[str]]:
    """加载小组赛所有对阵队对，用于过滤淘汰赛结果。"""
    try:
        data = json.load(open(GROUP_SCHEDULE_PATH, encoding="utf-8"))
    except Exception:
        return set()
    pairs = set()
    for m in data.get("matches", []):
        ta = m.get("team_a")
        tb = m.get("team_b")
        if ta and tb:
            pairs.add(frozenset({ta, tb}))
    return pairs


def fetch_espn_knockout_results(start_day: datetime, end_day: datetime) -> list[dict]:
    """从 ESPN scoreboard 抓取指定日期范围内的已完赛比赛结果（过滤小组赛）。"""
    results = []
    group_pairs = _load_group_match_pairs()
    day = start_day
    while day <= end_day:
        date_str = day.strftime("%Y%m%d")
        url = (
            "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/"
            f"scoreboard?dates={date_str}"
        )
        try:
            data = http_get_json(url, timeout=15)
        except Exception as e:
            print(f"[WARN] ESPN {date_str} 请求失败: {e}")
            day += timedelta(days=1)
            continue

        for ev in data.get("events", []):
            status = ev.get("status", {}).get("type", {}).get("name", "")
            if status not in ("STATUS_FULL_TIME", "STATUS_FINAL", "STATUS_FINAL_AET", "STATUS_FINAL_PEN"):
                continue
            comps = ev.get("competitions", [])
            if not comps:
                continue
            comp = comps[0]
            teams = comp.get("competitors", [])
            if len(teams) != 2:
                continue
            home = next((t for t in teams if t.get("homeAway") == "home"), None)
            away = next((t for t in teams if t.get("homeAway") == "away"), None)
            if not home or not away:
                continue
            home_name = normalize_team(home.get("team", {}).get("displayName", ""))
            away_name = normalize_team(away.get("team", {}).get("displayName", ""))
            # 过滤小组赛（同名队对且出现在小组赛赛程中）
            if frozenset({home_name, away_name}) in group_pairs:
                continue
            home_score = int(home.get("score", 0))
            away_score = int(away.get("score", 0))
            event_date = ev.get("date", "")[:10]
            winner = (
                "draw"
                if home_score == away_score
                else (home_name if home_score > away_score else away_name)
            )
            results.append({
                "date": event_date,
                "team_a": home_name,
                "team_b": away_name,
                "score": f"{home_score}-{away_score}",
                "winner": winner,
                "source": "ESPN",
            })
        day += timedelta(days=1)
    return results


def match_result_to_fixture(result: dict, fixtures: dict[int, dict]) -> int | None:
    """根据队名匹配，返回对应的 match_id。"""
    r_a, r_b = result["team_a"], result["team_b"]
    for mid, f in fixtures.items():
        f_a, f_b = f["team_a"], f["team_b"]
        if {r_a, r_b} == {f_a, f_b}:
            return mid
    return None


def load_existing_results() -> dict:
    if not RESULTS_PATH.exists():
        return {"_source": "ESPN FIFA World Cup scoreboard", "results": {}}
    try:
        return json.load(open(RESULTS_PATH, encoding="utf-8"))
    except Exception:
        return {"_source": "ESPN FIFA World Cup scoreboard", "results": {}}


def main():
    today = datetime.now(timezone.utc)
    # 淘汰赛从 6/28 开始，到决赛约 7/19，往前多预留几天
    start = datetime(2026, 6, 25, tzinfo=timezone.utc)
    end = today + timedelta(days=2)

    print(f"[{datetime.now(timezone.utc).isoformat()}] 开始更新淘汰赛结果...")

    fixtures = fetch_knockout_fixtures()
    if not fixtures:
        print("[WARN] 未获取到 1/16 决赛对阵表，跳过更新")
        return

    espn_results = fetch_espn_knockout_results(start, end)
    print(f"[INFO] ESPN 返回 {len(espn_results)} 场已完赛")

    data = load_existing_results()
    existing = data.get("results", {})
    updated = 0
    added = 0

    for r in espn_results:
        mid = match_result_to_fixture(r, fixtures)
        if mid is None:
            print(f"[WARN] 未匹配到 match_id: {r['team_a']} vs {r['team_b']}")
            continue
        key = str(mid)
        if key in existing:
            if existing[key].get("score") != r["score"]:
                existing[key] = r
                updated += 1
        else:
            existing[key] = r
            added += 1

    data["results"] = existing
    data["_last_updated"] = datetime.now(timezone.utc).isoformat()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # 同时保存 ESPN 抓取的原始全部已完赛场次（用于「实际赛程安排」tab 直接展示真实赛果，
    # 不依赖与预测 bracket 的 match_id 匹配）
    actual_matches_data = {
        "_source": "ESPN FIFA World Cup scoreboard",
        "_last_updated": datetime.now(timezone.utc).isoformat(),
        "n_matches": len(espn_results),
        "matches": sorted(espn_results, key=lambda x: x.get("date", "")),
    }
    with open(ACTUAL_MATCHES_PATH, "w", encoding="utf-8") as f:
        json.dump(actual_matches_data, f, ensure_ascii=False, indent=2)

    print(f"[INFO] 更新完成：新增 {added} 场，更新 {updated} 场，总计 {len(existing)} 场")
    print(f"[INFO] 已保存全部 ESPN 实际赛果 {len(espn_results)} 场到 {ACTUAL_MATCHES_PATH}")


if __name__ == "__main__":
    main()
