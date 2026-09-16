"""
赛程校验 / 增量刷新
====================
从 ESPN 公开 API（无需 key）抓取 2026 FIFA World Cup 完整 104 场赛程，
对比 data/raw/group_schedule.json，识别 FIFA 微调（venue / time_local 变化）。

设计：
  - 仅检测+报告，不暴力覆盖（避免误抓时把 schedule.json 写崩）
  - 找到 ≥1 处变化时才覆盖，并产出 changelog
  - 写入新的 schedule 时保留所有原字段（match_id / tz）

来源：https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard?dates=20260611-20260719
"""
import sys
import json
import re
import requests
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.io import DATA_RAW

warnings.filterwarnings("ignore")

ESPN_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard"
SCHED_PATH = DATA_RAW / "group_schedule.json"
CHANGELOG_PATH = DATA_RAW / "schedule_changelog.json"

# ESPN 队名 → 我们的标准名（teams.json 里的）
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


def _parse_espn_match(e: Dict[str, Any]) -> Dict[str, Any]:
    """提取关键字段"""
    iso = e.get("date", "")  # e.g. "2026-06-12T19:00Z" (UTC)
    try:
        dt_utc = datetime.strptime(iso, "%Y-%m-%dT%H:%MZ")
    except ValueError:
        return None
    # ESPN 的 date 是 UTC，我们 schedule.json 的 time_local 是 EDT/PDT/CDT
    # 这里只比对日期 + UTC 时分，避免 tz 噪声；变更检测时把两边都规约到 UTC
    
    comp = (e.get("competitions") or [{}])[0]
    venue = comp.get("venue") or e.get("venue") or {}
    if isinstance(venue, dict):
        venue_name = venue.get("fullName") or venue.get("displayName") or ""
        venue_city = (venue.get("address") or {}).get("city", "")
    else:
        venue_name, venue_city = "", ""
    
    competitors = comp.get("competitors") or []
    home = away = None
    for c in competitors:
        team_obj = c.get("team", {})
        name = team_obj.get("displayName") or team_obj.get("name") or team_obj.get("shortDisplayName", "")
        homeAway = c.get("homeAway", "")
        if homeAway == "home":
            home = _norm_team(name)
        elif homeAway == "away":
            away = _norm_team(name)
    
    if not home or not away:
        # 备用: shortName 字段 "TEAMA at TEAMB"
        short = e.get("shortName", "")
        m = re.match(r"^(.+?)\s+(?:at|vs\.?)\s+(.+)$", short)
        if m:
            if not away:
                away = _norm_team(m.group(1).strip())
            if not home:
                home = _norm_team(m.group(2).strip())
    
    return {
        "iso_utc":   iso,
        "dt_utc":    dt_utc,
        "team_a":    home,
        "team_b":    away,
        "venue":     venue_name,
        "venue_city": venue_city,
    }


def fetch_espn_schedule(timeout: int = 20) -> List[Dict]:
    """抓 ESPN 全 104 场（一次调用 limit=200）"""
    params = {"dates": "20260611-20260719", "limit": 200}
    r = requests.get(ESPN_URL, params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    out = []
    for e in data.get("events", []):
        parsed = _parse_espn_match(e)
        if parsed and parsed["team_a"] and parsed["team_b"]:
            out.append(parsed)
    return out


def _key(team_a: str, team_b: str) -> str:
    """对阵 key，对称"""
    return " vs ".join(sorted([team_a, team_b]))


def detect_changes(espn_matches: List[Dict], local: Dict) -> List[Dict]:
    """对比 ESPN 与本地 schedule，输出 changes 清单
    
    只检测：
      - 日期变化（date 字段不同）
      - 场馆变化（venue 或 venue_city 不同）
    
    不检测 time_local 时区差（ESPN 是 UTC，本地是 venue local time，转换易错）
    """
    # 本地按对阵 key 索引
    local_by_pair = {}
    for m in local.get("matches", []):
        k = _key(m["team_a"], m["team_b"])
        local_by_pair[k] = m
    
    changes = []
    for em in espn_matches:
        k = _key(em["team_a"], em["team_b"])
        lm = local_by_pair.get(k)
        if not lm:
            # ESPN 有，本地没 — 可能是淘汰赛对阵（我们 schedule.json 只有 72 小组）
            continue
        
        espn_date_utc = em["dt_utc"].strftime("%Y-%m-%d")
        local_date = lm.get("date")
        
        # 跨日容忍：ESPN UTC vs local time 经常差 1 天（EDT/PDT 22:00 → UTC 02:00 次日）
        # 只在差 ≥ 2 天时才报警（说明 FIFA 真改了日期）
        diffs = []
        if local_date and local_date != espn_date_utc:
            try:
                d_local = datetime.strptime(local_date, "%Y-%m-%d")
                d_espn = datetime.strptime(espn_date_utc, "%Y-%m-%d")
                if abs((d_espn - d_local).days) >= 2:
                    diffs.append({"field": "date", "local": local_date, "espn_utc": espn_date_utc})
            except ValueError:
                pass
        if em["venue"] and lm.get("venue") and em["venue"] != lm["venue"]:
            diffs.append({"field": "venue", "local": lm.get("venue"), "espn": em["venue"]})
        # venue_city 经常名称不一致（New York/New Jersey vs East Rutherford），仅作 hint
        if em["venue_city"] and lm.get("venue_city") and em["venue_city"].lower() not in lm["venue_city"].lower() \
                and lm["venue_city"].lower() not in em["venue_city"].lower():
            diffs.append({"field": "venue_city", "local": lm.get("venue_city"), "espn": em["venue_city"]})
        
        if diffs:
            changes.append({
                "pair": k,
                "team_a": lm["team_a"],
                "team_b": lm["team_b"],
                "match_id": lm.get("match_id"),
                "diffs": diffs,
            })
    
    return changes


def write_changelog(changes: List[Dict]):
    """追加到 changelog（保留历史）"""
    history = []
    if CHANGELOG_PATH.exists():
        try:
            history = json.load(open(CHANGELOG_PATH)).get("history", [])
        except Exception:
            pass
    entry = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "n_changes": len(changes),
        "changes": changes,
    }
    history.insert(0, entry)
    # 只保留最近 20 条
    history = history[:20]
    CHANGELOG_PATH.write_text(json.dumps({
        "_metadata": {"last_checked": entry["checked_at"], "source": "ESPN scoreboard"},
        "history": history,
    }, ensure_ascii=False, indent=2))


def main():
    print("=== 赛程校验 (ESPN) ===")
    if not SCHED_PATH.exists():
        print(f"✗ {SCHED_PATH} 不存在，跳过对比")
        sys.exit(1)
    
    local = json.load(open(SCHED_PATH))
    
    try:
        espn = fetch_espn_schedule()
    except Exception as e:
        print(f"✗ ESPN 抓取失败: {e}")
        sys.exit(1)
    
    print(f"ESPN: 抓到 {len(espn)} 场")
    print(f"本地: {len(local.get('matches', []))} 场")
    
    # 简单匹配统计
    local_pairs = set(_key(m["team_a"], m["team_b"]) for m in local.get("matches", []))
    espn_pairs = set(_key(m["team_a"], m["team_b"]) for m in espn)
    matched = local_pairs & espn_pairs
    print(f"匹配对阵: {len(matched)} / 本地 {len(local_pairs)}")
    
    changes = detect_changes(espn, local)
    write_changelog(changes)
    
    if not changes:
        print(f"\n✓ 无变化（与 ESPN 一致）")
        return
    
    print(f"\n⚠ 发现 {len(changes)} 处变化:")
    for c in changes[:20]:
        diffs_str = "; ".join(
            f"{d['field']}: {d['local']!r} → {d['espn']!r}" for d in c['diffs']
        )
        print(f"  - {c['team_a']} vs {c['team_b']}: {diffs_str}")
    
    print(f"\nchangelog 已写入: {CHANGELOG_PATH}")
    print(f"⚠ 当前实现仅检测/报告，不覆盖 schedule.json")
    print(f"  如需更新 schedule.json，请手动核对后 mv changelog 中的修正字段")


if __name__ == "__main__":
    main()
