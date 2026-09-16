"""
构造 72 场小组赛赛程数据（FIFA 2026 真实日期 + 场馆）
=========================================================
来源：WebFetch from kickoffadventures.com（FIFA 官方赛程派生）
+ 把 UEFA Play-off A-D / FIFA Play-off 1-2 占位符替换为 groups.json 里的实际球队。

输出：data/raw/group_schedule.json
"""
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent

# 占位符 → 实际球队映射（依据已抽签确定的最终参赛队）
PLAYOFF_TEAM = {
    "UEFA Play-off A": "Bosnia",       # B 组第 4 位（与 Switzerland/Canada/Qatar 同组）
    "UEFA Play-off B": "Sweden",       # F 组第 4 位（与 Netherlands/Japan/Tunisia 同组）
    "UEFA Play-off C": "Turkiye",      # D 组第 4 位（与 USA/Australia/Paraguay 同组）
    "UEFA Play-off D": "Czechia",      # A 组第 4 位（与 Mexico/South Africa/Korea Republic 同组）
    "FIFA Play-off 1": "DR Congo",     # K 组第 4 位（与 Portugal/Colombia/Uzbekistan 同组）
    "FIFA Play-off 2": "Iraq",         # I 组第 4 位（与 France/Senegal/Norway 同组）
}


RAW_SCHEDULE = [
    # === Group A ===
    {"group":"A","date":"2026-06-11","time_local":"15:00","tz":"EDT","venue":"Estadio Azteca","venue_city":"Mexico City","team_a":"Mexico","team_b":"South Africa"},
    {"group":"A","date":"2026-06-11","time_local":"21:00","tz":"EDT","venue":"Estadio Akron","venue_city":"Guadalajara","team_a":"Korea Republic","team_b":"UEFA Play-off D"},
    {"group":"A","date":"2026-06-18","time_local":"12:00","tz":"EDT","venue":"Mercedes-Benz Stadium","venue_city":"Atlanta","team_a":"UEFA Play-off D","team_b":"South Africa"},
    {"group":"A","date":"2026-06-18","time_local":"21:00","tz":"EDT","venue":"Estadio Akron","venue_city":"Guadalajara","team_a":"Mexico","team_b":"Korea Republic"},
    {"group":"A","date":"2026-06-24","time_local":"21:00","tz":"EDT","venue":"Estadio Azteca","venue_city":"Mexico City","team_a":"UEFA Play-off D","team_b":"Mexico"},
    {"group":"A","date":"2026-06-24","time_local":"21:00","tz":"EDT","venue":"Estadio BBVA","venue_city":"Monterrey","team_a":"South Africa","team_b":"Korea Republic"},
    # === Group B ===
    {"group":"B","date":"2026-06-12","time_local":"15:00","tz":"EDT","venue":"BMO Field","venue_city":"Toronto","team_a":"Canada","team_b":"UEFA Play-off A"},
    {"group":"B","date":"2026-06-13","time_local":"15:00","tz":"EDT","venue":"Levi's Stadium","venue_city":"San Francisco Bay Area","team_a":"Qatar","team_b":"Switzerland"},
    {"group":"B","date":"2026-06-18","time_local":"15:00","tz":"EDT","venue":"SoFi Stadium","venue_city":"Los Angeles","team_a":"Switzerland","team_b":"UEFA Play-off A"},
    {"group":"B","date":"2026-06-18","time_local":"21:00","tz":"EDT","venue":"BC Place","venue_city":"Vancouver","team_a":"Canada","team_b":"Qatar"},
    {"group":"B","date":"2026-06-24","time_local":"21:00","tz":"EDT","venue":"BC Place","venue_city":"Vancouver","team_a":"Switzerland","team_b":"Canada"},
    {"group":"B","date":"2026-06-24","time_local":"15:00","tz":"EDT","venue":"Lumen Field","venue_city":"Seattle","team_a":"UEFA Play-off A","team_b":"Qatar"},
    # === Group C ===
    {"group":"C","date":"2026-06-13","time_local":"18:00","tz":"EDT","venue":"MetLife Stadium","venue_city":"New York/New Jersey","team_a":"Brazil","team_b":"Morocco"},
    {"group":"C","date":"2026-06-13","time_local":"21:00","tz":"EDT","venue":"Gillette Stadium","venue_city":"Boston","team_a":"Haiti","team_b":"Scotland"},
    {"group":"C","date":"2026-06-19","time_local":"18:00","tz":"EDT","venue":"Gillette Stadium","venue_city":"Boston","team_a":"Scotland","team_b":"Morocco"},
    {"group":"C","date":"2026-06-19","time_local":"21:00","tz":"EDT","venue":"Lincoln Financial Field","venue_city":"Philadelphia","team_a":"Brazil","team_b":"Haiti"},
    {"group":"C","date":"2026-06-24","time_local":"18:00","tz":"EDT","venue":"Hard Rock Stadium","venue_city":"Miami","team_a":"Scotland","team_b":"Brazil"},
    {"group":"C","date":"2026-06-24","time_local":"18:00","tz":"EDT","venue":"Mercedes-Benz Stadium","venue_city":"Atlanta","team_a":"Morocco","team_b":"Haiti"},
    # === Group D ===
    {"group":"D","date":"2026-06-12","time_local":"21:00","tz":"EDT","venue":"SoFi Stadium","venue_city":"Los Angeles","team_a":"USA","team_b":"Paraguay"},
    {"group":"D","date":"2026-06-13","time_local":"00:00","tz":"EDT","venue":"BC Place","venue_city":"Vancouver","team_a":"Australia","team_b":"UEFA Play-off C"},
    {"group":"D","date":"2026-06-19","time_local":"15:00","tz":"EDT","venue":"Lumen Field","venue_city":"Seattle","team_a":"USA","team_b":"Australia"},
    {"group":"D","date":"2026-06-19","time_local":"21:00","tz":"EDT","venue":"Levi's Stadium","venue_city":"San Francisco Bay Area","team_a":"UEFA Play-off C","team_b":"Paraguay"},
    {"group":"D","date":"2026-06-25","time_local":"22:00","tz":"EDT","venue":"SoFi Stadium","venue_city":"Los Angeles","team_a":"UEFA Play-off C","team_b":"USA"},
    {"group":"D","date":"2026-06-25","time_local":"22:00","tz":"EDT","venue":"Levi's Stadium","venue_city":"San Francisco Bay Area","team_a":"Paraguay","team_b":"Australia"},
    # === Group E ===
    {"group":"E","date":"2026-06-14","time_local":"13:00","tz":"EDT","venue":"NRG Stadium","venue_city":"Houston","team_a":"Germany","team_b":"Curacao"},
    {"group":"E","date":"2026-06-14","time_local":"19:00","tz":"EDT","venue":"Lincoln Financial Field","venue_city":"Philadelphia","team_a":"Cote d'Ivoire","team_b":"Ecuador"},
    {"group":"E","date":"2026-06-20","time_local":"16:00","tz":"EDT","venue":"BMO Field","venue_city":"Toronto","team_a":"Germany","team_b":"Cote d'Ivoire"},
    {"group":"E","date":"2026-06-20","time_local":"20:00","tz":"EDT","venue":"GEHA Field at Arrowhead Stadium","venue_city":"Kansas City","team_a":"Ecuador","team_b":"Curacao"},
    {"group":"E","date":"2026-06-25","time_local":"16:00","tz":"EDT","venue":"MetLife Stadium","venue_city":"New York/New Jersey","team_a":"Ecuador","team_b":"Germany"},
    {"group":"E","date":"2026-06-25","time_local":"16:00","tz":"EDT","venue":"Lincoln Financial Field","venue_city":"Philadelphia","team_a":"Curacao","team_b":"Cote d'Ivoire"},
    # === Group F ===
    {"group":"F","date":"2026-06-14","time_local":"16:00","tz":"EDT","venue":"AT&T Stadium","venue_city":"Dallas","team_a":"Netherlands","team_b":"Japan"},
    {"group":"F","date":"2026-06-14","time_local":"21:00","tz":"EDT","venue":"Estadio BBVA","venue_city":"Monterrey","team_a":"UEFA Play-off B","team_b":"Tunisia"},
    {"group":"F","date":"2026-06-20","time_local":"13:00","tz":"EDT","venue":"NRG Stadium","venue_city":"Houston","team_a":"Netherlands","team_b":"UEFA Play-off B"},
    {"group":"F","date":"2026-06-21","time_local":"00:00","tz":"EDT","venue":"Estadio BBVA","venue_city":"Monterrey","team_a":"Tunisia","team_b":"Japan"},
    {"group":"F","date":"2026-06-25","time_local":"19:00","tz":"EDT","venue":"AT&T Stadium","venue_city":"Dallas","team_a":"Japan","team_b":"UEFA Play-off B"},
    {"group":"F","date":"2026-06-25","time_local":"19:00","tz":"EDT","venue":"GEHA Field at Arrowhead Stadium","venue_city":"Kansas City","team_a":"Tunisia","team_b":"Netherlands"},
    # === Group G ===
    {"group":"G","date":"2026-06-15","time_local":"21:00","tz":"EDT","venue":"SoFi Stadium","venue_city":"Los Angeles","team_a":"Iran","team_b":"New Zealand"},
    {"group":"G","date":"2026-06-15","time_local":"15:00","tz":"EDT","venue":"Lumen Field","venue_city":"Seattle","team_a":"Belgium","team_b":"Egypt"},
    {"group":"G","date":"2026-06-21","time_local":"15:00","tz":"EDT","venue":"SoFi Stadium","venue_city":"Los Angeles","team_a":"Belgium","team_b":"Iran"},
    {"group":"G","date":"2026-06-21","time_local":"21:00","tz":"EDT","venue":"BC Place","venue_city":"Vancouver","team_a":"New Zealand","team_b":"Egypt"},
    {"group":"G","date":"2026-06-26","time_local":"23:00","tz":"EDT","venue":"Lumen Field","venue_city":"Seattle","team_a":"Egypt","team_b":"Iran"},
    {"group":"G","date":"2026-06-26","time_local":"23:00","tz":"EDT","venue":"BC Place","venue_city":"Vancouver","team_a":"New Zealand","team_b":"Belgium"},
    # === Group H ===
    {"group":"H","date":"2026-06-15","time_local":"12:00","tz":"EDT","venue":"Mercedes-Benz Stadium","venue_city":"Atlanta","team_a":"Spain","team_b":"Cape Verde"},
    {"group":"H","date":"2026-06-15","time_local":"18:00","tz":"EDT","venue":"Hard Rock Stadium","venue_city":"Miami","team_a":"Saudi Arabia","team_b":"Uruguay"},
    {"group":"H","date":"2026-06-21","time_local":"12:00","tz":"EDT","venue":"Mercedes-Benz Stadium","venue_city":"Atlanta","team_a":"Spain","team_b":"Saudi Arabia"},
    {"group":"H","date":"2026-06-21","time_local":"18:00","tz":"EDT","venue":"Hard Rock Stadium","venue_city":"Miami","team_a":"Uruguay","team_b":"Cape Verde"},
    {"group":"H","date":"2026-06-26","time_local":"20:00","tz":"EDT","venue":"NRG Stadium","venue_city":"Houston","team_a":"Cape Verde","team_b":"Saudi Arabia"},
    {"group":"H","date":"2026-06-26","time_local":"20:00","tz":"EDT","venue":"Estadio Akron","venue_city":"Guadalajara","team_a":"Uruguay","team_b":"Spain"},
    # === Group I ===
    {"group":"I","date":"2026-06-16","time_local":"15:00","tz":"EDT","venue":"MetLife Stadium","venue_city":"New York/New Jersey","team_a":"France","team_b":"Senegal"},
    {"group":"I","date":"2026-06-16","time_local":"18:00","tz":"EDT","venue":"Gillette Stadium","venue_city":"Boston","team_a":"FIFA Play-off 2","team_b":"Norway"},
    {"group":"I","date":"2026-06-22","time_local":"17:00","tz":"EDT","venue":"Lincoln Financial Field","venue_city":"Philadelphia","team_a":"France","team_b":"FIFA Play-off 2"},
    {"group":"I","date":"2026-06-22","time_local":"20:00","tz":"EDT","venue":"MetLife Stadium","venue_city":"New York/New Jersey","team_a":"Norway","team_b":"Senegal"},
    {"group":"I","date":"2026-06-26","time_local":"15:00","tz":"EDT","venue":"Gillette Stadium","venue_city":"Boston","team_a":"Norway","team_b":"France"},
    {"group":"I","date":"2026-06-26","time_local":"15:00","tz":"EDT","venue":"BMO Field","venue_city":"Toronto","team_a":"Senegal","team_b":"FIFA Play-off 2"},
    # === Group J ===
    {"group":"J","date":"2026-06-16","time_local":"21:00","tz":"EDT","venue":"GEHA Field at Arrowhead Stadium","venue_city":"Kansas City","team_a":"Argentina","team_b":"Algeria"},
    {"group":"J","date":"2026-06-17","time_local":"00:00","tz":"EDT","venue":"Levi's Stadium","venue_city":"San Francisco Bay Area","team_a":"Austria","team_b":"Jordan"},
    {"group":"J","date":"2026-06-22","time_local":"13:00","tz":"EDT","venue":"AT&T Stadium","venue_city":"Dallas","team_a":"Argentina","team_b":"Austria"},
    {"group":"J","date":"2026-06-22","time_local":"23:00","tz":"EDT","venue":"Levi's Stadium","venue_city":"San Francisco Bay Area","team_a":"Jordan","team_b":"Algeria"},
    {"group":"J","date":"2026-06-27","time_local":"22:00","tz":"EDT","venue":"GEHA Field at Arrowhead Stadium","venue_city":"Kansas City","team_a":"Algeria","team_b":"Austria"},
    {"group":"J","date":"2026-06-27","time_local":"22:00","tz":"EDT","venue":"AT&T Stadium","venue_city":"Dallas","team_a":"Jordan","team_b":"Argentina"},
    # === Group K ===
    {"group":"K","date":"2026-06-17","time_local":"13:00","tz":"EDT","venue":"NRG Stadium","venue_city":"Houston","team_a":"Portugal","team_b":"FIFA Play-off 1"},
    {"group":"K","date":"2026-06-17","time_local":"22:00","tz":"EDT","venue":"Estadio Azteca","venue_city":"Mexico City","team_a":"Uzbekistan","team_b":"Colombia"},
    {"group":"K","date":"2026-06-23","time_local":"13:00","tz":"EDT","venue":"NRG Stadium","venue_city":"Houston","team_a":"Portugal","team_b":"Uzbekistan"},
    {"group":"K","date":"2026-06-23","time_local":"22:00","tz":"EDT","venue":"Estadio Akron","venue_city":"Guadalajara","team_a":"Colombia","team_b":"FIFA Play-off 1"},
    {"group":"K","date":"2026-06-27","time_local":"19:00","tz":"EDT","venue":"Hard Rock Stadium","venue_city":"Miami","team_a":"Colombia","team_b":"Portugal"},
    {"group":"K","date":"2026-06-27","time_local":"19:00","tz":"EDT","venue":"Mercedes-Benz Stadium","venue_city":"Atlanta","team_a":"FIFA Play-off 1","team_b":"Uzbekistan"},
    # === Group L ===
    {"group":"L","date":"2026-06-17","time_local":"16:00","tz":"EDT","venue":"AT&T Stadium","venue_city":"Dallas","team_a":"England","team_b":"Croatia"},
    {"group":"L","date":"2026-06-17","time_local":"19:00","tz":"EDT","venue":"BMO Field","venue_city":"Toronto","team_a":"Ghana","team_b":"Panama"},
    {"group":"L","date":"2026-06-23","time_local":"16:00","tz":"EDT","venue":"Gillette Stadium","venue_city":"Boston","team_a":"England","team_b":"Ghana"},
    {"group":"L","date":"2026-06-23","time_local":"19:00","tz":"EDT","venue":"BMO Field","venue_city":"Toronto","team_a":"Panama","team_b":"Croatia"},
    {"group":"L","date":"2026-06-27","time_local":"17:00","tz":"EDT","venue":"MetLife Stadium","venue_city":"New York/New Jersey","team_a":"Panama","team_b":"England"},
    {"group":"L","date":"2026-06-27","time_local":"17:00","tz":"EDT","venue":"Lincoln Financial Field","venue_city":"Philadelphia","team_a":"Croatia","team_b":"Ghana"},
]


def build():
    # 替换 play-off 占位符
    for m in RAW_SCHEDULE:
        if m["team_a"] in PLAYOFF_TEAM:
            m["team_a"] = PLAYOFF_TEAM[m["team_a"]]
        if m["team_b"] in PLAYOFF_TEAM:
            m["team_b"] = PLAYOFF_TEAM[m["team_b"]]
    
    # 添加全局 match_id（1-72，按时间排序）
    sorted_matches = sorted(RAW_SCHEDULE, key=lambda m: (m["date"], m["time_local"]))
    for i, m in enumerate(sorted_matches, 1):
        m["match_id"] = i
    # 恢复原顺序（按 group + date）
    for i, m in enumerate(RAW_SCHEDULE, 1):
        pass  # match_id 已经在 sorted_matches 里设置了
    
    output = {
        "_comment": "FIFA 2026 World Cup 72 场小组赛赛程（含日期/场馆/时间）",
        "_source": "kickoffadventures.com 派生 (2026.06.12 fetch)；play-off 占位符按已抽签结果替换",
        "_total_matches": len(RAW_SCHEDULE),
        "_playoff_mapping": PLAYOFF_TEAM,
        "matches": sorted_matches,
    }
    
    out_path = ROOT / "data" / "raw" / "group_schedule.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    
    # 校验
    print(f"✅ 写入 {out_path}")
    print(f"   总场次: {len(sorted_matches)}")
    
    # 按 group 统计
    from collections import Counter
    gc = Counter(m["group"] for m in sorted_matches)
    for g in sorted(gc):
        print(f"   {g} 组: {gc[g]} 场")
    
    # 检查所有球队都是 teams.json 中已知的
    teams = set(json.load(open(ROOT / "data" / "raw" / "teams.json"))["teams"].keys())
    unknown = set()
    for m in sorted_matches:
        if m["team_a"] not in teams:
            unknown.add(m["team_a"])
        if m["team_b"] not in teams:
            unknown.add(m["team_b"])
    if unknown:
        print(f"\n⚠ 未识别的球队: {unknown}")
    else:
        print(f"\n✅ 所有球队均能在 teams.json 中找到")


if __name__ == "__main__":
    build()
