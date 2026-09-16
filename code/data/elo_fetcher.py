"""
真实的 Elo 数据抓取器
来源：eloratings.net 隐藏 TSV 接口（World.tsv）

TSV 列定义（前 4 列即可）:
  col[0] = current rank (overall)
  col[1] = current rank (limited?)
  col[2] = ISO country code (e.g. ES, AR, FR)
  col[3] = current Elo rating  ← 我们要的字段
  ...其他列为历史数据
"""
import requests
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=Warning)

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.io import DATA_RAW

# ISO 国家代码 → teams.json 中的球队名映射
# 仅含 2026 World Cup 48 强 + 几个常见国家
ISO_TO_TEAM = {
    "ES": "Spain",
    "AR": "Argentina",
    "FR": "France",
    "EN": "England",
    "BR": "Brazil",
    "PT": "Portugal",
    "CO": "Colombia",
    "NL": "Netherlands",
    "EC": "Ecuador",
    "DE": "Germany",
    "NO": "Norway",
    "HR": "Croatia",
    "TR": "Turkiye",
    "JP": "Japan",
    "BE": "Belgium",
    "UY": "Uruguay",
    "CH": "Switzerland",
    "MX": "Mexico",
    "SN": "Senegal",
    "PY": "Paraguay",
    "AT": "Austria",
    "MA": "Morocco",
    "CA": "Canada",
    "UA": "Ukraine",  # 未参赛
    "AU": "Australia",
    "DZ": "Algeria",
    "IR": "Iran",
    "KR": "Korea Republic",
    "CZ": "Czechia",
    "RS": "Serbia",  # 未参赛
    "PA": "Panama",
    "US": "USA",
    "UZ": "Uzbekistan",
    "SE": "Sweden",
    "EG": "Egypt",
    "CI": "Cote d'Ivoire",
    "TN": "Tunisia",
    "JO": "Jordan",
    "CD": "DR Congo",
    "BA": "Bosnia",
    "CV": "Cape Verde",
    "SA": "Saudi Arabia",
    "IQ": "Iraq",
    "QA": "Qatar",
    "ZA": "South Africa",
    "GH": "Ghana",
    "NZ": "New Zealand",
    "HT": "Haiti",
    "CW": "Curacao",
}


def fetch_elo_tsv(url: str = "https://www.eloratings.net/World.tsv", timeout: int = 15) -> list:
    """
    从 eloratings.net 抓取 TSV 原始数据
    返回行列表（每行是字符串数组）
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Accept": "text/tab-separated-values,*/*",
        "Referer": "https://www.eloratings.net/World",
    }
    
    print(f"📡 正在抓取: {url}")
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    
    text = r.text
    rows = []
    for line in text.splitlines():
        cols = line.split("\t")
        if len(cols) >= 4:
            rows.append(cols)
    
    print(f"✅ 抓到 {len(rows)} 行数据")
    return rows


def parse_elo_data(rows: list) -> dict:
    """
    解析 TSV 数据，返回 {country_iso: elo_rating}
    """
    elo_dict = {}
    for cols in rows:
        try:
            iso = cols[2].strip()
            elo = int(cols[3])
            elo_dict[iso] = elo
        except (ValueError, IndexError):
            continue
    return elo_dict


def map_to_teams(elo_dict: dict, teams_data: dict) -> dict:
    """
    把 ISO Elo 映射到 teams.json 中的球队名
    返回 {team_name: new_elo}
    """
    updates = {}
    for iso, team_name in ISO_TO_TEAM.items():
        if iso in elo_dict and team_name in teams_data:
            updates[team_name] = elo_dict[iso]
    return updates


def diff_and_update(updates: dict, teams_data: dict, threshold: int = 1) -> list:
    """
    对比新旧 Elo，返回变化清单
    threshold: 变化超过多少分才更新（避免微小波动）
    """
    changes = []
    for team, new_elo in updates.items():
        old_elo = teams_data[team].get("elo", 0)
        if abs(new_elo - old_elo) >= threshold:
            changes.append({
                "team": team,
                "old": old_elo,
                "new": new_elo,
                "diff": new_elo - old_elo
            })
    return changes


def write_updates(changes: list) -> int:
    """
    把变化批量写回 teams.json
    """
    file_path = DATA_RAW / "teams.json"
    with open(file_path, "r") as f:
        data = json.load(f)
    
    if "_last_updated" not in data:
        data["_last_updated"] = {}
    
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    success = 0
    
    for c in changes:
        team = c["team"]
        if team not in data["teams"]:
            continue
        data["teams"][team]["elo"] = c["new"]
        if team not in data["_last_updated"]:
            data["_last_updated"][team] = {}
        data["_last_updated"][team]["elo"] = timestamp
        success += 1
    
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return success


def main(dry_run: bool = False):
    """
    主流程：抓 → 解析 → 对比 → 更新
    """
    print("=" * 60)
    print("🔢 Elo Fetcher v1.0 — eloratings.net 实时抓取")
    print("=" * 60)
    
    # 1. 抓取
    rows = fetch_elo_tsv()
    
    # 2. 解析
    elo_dict = parse_elo_data(rows)
    print(f"\n📊 解析得到 {len(elo_dict)} 个国家队 Elo 数据")
    
    # 3. 加载现有 teams.json
    with open(DATA_RAW / "teams.json", "r") as f:
        teams_data = json.load(f)["teams"]
    
    # 4. 映射 + 对比
    updates = map_to_teams(elo_dict, teams_data)
    print(f"📋 匹配到 {len(updates)} 支参赛球队")
    
    changes = diff_and_update(updates, teams_data, threshold=1)
    
    if not changes:
        print("\n✅ 所有球队 Elo 已是最新，无变化")
        return
    
    print(f"\n🔄 检测到 {len(changes)} 支球队 Elo 变化：\n")
    print(f"{'Team':<18} {'Old':<6} → {'New':<6}  {'Diff':<8}")
    print("-" * 45)
    for c in sorted(changes, key=lambda x: -abs(x["diff"])):
        sign = "+" if c["diff"] > 0 else ""
        print(f"{c['team']:<18} {c['old']:<6}   {c['new']:<6}  {sign}{c['diff']}")
    
    # 5. 写入
    if dry_run:
        print("\n🚫 dry-run 模式，未写入文件")
    else:
        n = write_updates(changes)
        print(f"\n✅ 已写入 teams.json，更新 {n} 个字段")
        print(f"📅 时间戳：{time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    main(dry_run=dry_run)
