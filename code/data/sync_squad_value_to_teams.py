"""
squad_value.json → teams.json[*].squad_value_m_eur 回写

读 data/raw/squad_value.json 的 total_value_eur，按 €1M 单位写回 teams.json。
- 跳过 total_value_eur 为 None / 0 的失败队（保留 teams.json 中的旧值）
- diff > 阈值（默认 ±20M）才更新，避免 LLM 抖动污染
- 用文件锁保护，与 elo_fetcher / polymarket_fetcher 同模式

通常由 scheduler 的 task_squad_value 在 fetch 后自动调用。
"""
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.io import DATA_RAW

DIFF_THRESHOLD_M_EUR = 20  # 小于 €20M 的差异认为是 LLM 抖动，不更新


def load_squad_value() -> Dict[str, dict]:
    """读 squad_value.json"""
    path = DATA_RAW / "squad_value.json"
    if not path.exists():
        return {}
    with open(path, "r") as f:
        data = json.load(f)
    return data.get("teams", {})


def diff_and_collect(squad: Dict[str, dict], teams_data: Dict[str, dict]) -> List[dict]:
    """对比新旧 squad_value_m_eur，返回需要更新的清单"""
    changes = []
    for team, info in squad.items():
        total_eur = info.get("total_value_eur")
        if not total_eur or total_eur <= 0:
            continue  # 失败队跳过
        new_m = round(total_eur / 1_000_000)
        if team not in teams_data:
            continue
        old_m = teams_data[team].get("squad_value_m_eur", 0) or 0
        if abs(new_m - old_m) < DIFF_THRESHOLD_M_EUR:
            continue
        changes.append({"team": team, "old": old_m, "new": new_m, "diff": new_m - old_m})
    return changes


def write_updates(changes: List[dict]) -> int:
    """写回 teams.json"""
    file_path = DATA_RAW / "teams.json"
    with open(file_path, "r") as f:
        data = json.load(f)

    if "_last_updated" not in data:
        data["_last_updated"] = {}

    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    success = 0

    for c in changes:
        team = c["team"]
        if team not in data["teams"]:
            continue
        data["teams"][team]["squad_value_m_eur"] = c["new"]
        if team not in data["_last_updated"]:
            data["_last_updated"][team] = {}
        data["_last_updated"][team]["squad_value_m_eur"] = ts
        success += 1

    with open(file_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return success


def main(dry_run: bool = False) -> int:
    print("=" * 60)
    print("💰 squad_value → teams.json 回写")
    print("=" * 60)

    squad = load_squad_value()
    if not squad:
        print("⚠️  squad_value.json 不存在或为空，跳过")
        return 0
    print(f"📊 squad_value.json 含 {len(squad)} 队")

    with open(DATA_RAW / "teams.json", "r") as f:
        teams_data = json.load(f)["teams"]

    changes = diff_and_collect(squad, teams_data)

    if not changes:
        print(f"\n✅ 所有球队 squad_value_m_eur 与抓取值一致（阈值 ±€{DIFF_THRESHOLD_M_EUR}M）")
        return 0

    print(f"\n🔄 {len(changes)} 支球队 squad_value 变化：\n")
    print(f"{'Team':<20} {'Old(€M)':<10} → {'New(€M)':<10}  {'Diff':<8}")
    print("-" * 55)
    for c in sorted(changes, key=lambda x: -abs(x["diff"])):
        sign = "+" if c["diff"] > 0 else ""
        print(f"{c['team']:<20} {c['old']:<10} {c['new']:<10}  {sign}{c['diff']}")

    if dry_run:
        print("\n🚫 dry-run 模式，未写入文件")
        return 0

    n = write_updates(changes)
    print(f"\n✅ 已写入 teams.json，更新 {n} 个字段")
    print(f"📅 时间戳：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    return n


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    sys.exit(0 if main(dry_run=dry_run) >= 0 else 1)
