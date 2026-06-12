"""
小组第 1 名概率分布采样器
==========================
从 MC N 次模拟里，统计每个小组里每支队伍排第 1 的频次，
得到「第一概率」用于替换原本"按 Elo 直接排序"的做法。

输出：data/outputs/group_rank_dist.json
{
  "n_simulations": 100000,
  "groups": {
    "K": {
      "Portugal":  {"first_pct": 38.5, "first_count": 38470},
      "Colombia":  {"first_pct": 35.7, ...},
      "Saudi Arabia": {...},
      "Iran": {...}
    },
    ...
  }
}
"""
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.monte_carlo import simulate_tournament
from utils.io import load_teams, load_groups, save_output


def run(n_simulations: int = 100000, seed: int = 42):
    teams = load_teams()
    groups = load_groups()
    rng = np.random.default_rng(seed)

    # {group_id: {team: count_of_being_first}}
    first_count = defaultdict(lambda: defaultdict(int))

    print(f"=== 跑 {n_simulations} 次 MC，统计每队小组第 1 名频次 ===")
    progress_step = max(1, n_simulations // 20)

    for i in range(n_simulations):
        if (i + 1) % progress_step == 0:
            pct = (i + 1) * 100 // n_simulations
            print(f"  Progress: {i+1}/{n_simulations} ({pct}%)")

        _, bracket_record = simulate_tournament(teams, groups, rng, return_bracket=True)
        if bracket_record is None:
            continue

        gs = bracket_record.get("group_standings", {})
        for g_id, standings in gs.items():
            # standings = [(team, rank, pts, gd, gf), ...]
            for team, rank, _, _, _ in standings:
                if rank == 1:
                    first_count[g_id][team] += 1

    # 聚合
    out_groups = {}
    for g_id, team_counts in first_count.items():
        # 该组所有出现过的球队（理论上 4 支都会出现）
        all_teams_in_group = sorted(groups[g_id])
        per_team = {}
        for t in all_teams_in_group:
            c = team_counts.get(t, 0)
            per_team[t] = {
                "first_count": c,
                "first_pct": round(c * 100 / n_simulations, 2),
            }
        out_groups[g_id] = per_team

    output = {
        "n_simulations": n_simulations,
        "method": "每队小组第 1 名概率 = MC 该队排第 1 的频次 / 总模拟数",
        "groups": out_groups,
    }
    save_output("group_rank_dist.json", output)
    print("\n已保存到: data/outputs/group_rank_dist.json")

    # 摘要
    print("\n=== 摘要（每组 Top 1 第一概率最高的球队） ===")
    for g_id in sorted(out_groups):
        team_probs = out_groups[g_id]
        top = sorted(team_probs.items(), key=lambda x: -x[1]["first_pct"])
        line = "  ".join(f"{t}={d['first_pct']:>5.1f}%" for t, d in top)
        print(f"{g_id}: {line}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
    run(n_simulations=n)
