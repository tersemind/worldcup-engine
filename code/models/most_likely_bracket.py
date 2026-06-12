"""
最可能剧本采样器（方案 B）
==========================
从 MC N 次模拟里，对每个 match_id 统计「(team_a, team_b, winner)」频次，
取频次最高者作为该场最可能对阵。

输出：data/outputs/most_likely_bracket.json，供 tournament_api 读取，
替换原来的"贪心 Top-1"推演逻辑。

关键设计：
- match_id 是 bracket 固定槽位，跨次模拟稳定
- 但同一个 slot 在不同模拟中可能由不同球队占据（黑马打掉强队后晋级）
- 我们对 (team_a, team_b) 这个有序对做计数，找出最常出现的
- 频次本身就是该对阵的实际发生概率（占总模拟数比例）

用法：
    python3 code/models/most_likely_bracket.py 100000
"""
import sys
import json
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.monte_carlo import simulate_tournament
from utils.io import load_teams, load_groups, save_output


def run(n_simulations: int = 100000, seed: int = 42):
    """
    跑 N 次完整 MC，对每个 match_id 收集 (team_a, team_b, winner) 频次。
    
    输出结构：
    {
        "n_simulations": 100000,
        "method": "most-likely matchup per slot (sample mode of MC)",
        "rounds": {
            "r32": [
                {
                    "match_id": 73, "team_a": "Canada", "team_b": "Morocco",
                    "winner": "Morocco",
                    "count": 32847, "prob_pct": 32.85,
                    "alt_matchups": [["Mexico", "Morocco", 18403], ...]  # 候选 Top 5
                },
                ...
            ],
            "r16": [...], "qf": [...], "sf": [...], "final": {...}
        }
    }
    """
    teams = load_teams()
    groups = load_groups()
    rng = np.random.default_rng(seed)
    
    # 每个 match_id 计数器： (team_a, team_b) → count
    # 注：球队对是无序的（A vs B == B vs A），用 frozenset 但保留顺序信息时存有序 tuple
    matchup_count = defaultdict(Counter)   # {match_id: Counter({(a,b): n})}
    winner_count  = defaultdict(Counter)   # {match_id: Counter({((a,b), winner): n})}
    round_of = {}                          # {match_id: "r32"|"r16"|"qf"|"sf"|"final"}
    
    print(f"=== 跑 {n_simulations} 次 MC 采样，统计每场最可能对阵 ===")
    progress_step = max(1, n_simulations // 20)
    
    for i in range(n_simulations):
        if (i + 1) % progress_step == 0:
            pct = (i + 1) * 100 // n_simulations
            print(f"  Progress: {i+1}/{n_simulations} ({pct}%)")
        
        _, bracket_record = simulate_tournament(teams, groups, rng, return_bracket=True)
        if bracket_record is None:
            continue
        
        for round_name in ("r32", "r16", "qf", "sf"):
            for mid, ta, tb, w in bracket_record[round_name]:
                # key 加 round 前缀，避免不同轮次 match_id 撞号（r16=1 vs qf=1）
                key = (round_name, mid)
                pair = tuple(sorted([ta, tb]))   # 标准化对阵（A vs B == B vs A）
                matchup_count[key][pair] += 1
                winner_count[key][(pair, w)] += 1
                round_of[key] = (round_name, mid)
        
        # final 单独处理
        ta, tb, w = bracket_record["final"]
        key = ("final", 1)
        pair = tuple(sorted([ta, tb]))
        matchup_count[key][pair] += 1
        winner_count[key][(pair, w)] += 1
        round_of[key] = ("final", 1)
    
    # 聚合：每个 (round, match_id) 取频次最高的对阵 + 对应胜方
    result = {"r32": [], "r16": [], "qf": [], "sf": [], "final": None}
    
    for key, counter in matchup_count.items():
        round_name, mid = key
        top_pair, top_count = counter.most_common(1)[0]
        ta, tb = top_pair
        
        # 在这个对阵下找最常见胜方
        w_a = winner_count[key].get((top_pair, ta), 0)
        w_b = winner_count[key].get((top_pair, tb), 0)
        winner = ta if w_a >= w_b else tb
        winner_prob = max(w_a, w_b) / top_count if top_count > 0 else 0
        
        # 备选对阵 Top 5
        alts = [
            {"team_a": p[0], "team_b": p[1], "count": c,
             "prob_pct": round(c * 100 / n_simulations, 2)}
            for p, c in counter.most_common(5)
        ]
        
        entry = {
            "match_id": mid,
            "team_a": ta,
            "team_b": tb,
            "winner": winner,
            "count": top_count,
            "prob_pct": round(top_count * 100 / n_simulations, 2),
            "winner_conditional_pct": round(winner_prob * 100, 1),
            "alt_matchups": alts,
        }
        if round_name == "final":
            result["final"] = entry
        else:
            result[round_name].append(entry)
    
    # 按 match_id 排序
    for r in ("r32", "r16", "qf", "sf"):
        result[r].sort(key=lambda x: x["match_id"])
    
    output = {
        "n_simulations": n_simulations,
        "method": "最可能剧本：每个 match_id 取 MC 出现频次最高的对阵 + 该对阵下最常胜方",
        "rounds": result,
    }
    
    save_output("most_likely_bracket.json", output)
    print(f"\n已保存到: data/outputs/most_likely_bracket.json")
    
    # 打印摘要
    print("\n=== 摘要 ===")
    for r_name, r_label in [("r32", "1/16 决赛"), ("r16", "1/8 决赛"),
                             ("qf", "1/4 决赛"), ("sf", "半决赛")]:
        if result[r_name]:
            avg = sum(m["prob_pct"] for m in result[r_name]) / len(result[r_name])
            min_m = min(result[r_name], key=lambda x: x["prob_pct"])
            print(f"{r_label}: {len(result[r_name])} 场，"
                  f"最可能对阵平均概率 {avg:.1f}%，"
                  f"最不确定 M{min_m['match_id']} = {min_m['team_a']} vs {min_m['team_b']} ({min_m['prob_pct']}%)")
    
    if result["final"]:
        f = result["final"]
        print(f"\n决赛最可能: {f['team_a']} vs {f['team_b']} ({f['prob_pct']}%) "
              f"→ {f['winner']} 夺冠 (条件概率 {f['winner_conditional_pct']}%)")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
    run(n_simulations=n)
