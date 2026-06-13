"""
AI-weighted Monte Carlo 入口（阶段 1 / 阶段 2 验证脚本）
==============================================

独立于 scheduler 运行：
  1. 读 synthesizer_report.json（不修改）
  2. 阶段 1：反求每队 ΔE（含 8 项 adj 总和）
     阶段 2：分离事件级 ΔE（4 项队级 adj）+ 比赛级 ΔE_per_match（4 项 per-match adj）
  3. 跑 run_monte_carlo(elo_shifts=..., match_shifts=...)
  4. 输出到 mc_simulation_n{N}_ai.json
  5. 打印对比报告：AI-MC vs synthesizer.final_probability

用法：
  python3 code/run_ai_weighted_mc.py            # 默认 N=100000，阶段 2
  python3 code/run_ai_weighted_mc.py 30000      # 自定义 N
  python3 code/run_ai_weighted_mc.py 100000 12  # 指定 ELO_PER_PP=12
  python3 code/run_ai_weighted_mc.py 100000 10 phase1  # 强制阶段 1（队级聚合）
  python3 code/run_ai_weighted_mc.py 100000 10 phase3  # 阶段 3（分段 ELO_PER_PP）

阶段 3 分段系数可由环境变量覆盖：
  AI_WEIGHTED_ELO_PER_PP_HIGH (default 12.5)  # mc_base ≥ 8pp
  AI_WEIGHTED_ELO_PER_PP_MID  (default 10.0)  # mc_base 3-8pp
  AI_WEIGHTED_ELO_PER_PP_LOW  (default  8.0)  # mc_base < 3pp
"""
from __future__ import annotations
import sys
import os
import json
import time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from models.monte_carlo import run_monte_carlo
from models.ai_weighted_baseline import (
    compute_elo_shifts, explain_shifts,
    compute_event_level_elo_shifts, compute_match_level_shifts,
    compute_event_level_elo_shifts_segmented, compute_match_level_shifts_segmented,
    DEFAULT_ELO_PER_PP_HIGH, DEFAULT_ELO_PER_PP_MID, DEFAULT_ELO_PER_PP_LOW,
)
from utils.io import save_output, load_teams


def main():
    # 解析参数
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
    if len(sys.argv) > 2:
        os.environ["AI_WEIGHTED_ELO_PER_PP"] = sys.argv[2]
    elo_per_pp = float(os.environ.get("AI_WEIGHTED_ELO_PER_PP", "10.0"))
    # phase: phase1 = 阶段 1 (队级 8 项聚合)；其余 = 阶段 2 (事件级队级 + 比赛级 per-match)
    phase = sys.argv[3] if len(sys.argv) > 3 else "phase2"

    print("=" * 70)
    print(f"AI-Weighted Monte Carlo  ({phase.upper()})")
    print("=" * 70)
    print(f"  N simulations:  {n:,}")
    print(f"  ELO_PER_PP:     {elo_per_pp}")
    print(f"  Mode:           {phase}")
    print()

    # Step 1: 计算 Elo shifts
    if phase == "phase1":
        print("[1/4] 阶段 1：从 synthesizer 反求队级 ΔE（8 项聚合）...")
        shifts = compute_elo_shifts()
        match_shifts = None
    elif phase == "phase3":
        print(f"[1/4] 阶段 3：分段 ELO_PER_PP（HIGH={DEFAULT_ELO_PER_PP_HIGH} "
              f"MID={DEFAULT_ELO_PER_PP_MID} LOW={DEFAULT_ELO_PER_PP_LOW}）...")
        shifts = compute_event_level_elo_shifts_segmented()
        try:
            teams = load_teams()
        except Exception:
            teams = None
        match_shifts = compute_match_level_shifts_segmented(teams_data=teams)
        print(f"      ✓ 事件级 {len(shifts)} 队 / 比赛级 {len(match_shifts)} 场（分段反求）")
    else:
        print("[1/4] 阶段 2：分离事件级（队级）+ 比赛级（per-match）ΔE...")
        shifts = compute_event_level_elo_shifts()
        try:
            teams = load_teams()
        except Exception:
            teams = None
        match_shifts = compute_match_level_shifts(teams_data=teams)
        print(f"      ✓ 事件级 {len(shifts)} 队 / 比赛级 {len(match_shifts)} 场")

    if not shifts:
        print("❌ synthesizer_report.json 不可读或为空，退出")
        return 1
    print(f"      ✓ 队级 {len(shifts)} 队")
    print()
    print(explain_shifts(shifts, top_n=8))
    print()

    if match_shifts:
        print(f"  比赛级 Top 6（绝对 ΔE 最大）:")
        sorted_ms = sorted(match_shifts.items(),
                            key=lambda x: -(abs(x[1][0]) + abs(x[1][1])))
        for (a, b), (ea, eb) in sorted_ms[:6]:
            print(f"    {a:<18} vs {b:<18}  ΔE_a={ea:+5.1f}  ΔE_b={eb:+5.1f}")
        print()

    # Step 2: 跑 AI-weighted MC
    print(f"[2/4] 运行 AI-weighted MC (N={n:,})...")
    t0 = time.time()
    probs = run_monte_carlo(
        n_simulations=n,
        seed=42,
        verbose=(n >= 10000),
        elo_shifts=shifts,
        match_shifts=match_shifts,
    )
    dt = time.time() - t0
    print(f"      ✓ 耗时 {dt:.1f}s")
    print()

    # Step 3: 保存到独立文件（阶段 3 单独存，便于对比阶段 2）
    if phase == "phase3":
        out_name = f"mc_simulation_n{n}_ai_phase3.json"
    else:
        out_name = f"mc_simulation_n{n}_ai.json"
    print(f"[3/4] 保存到 data/outputs/{out_name}...")
    save_output(out_name, probs)
    print(f"      ✓")
    print()

    # Step 4: 对比报告
    print(f"[4/4] 对比报告：AI-MC vs synthesizer.final_probability")
    print()
    synth = json.load(open(ROOT.parent / "data" / "outputs" / "synthesizer_report.json"))
    mc_orig = json.load(open(ROOT.parent / "data" / "outputs" / f"mc_simulation_n{n}.json")) \
              if (ROOT.parent / "data" / "outputs" / f"mc_simulation_n{n}.json").exists() else {}

    # 按 AI-MC champion 排序
    sorted_teams = sorted(probs.items(), key=lambda x: -x[1]["champion"])

    print(f"{'Rank':<5} {'Team':<18} {'mc_base%':>9} {'synth_final%':>13} "
          f"{'AI-MC%':>8} {'ΔElo':>7} {'AI-synth':>10}")
    print("-" * 80)
    for rank, (team, p) in enumerate(sorted_teams[:16], 1):
        mc_base = mc_orig.get(team, {}).get("champion", 0) * 100
        synth_final = synth.get(team, {}).get("final_probability", 0)
        ai_mc = p["champion"] * 100
        delta_elo = shifts.get(team, 0)
        ai_synth_diff = ai_mc - synth_final
        print(f"{rank:<5} {team:<18} {mc_base:>8.2f}% {synth_final:>12.2f}% "
              f"{ai_mc:>7.2f}% {delta_elo:>+7.1f} {ai_synth_diff:>+9.2f}pp")
    print()

    # 精度统计：AI-MC vs synth 的 MAE
    diffs = []
    for team, p in probs.items():
        synth_v = synth.get(team, {}).get("final_probability", 0)
        diffs.append(abs(p["champion"] * 100 - synth_v))
    if diffs:
        mae = sum(diffs) / len(diffs)
        max_d = max(diffs)
        print(f"AI-MC 与 synthesizer.final_probability 一致性：")
        print(f"  MAE  = {mae:.2f} pp")
        print(f"  Max  = {max_d:.2f} pp")
        print(f"  → MAE < 1pp 表示反求解器精度良好；> 3pp 需要调整 ELO_PER_PP")

    # SF Top 4 对比
    print()
    print("半决赛 Top 4（AI-MC 口径下）：")
    sf_top = sorted(probs.items(), key=lambda x: -x[1]["semifinal"])[:4]
    for r, (t, p) in enumerate(sf_top, 1):
        synth_sf = synth.get(t, {}).get("advance_to_sf", 0)
        print(f"  {r}. {t:<16} AI-MC SF={p['semifinal']*100:.2f}%  synth SF={synth_sf:.2f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
