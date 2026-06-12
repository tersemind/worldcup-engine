#!/usr/bin/env python3
"""
WorldCup Engine v1.2 — 一键全流程

依次执行：
1. 蒙特卡洛模拟（含决赛对阵收集）
2. 综合预测（自动读伤病库）
3. 三情景模拟
4. 套利信号生成
5. 多语言报告输出
"""
import sys
import time
import subprocess
from pathlib import Path

CODE = Path(__file__).parent


def run(cmd: list, label: str):
    print(f"\n{'='*80}")
    print(f"▶ {label}")
    print(f"{'='*80}")
    t0 = time.time()
    result = subprocess.run([sys.executable] + cmd, cwd=CODE.parent)
    elapsed = time.time() - t0
    print(f"\n⏱️  {label} 耗时: {elapsed:.1f} 秒")
    return result.returncode


def main(n_sim: int = 100000):
    print(f"\n🚀 WorldCup Engine v1.2 — 全流程执行（N={n_sim:,}）")
    
    t_start = time.time()
    
    # 1. 决赛对阵分析（自带 100k 蒙特卡洛 + 同时输出概率）
    run([str(CODE / "models" / "finals_analyzer.py"), str(n_sim)], "Step 1/4: 蒙特卡洛 + 决赛对阵分析")
    
    # 2. 综合预测（叠加伤病/情境/心理调整）
    run([str(CODE / "models" / "synthesizer.py")], "Step 2/4: 综合预测")
    
    # 3. 三情景平行（耗时较长，可跳过）
    if n_sim >= 50000:
        # 三情景每个跑 n_sim 次，会比较慢
        scenario_n = max(20000, n_sim // 5)  # 三情景用 1/5 模拟次数加速
        run([str(CODE / "run_scenarios.py"), str(scenario_n)], 
            f"Step 3/4: 三情景平行模拟（每情景 {scenario_n:,} 次）")
    else:
        print("\n⏭️  跳过三情景（n_sim < 50k）")
    
    # 4. 套利信号
    run([str(CODE / "models" / "arbitrage.py"), "10000"], "Step 4/5: 套利信号生成")
    
    # 5. 三语报告
    run([str(CODE / "models" / "report_writer.py"), "all"], "Step 5/5: 多语言报告")
    
    elapsed = time.time() - t_start
    print(f"\n{'='*80}")
    print(f"🎉 全流程完成！总耗时 {elapsed:.1f} 秒")
    print(f"{'='*80}")
    print(f"\n📁 输出文件：")
    print(f"  - mc_simulation_n{n_sim}.json     # 蒙特卡洛")
    print(f"  - finals_matchups.json            # 决赛对阵")
    print(f"  - synthesizer_report.json         # 综合预测")
    print(f"  - three_scenarios.json            # 三情景")
    print(f"  - arbitrage_signals.json          # 套利信号")
    print(f"  - report_zh.md / report_en.md / report_es.md  # 三语报告")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
    main(n)
