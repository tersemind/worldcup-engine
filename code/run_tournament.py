#!/usr/bin/env python3
"""
WorldCup Predict 主入口：完整跑通
  Step 1: 蒙特卡洛模拟（默认 100,000 次）
  Step 2: 综合预测（叠加情境调整）
  Step 3: 输出报告
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from models.monte_carlo import run_monte_carlo
from models.synthesizer import synthesize, print_report
from utils.io import save_output


def main(n_sim: int = 100000, seed: int = 42):
    print("=" * 80)
    print(f"🚀 WorldCup Predict v1.0 — Tournament Prediction")
    print(f"   蒙特卡洛模拟次数: {n_sim:,}")
    print(f"   随机种子: {seed}")
    print("=" * 80)
    
    # Step 1: 蒙特卡洛
    t0 = time.time()
    print("\n📊 Step 1/2: 执行蒙特卡洛模拟...")
    mc_probs = run_monte_carlo(n_simulations=n_sim, seed=seed, verbose=True)
    t1 = time.time()
    print(f"   完成耗时：{t1-t0:.1f} 秒")
    
    save_output(f"mc_simulation_n{n_sim}.json", mc_probs)
    
    # Step 2: 综合预测
    print("\n🎯 Step 2/2: 执行综合预测（叠加情境/伤病/心理调整）...")
    results = synthesize(mc_probs)
    
    save_output("synthesizer_report.json", results)
    
    # 输出报告
    print_report(results, top_n=16)
    
    t2 = time.time()
    print(f"\n⏱️  总耗时：{t2-t0:.1f} 秒")
    print(f"📁 输出文件：")
    print(f"   - data/outputs/mc_simulation_n{n_sim}.json")
    print(f"   - data/outputs/synthesizer_report.json")
    
    return results


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    main(n, seed)
