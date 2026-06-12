"""
决赛对阵概率分析器（#20）

输出：
- 所有可能的决赛对阵 + 联合概率
- 决赛对阵热力图（文本版）
- Top N 最可能决赛
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.monte_carlo import run_monte_carlo
from utils.io import save_output


def analyze_finals(n_simulations: int = 100000, top_n: int = 20):
    """运行带决赛对阵收集的蒙特卡洛"""
    print("=" * 80)
    print(f"🏆 决赛对阵概率分析 (N={n_simulations:,})")
    print("=" * 80)
    
    t0 = time.time()
    probs, finals = run_monte_carlo(
        n_simulations=n_simulations, return_finals=True, verbose=True
    )
    t1 = time.time()
    print(f"\n⏱️  耗时: {t1-t0:.1f} 秒\n")
    
    # 输出 Top N 决赛对阵
    print(f"📊 Top {top_n} 最可能决赛对阵：\n")
    print(f"  {'Rank':<5} {'对阵':<35} {'概率':<8} {'次数':<8}")
    print("  " + "-" * 60)
    for i, m in enumerate(finals[:top_n], 1):
        print(f"  {i:<5} {m['team_a']:<14} vs {m['team_b']:<14} "
              f"{m['probability']*100:>5.2f}%   {m['count']:>5}")
    
    # 标注 Reference 的预测
    print(f"\n📌 参考报告 Top 3 决赛对阵预测：")
    print(f"  1. Argentina vs Spain  (参考 2.72%)")
    print(f"  2. Argentina vs France (参考 2.16%)")
    print(f"  3. England vs Spain    (参考 1.83%)")
    
    # 总决赛参赛者次数（双重计数）
    total_finalists = sum(m["count"] * 2 for m in finals)
    print(f"\n📌 总决赛参赛次数（应 = {n_simulations*2}）: {total_finalists}")
    
    # 决赛参赛概率排行（每队进决赛 N 次）
    final_appearance = {}
    for m in finals:
        final_appearance[m["team_a"]] = final_appearance.get(m["team_a"], 0) + m["count"]
        final_appearance[m["team_b"]] = final_appearance.get(m["team_b"], 0) + m["count"]
    
    print(f"\n📊 进决赛 Top 10 球队：\n")
    sorted_finalists = sorted(final_appearance.items(), key=lambda x: -x[1])
    for i, (team, count) in enumerate(sorted_finalists[:10], 1):
        prob = count / n_simulations
        print(f"  {i}. {team:<14} {prob*100:>5.2f}% ({count:>5} 次)")
    
    # 文本热力图（Top 8 球队对 Top 8 球队的对阵概率）
    print_heatmap(finals, n_simulations)
    
    # 保存
    output = {
        "n_simulations": n_simulations,
        "total_unique_matchups": len(finals),
        "top_matchups": finals[:50],
        "final_appearance": dict(sorted_finalists),
    }
    save_output("finals_matchups.json", output)
    print(f"\n✅ 已保存到 data/outputs/finals_matchups.json")
    
    return probs, finals


def print_heatmap(finals: list, n_sim: int):
    """文本版决赛对阵热力图（Top 8 队）"""
    # 找出现频次 Top 8 球队
    appearance = {}
    for m in finals:
        appearance[m["team_a"]] = appearance.get(m["team_a"], 0) + m["count"]
        appearance[m["team_b"]] = appearance.get(m["team_b"], 0) + m["count"]
    top_8 = [t for t, _ in sorted(appearance.items(), key=lambda x: -x[1])[:8]]
    
    # 构建概率矩阵
    matrix = {ta: {tb: 0.0 for tb in top_8} for ta in top_8}
    for m in finals:
        ta, tb = m["team_a"], m["team_b"]
        if ta in top_8 and tb in top_8:
            matrix[ta][tb] = m["probability"] * 100
            matrix[tb][ta] = m["probability"] * 100
    
    # 打印
    print(f"\n🔥 决赛对阵热力图（Top 8 球队，单位 %）：\n")
    abbr = {t: t[:3].upper() for t in top_8}
    
    # 表头
    header = " " * 14 + "  ".join(f"{abbr[t]:>5}" for t in top_8)
    print(f"  {header}")
    
    # 每行
    for ta in top_8:
        row = f"  {ta:<13}"
        for tb in top_8:
            v = matrix[ta][tb]
            if ta == tb:
                row += f"  {'  -  ':>5}"
            elif v >= 1.0:
                row += f"  🔥{v:>3.1f}"
            elif v >= 0.5:
                row += f"  ▓{v:>4.2f}"
            elif v >= 0.1:
                row += f"  ░{v:>4.2f}"
            else:
                row += f"  {v:>5.2f}"
        print(row)
    
    print(f"\n  说明：🔥 ≥1% (热门决赛) | ▓ ≥0.5% | ░ ≥0.1% | 数字 <0.1%")


def main(n_sim: int = 100000):
    analyze_finals(n_sim)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
    main(n)
