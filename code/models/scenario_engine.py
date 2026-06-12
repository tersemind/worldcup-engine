"""
三情景平行模拟引擎（参考方法论 表 2.3）

三种情景对核心参数做不同扰动：

参数            | Bull (乐观)    | Base (基准)    | Bear (悲观)
----------------|----------------|----------------|----------------
核心球员伤病     | 无伤病(0%)      | 历史均值(5%)    | 大规模伤病潮(15%)
强队优势         | +2% boost      | 中性            | -2% penalty
冷门概率         | -25%            | 基础            | +50%
ELO 动态调整     | 强者更强         | 当前值           | 重置为 1 年前
天气/海拔影响    | 无极端           | 历史均值         | 持续极端

每个情景独立跑 N 次蒙特卡洛，输出三套概率分布
"""
import numpy as np
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.monte_carlo import simulate_tournament
from models.bracket_engine import load_bracket
from utils.io import load_teams, load_groups, save_output


# ============ 情景参数配置 ============
SCENARIOS = {
    "bull": {
        "name": "乐观情景 (Bull Case)",
        "description": "强队优势放大、无伤病、冷门减少",
        "elo_strong_boost": +30,    # Top 8 球队 Elo 加成
        "elo_weak_penalty": 0,      # 弱队 Elo 不变
        "injury_rate": 0.0,         # 0% 伤病率
        "upset_factor": 0.75,       # 冷门概率 ×0.75
    },
    "base": {
        "name": "基准情景 (Base Case)",
        "description": "历史平均状态，模型默认",
        "elo_strong_boost": 0,
        "elo_weak_penalty": 0,
        "injury_rate": 0.05,
        "upset_factor": 1.0,
    },
    "bear": {
        "name": "悲观情景 (Bear Case)",
        "description": "强队折扣、伤病潮、冷门增多",
        "elo_strong_boost": -30,    # Top 8 Elo 折扣
        "elo_weak_penalty": +15,    # 弱队 Elo 提升
        "injury_rate": 0.15,
        "upset_factor": 1.5,        # 冷门概率 ×1.5
    },
}

# Top 8 球队（被 boost/penalty 的强队）
TOP_TEAMS = ["Spain", "France", "Argentina", "England", "Brazil", "Portugal", "Netherlands", "Germany"]


def apply_scenario(teams_data: dict, scenario_key: str) -> dict:
    """
    对 teams 数据应用情景参数（不修改原数据）
    
    返回：调整后的 teams_data 副本
    """
    cfg = SCENARIOS[scenario_key]
    adjusted = {}
    for name, data in teams_data.items():
        new_data = dict(data)  # 浅拷贝
        if name in TOP_TEAMS:
            new_data["elo"] = data["elo"] + cfg["elo_strong_boost"]
        else:
            new_data["elo"] = data["elo"] + cfg["elo_weak_penalty"]
        adjusted[name] = new_data
    return adjusted


def inject_injury(teams_data: dict, injury_rate: float, rng: np.random.Generator) -> dict:
    """
    随机注入伤病：以 injury_rate 概率让某球队损失核心球员
    Elo 临时下调 30 分（约相当于损失一个核心球员）
    """
    if injury_rate <= 0:
        return teams_data
    
    adjusted = {}
    for name, data in teams_data.items():
        new_data = dict(data)
        # 仅 Top 16 球队会受核心球员伤病影响
        if rng.random() < injury_rate and name in TOP_TEAMS:
            new_data["elo"] = data["elo"] - 30
        adjusted[name] = new_data
    return adjusted


def run_scenario_simulation(scenario_key: str, n_simulations: int, seed: int = 42,
                              verbose: bool = True) -> dict:
    """
    运行单一情景的 N 次蒙特卡洛模拟
    """
    teams_raw = load_teams()
    qualified = {n: d for n, d in teams_raw.items() if d.get("group") != "_"}
    groups = load_groups()
    bracket = load_bracket()
    
    # 应用情景的全局调整
    teams_scenario = apply_scenario(qualified, scenario_key)
    cfg = SCENARIOS[scenario_key]
    
    rng = np.random.default_rng(seed)
    
    stage_counts = {t: {"GROUP": 0, "R32_WIN": 0, "R16_WIN": 0, "QF_WIN": 0,
                        "SF_WIN": 0, "FINAL_LOSE": 0, "CHAMPION": 0}
                    for t in qualified}
    
    if verbose:
        print(f"\n🎬 {cfg['name']}")
        print(f"   {cfg['description']}")
        print(f"   参数: Elo强 {cfg['elo_strong_boost']:+d} / Elo弱 {cfg['elo_weak_penalty']:+d} / 伤病率 {cfg['injury_rate']*100:.0f}%")
    
    for sim_idx in range(n_simulations):
        if verbose and (sim_idx + 1) % max(1, n_simulations // 5) == 0:
            print(f"   Progress: {sim_idx + 1}/{n_simulations} ({(sim_idx+1)/n_simulations*100:.0f}%)")
        
        # 每场模拟独立注入伤病
        teams_this_sim = inject_injury(teams_scenario, cfg["injury_rate"], rng)
        result = simulate_tournament(teams_this_sim, groups, rng, bracket=bracket)
        for team, stage in result.items():
            if team in stage_counts:
                stage_counts[team][stage] += 1
    
    # 计算概率
    probs = {}
    for team, counts in stage_counts.items():
        total = sum(counts.values())
        if total == 0:
            continue
        deeper = lambda *stages: sum(counts[s] for s in stages)
        probs[team] = {
            "elo": qualified[team]["elo"],
            "market_implied": qualified[team]["market_implied"],
            "champion": round(counts["CHAMPION"] / total, 4),
            "final": round(deeper("FINAL_LOSE", "CHAMPION") / total, 4),
            "semifinal": round(deeper("QF_WIN", "SF_WIN", "FINAL_LOSE", "CHAMPION") / total, 4),
            "quarterfinal": round(deeper("R16_WIN", "QF_WIN", "SF_WIN", "FINAL_LOSE", "CHAMPION") / total, 4),
            "round_of_16": round(deeper("R32_WIN", "R16_WIN", "QF_WIN", "SF_WIN", "FINAL_LOSE", "CHAMPION") / total, 4),
            "n_simulations": total,
        }
    return probs


def run_three_scenarios(n_simulations: int = 100000, seed: int = 42) -> dict:
    """
    运行三情景并行模拟，返回 {scenario: probs}
    """
    print("=" * 80)
    print(f"🎬 WorldCup Engine — 三情景平行模拟（每情景 {n_simulations:,} 次）")
    print("=" * 80)
    
    results = {}
    t0 = time.time()
    
    for scenario_key in ["bull", "base", "bear"]:
        t_scenario_start = time.time()
        results[scenario_key] = run_scenario_simulation(
            scenario_key, n_simulations, seed=seed + hash(scenario_key) % 1000
        )
        t_scenario_end = time.time()
        print(f"   完成耗时: {t_scenario_end - t_scenario_start:.1f} 秒")
    
    t1 = time.time()
    print(f"\n⏱️  三情景总耗时: {t1 - t0:.1f} 秒")
    
    return results


def print_three_scenario_report(results: dict, top_n: int = 12):
    """打印三情景对比报告"""
    base_probs = results["base"]
    sorted_teams = sorted(base_probs.items(), key=lambda x: -x[1]["champion"])
    
    print("\n" + "=" * 105)
    print(f"📊 三情景冠军概率对比（Top {top_n}）")
    print("=" * 105)
    print(f"{'Rank':<5} {'Team':<16} {'Bull%':<8} {'Base%':<8} {'Bear%':<8} {'区间宽度':<10} {'市场%':<8} {'Bull偏差':<10} {'Bear偏差':<10}")
    print("-" * 105)
    
    for rank, (team, base_data) in enumerate(sorted_teams[:top_n], 1):
        bull_p = results["bull"][team]["champion"] * 100
        base_p = base_data["champion"] * 100
        bear_p = results["bear"][team]["champion"] * 100
        market = base_data["market_implied"] * 100
        width = bull_p - bear_p
        bull_bias = bull_p - market
        bear_bias = bear_p - market
        
        print(f"{rank:<5} {team:<16} {bull_p:>5.2f}%   {base_p:>5.2f}%   {bear_p:>5.2f}%   "
              f"{width:>5.1f}pp     {market:>5.1f}%   {bull_bias:>+6.1f}pp     {bear_bias:>+6.1f}pp")
    
    print("=" * 105)


def print_scenario_insights(results: dict):
    """识别情景敏感的球队"""
    print("\n📈 情景敏感性分析（按区间宽度排序）：\n")
    
    sensitivities = []
    for team in results["base"]:
        bull = results["bull"][team]["champion"] * 100
        bear = results["bear"][team]["champion"] * 100
        base = results["base"][team]["champion"] * 100
        width = bull - bear
        sensitivities.append((team, bull, base, bear, width))
    
    sensitivities.sort(key=lambda x: -x[4])
    
    print("  📌 最受情景影响（区间最宽，强队居多）：")
    for team, bull, base, bear, width in sensitivities[:5]:
        print(f"    🎯 {team:<14} Bull {bull:>5.1f}% / Base {base:>5.1f}% / Bear {bear:>5.1f}% ({width:>4.1f}pp)")
    
    print("\n  📌 最稳定（区间最窄，弱队/中等队）：")
    for team, bull, base, bear, width in [s for s in sensitivities if s[2] > 1.0][-5:]:
        print(f"    🔒 {team:<14} Bull {bull:>5.1f}% / Base {base:>5.1f}% / Bear {bear:>5.1f}% ({width:>4.1f}pp)")


def expected_value(results: dict, weights: dict = None) -> dict:
    """
    计算三情景加权期望值
    默认权重：Bull 25%, Base 50%, Bear 25%
    """
    if weights is None:
        weights = {"bull": 0.25, "base": 0.50, "bear": 0.25}
    
    expected = {}
    for team in results["base"]:
        ev = sum(results[s][team]["champion"] * weights[s] for s in weights)
        expected[team] = ev
    
    return expected


def main(n_sim: int = 100000):
    results = run_three_scenarios(n_simulations=n_sim)
    
    print_three_scenario_report(results, top_n=12)
    print_scenario_insights(results)
    
    # 期望值
    ev = expected_value(results)
    print("\n💎 三情景加权期望值（25% Bull / 50% Base / 25% Bear）：\n")
    sorted_ev = sorted(ev.items(), key=lambda x: -x[1])
    print(f"  {'Rank':<5} {'Team':<16} {'EV%':<8} {'Market%':<8} {'EV偏差':<8}")
    print("  " + "-" * 55)
    for rank, (team, p) in enumerate(sorted_ev[:12], 1):
        market = results["base"][team]["market_implied"] * 100
        bias = p * 100 - market
        print(f"  {rank:<5} {team:<16} {p*100:>5.2f}%   {market:>5.1f}%   {bias:>+6.1f}pp")
    
    # 保存
    output = {
        "scenarios": {k: {"config": SCENARIOS[k], "probs": v} for k, v in results.items()},
        "expected_value": {t: round(p, 4) for t, p in ev.items()},
        "n_simulations_per_scenario": n_sim,
    }
    save_output("three_scenarios.json", output)
    print(f"\n✅ 已保存到: data/outputs/three_scenarios.json")
    
    return results


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
    main(n)
