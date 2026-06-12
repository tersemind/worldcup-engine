"""
综合预测器：在蒙特卡洛基准概率上叠加情境/伤病/心理调整
输出最终冠军概率 + 95% CI + 偏差识别
"""
import json
import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.io import load_teams, save_output, DATA_OUTPUTS


# 调整因子配置（可外部覆盖）
DEFAULT_ADJUSTMENTS = {
    # 球队 -> {health, context, psych} 调整百分点
    # ⚠ psych 必须与 PsychologyAgent.fallback_analyze 的 psych_factors 同步
    # ⚠ 详见 tests/test_engine.py::TestPsychologySync
    "Spain":      {"health": -1.5, "context": -1.3, "psych":  +0.3},   # 同步 Agent
    "France":     {"health": +0.5, "context": -0.6, "psych":  +1.0},
    "Germany":    {"health": -0.5, "context": -0.7, "psych":  +1.5},   # 注：synth 用 +1.5（不需要双重激励，Agent 端因 swarm 缩放用 +2.5）
    "Mexico":     {"health":  0.0, "context": +0.5, "psych":  +0.3},
    "USA":        {"health":  0.0, "context": +1.0, "psych":  +0.2},

    "Argentina":  {"health": -0.5, "context": +0.5, "psych":  -2.5},
    "England":    {"health": -1.0, "context": -1.0, "psych":  -0.2},
    "Brazil":     {"health": -2.5, "context": +0.3, "psych":  -0.2},
    "Portugal":   {"health": +0.5, "context": -0.3, "psych":  -0.2},
    "Netherlands":{"health": -2.5, "context": -0.3, "psych":  -0.5},
    "Belgium":    {"health":  0.0, "context": -0.3, "psych":  -0.3},

    # Elo 滞后队（2024 爆发后 2025-2026 回落）
    "Colombia":   {"health":  0.0, "context": +0.3, "psych":  -1.5},
    "Ecuador":    {"health":  0.0, "context": +0.2, "psych":  -1.0},
    "Norway":     {"health":  0.0, "context":  0.0, "psych":  -0.5},

    # 中性
    "Croatia":    {"health":  0.0, "context": -0.2, "psych":   0.0},
    "Morocco":    {"health": -0.3, "context": +0.0, "psych":   0.0},
    "Switzerland":{"health": -0.2, "context": -0.3, "psych":   0.0},
}


def compute_confidence_interval(p_champion: float, n_simulations: int) -> tuple:
    """
    根据蒙特卡洛样本量计算 95% 置信区间（Wald 近似）
    """
    if n_simulations < 30 or p_champion <= 0:
        return (max(0, p_champion - 0.05), p_champion + 0.05)
    
    # 标准误差
    se = math.sqrt(p_champion * (1 - p_champion) / n_simulations)
    z = 1.96  # 95% CI
    
    lower = max(0, p_champion - z * se - 0.02)  # 留 2pp 模型不确定性 buffer
    upper = min(1, p_champion + z * se + 0.02)
    
    return (lower, upper)


def confidence_level(p: float) -> str:
    """置信度档次"""
    if p > 0.60:
        return "高置信度"
    elif p >= 0.40:
        return "中置信度"
    else:
        return "低置信度"


def synthesize(mc_probs: dict, adjustments: dict = None, use_injuries: bool = True) -> dict:
    """
    将蒙特卡洛基准概率与情境调整融合
    
    输入：
        mc_probs: 蒙特卡洛输出 {team: {champion, market_implied, ...}}
        adjustments: {team: {health, context, psych}} 调整百分点
        use_injuries: 是否自动读取 injuries.json 覆盖 health 字段
    """
    if adjustments is None:
        # 深拷贝避免修改全局
        adjustments = {k: dict(v) for k, v in DEFAULT_ADJUSTMENTS.items()}
    
    # 自动加载伤病库，覆盖默认 health 调整
    if use_injuries:
        try:
            from data.injuries_fetcher import get_health_adjustments
            injury_adjs = get_health_adjustments()
            for team, total_pp in injury_adjs.items():
                if team not in adjustments:
                    adjustments[team] = {"health": 0, "context": 0, "psych": 0}
                # 覆盖 health（伤病库优先级最高）
                adjustments[team]["health"] = total_pp
        except Exception as e:
            pass  # 伤病库不可用时使用默认值
    
    # 始终读取最新 teams.json 中的市场隐含概率（覆盖 mc_probs 中的旧值）
    latest_teams = load_teams()

    # 叠加 per-match 信号（h2h / referee / weather / lineups）
    # 失败必须不影响主流程
    h2h_adjs: dict = {}
    referee_adjs: dict = {}
    weather_adjs: dict = {}
    lineup_adjs: dict = {}
    try:
        from data.match_context_adjustments import (
            get_h2h_adjustments, get_referee_adjustments,
            get_weather_adjustments, get_lineup_adjustments,
        )
        h2h_adjs = get_h2h_adjustments()
        referee_adjs = get_referee_adjustments(latest_teams)
        weather_adjs = get_weather_adjustments(latest_teams)
        lineup_adjs = get_lineup_adjustments()
    except Exception:
        pass  # 文件缺失或解析失败 → 不打扰主合成
    
    results = {}
    # 预计算 Transfermarkt 身价中位数（用于 squad_value 调整）
    sq_values = [d.get("squad_value_m_eur", 0) for d in latest_teams.values()
                  if d.get("group") != "_" and d.get("squad_value_m_eur", 0) > 0]
    median_sq = sorted(sq_values)[len(sq_values) // 2] if sq_values else 200

    for team, mc_data in mc_probs.items():
        base = mc_data["champion"]  # 蒙特卡洛基准
        adj = adjustments.get(team, {"health": 0, "context": 0, "psych": 0})

        # Transfermarkt 身价调整（P5: 5% 权重）
        # 非线性曲线，三段:
        #   ≤ median: 每 €200M 差 = -1pp (cap -2)
        #   median 到 +€600M: +1pp / 200M (线性)
        #   >+€600M (super-club): +0.8pp / 200M (额外加速)
        # 总 cap ±3.0pp，让 €1B+ 超级豪门差异化
        sq_value = latest_teams.get(team, {}).get("squad_value_m_eur", 0)
        if sq_value > 0:
            sq_diff = sq_value - median_sq
            if sq_diff <= 0:
                adj_squad = max(-2.0, (sq_diff / 200.0) * 1.0)
            elif sq_diff <= 600:
                adj_squad = (sq_diff / 200.0) * 1.0   # 0 → 3.0pp
            else:
                # 超过 €600M 差再加，但斜率温和（避免一刀切）
                adj_squad = 3.0 + ((sq_diff - 600) / 200.0) * 0.8
            adj_squad = max(-2.0, min(3.5, adj_squad))   # cap ±3.5
        else:
            adj_squad = 0.0

        # per-match 信号叠加：
        #   h2h 心理压制（cap ±1.0pp）
        #   referee 风格   （cap ±0.4pp）
        #   weather 极端   （cap ±0.4pp）
        #   lineups 缺主力 （cap ±0.5pp）
        adj_h2h = h2h_adjs.get(team, 0.0)
        adj_referee = referee_adjs.get(team, 0.0)
        adj_weather = weather_adjs.get(team, 0.0)
        adj_lineups = lineup_adjs.get(team, 0.0)

        # 调整：每个因子按百分点叠加
        adj_total = (adj["health"] + adj["context"] + adj["psych"] + adj_squad
                     + adj_h2h + adj_referee + adj_weather + adj_lineups) / 100.0
        adjusted = base + adj_total
        adjusted = max(0.001, min(0.50, adjusted))  # 边界
        
        n_sim = mc_data["n_simulations"]
        ci_low, ci_high = compute_confidence_interval(adjusted, n_sim)
        
        # 从最新 teams.json 取 market_implied（保证 refresh 后即时生效）
        if team in latest_teams:
            market = latest_teams[team].get("market_implied", mc_data["market_implied"])
        else:
            market = mc_data["market_implied"]
        bias = (adjusted - market) * 100  # pp
        
        results[team] = {
            "elo": mc_data["elo"],
            "squad_value_m_eur": sq_value,
            "mc_baseline": round(base * 100, 2),
            "adj_health": adj["health"],
            "adj_context": adj["context"],
            "adj_psych": adj["psych"],
            "adj_squad_value": round(adj_squad, 2),
            "adj_h2h": round(adj_h2h, 2),
            "adj_referee": round(adj_referee, 2),
            "adj_weather": round(adj_weather, 2),
            "adj_lineups": round(adj_lineups, 2),
            "final_probability": round(adjusted * 100, 2),
            "ci_lower": round(ci_low * 100, 2),
            "ci_upper": round(ci_high * 100, 2),
            "ci_width": round((ci_high - ci_low) * 100, 2),
            "market_implied": round(market * 100, 2),
            "bias_pp": round(bias, 2),
            "confidence_level": confidence_level(adjusted),
            "advance_to_final": round(mc_data["final"] * 100, 2),
            "advance_to_sf": round(mc_data["semifinal"] * 100, 2),
            "advance_to_qf": round(mc_data["quarterfinal"] * 100, 2),
            "advance_to_r16": round(mc_data["round_of_16"] * 100, 2),
        }
    
    return results


def print_report(results: dict, top_n: int = 12):
    """打印综合报告"""
    sorted_teams = sorted(results.items(), key=lambda x: -x[1]["final_probability"])
    
    print("\n" + "="*110)
    print("🏆 2026 World Cup Winner Probability — Real Engine v1.0")
    print("="*110)
    print(f"{'Rank':<5} {'Team':<16} {'Final%':<8} {'95% CI':<14} {'MC基准':<8} {'Health':<8} {'Context':<9} {'Psych':<8} {'Market':<8} {'Bias':<8}")
    print("-"*110)
    
    for rank, (team, r) in enumerate(sorted_teams[:top_n], 1):
        ci = f"[{r['ci_lower']:.1f}-{r['ci_upper']:.1f}]"
        bias_str = f"{r['bias_pp']:+.1f}pp"
        print(f"{rank:<5} {team:<16} {r['final_probability']:>5.2f}%   {ci:<14} {r['mc_baseline']:>5.1f}%   "
              f"{r['adj_health']:>+5.1f}   {r['adj_context']:>+5.1f}    {r['adj_psych']:>+5.1f}   "
              f"{r['market_implied']:>5.1f}%   {bias_str:<8}")
    
    print("="*110)
    
    # 偏差识别
    print("\n📊 市场偏差识别（按偏差幅度排序）：\n")
    sorted_by_bias = sorted(results.items(), key=lambda x: -abs(x[1]["bias_pp"]))
    print("  最被低估（正偏差，模型 > 市场）：")
    pos_bias = [(t, r) for t, r in sorted_by_bias if r["bias_pp"] > 0.5][:5]
    for team, r in pos_bias:
        print(f"    🟢 {team:<14} 模型 {r['final_probability']:>5.1f}% vs 市场 {r['market_implied']:>5.1f}% (+{r['bias_pp']:.1f}pp)")
    
    print("\n  最被高估（负偏差，模型 < 市场）：")
    neg_bias = [(t, r) for t, r in sorted_by_bias if r["bias_pp"] < -0.5][:5]
    for team, r in neg_bias:
        print(f"    🔴 {team:<14} 模型 {r['final_probability']:>5.1f}% vs 市场 {r['market_implied']:>5.1f}% ({r['bias_pp']:.1f}pp)")
    
    print(f"\n📌 所有概率 < 60% → 置信度档次：低置信度（48 队制首届赛事的固有不确定性）")


def main(mc_file: str = "mc_simulation_n100000.json"):
    """主入口"""
    mc_path = DATA_OUTPUTS / mc_file
    if not mc_path.exists():
        print(f"❌ 蒙特卡洛输出不存在：{mc_path}")
        print(f"   请先运行：python3 code/models/monte_carlo.py 100000")
        sys.exit(1)
    
    with open(mc_path, "r") as f:
        mc_probs = json.load(f)
    
    results = synthesize(mc_probs)
    print_report(results, top_n=12)
    
    save_output("synthesizer_report.json", results)
    print(f"\n✅ 已保存到 data/outputs/synthesizer_report.json")
    
    return results


if __name__ == "__main__":
    mc_file = sys.argv[1] if len(sys.argv) > 1 else "mc_simulation_n100000.json"
    main(mc_file)
