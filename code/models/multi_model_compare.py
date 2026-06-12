#!/usr/bin/env python3
"""
多模型对比器（#19 P4）

对比来源：
- 本引擎：synthesizer_report.json
- Reference：人工硬编码（来自 参考报告表 8.1）
- Sophia：人工硬编码（来自 sophia-top8.md）
- Opta：人工硬编码（来自 worldcuppass.com 抓取）
- 市场（Polymarket+Kalshi）：teams.json 中的 market_implied
- Goldman Sachs：人工硬编码

输出：
- 多源概率对比表
- 模型分歧度（Disagreement Index）
- 共识偏差识别（参考 Reference 德国分析框架）
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.io import DATA_OUTPUTS, load_teams


# 来自之前对话中各模型的预测（数据截至 2026-06-10）
EXTERNAL_PREDICTIONS = {
    "Reference": {
        # 参考报告表 8.1（基准概率）
        "Spain": 16.5, "France": 15.0, "Argentina": 12.0, "England": 11.0,
        "Germany": 11.0, "Brazil": 9.0, "Portugal": 7.0, "Netherlands": 4.0,
        "Colombia": 3.5, "Morocco": 1.5, "Belgium": 1.0, "Mexico": 1.0,
        "Japan": 1.2, "USA": 0.9, "Uruguay": 0.8,
    },
    "Sophia": {
        # Sophia top8 报告
        "Spain": 18.0, "France": 16.0, "England": 13.0, "Argentina": 9.0,
        "Portugal": 8.0, "Brazil": 7.0, "Germany": 6.0, "Netherlands": 4.0,
    },
    "Opta": {
        # worldcuppass.com 抓取的 25,000 次模拟
        "Spain": 16.1, "France": 13.0, "England": 11.2, "Argentina": 10.4,
        "Portugal": 7.0, "Brazil": 6.6, "Germany": 5.1, "Norway": 2.3,
    },
    "Goldman": {
        # Bloomberg Goldman Sachs 模型（部分球队）
        "Spain": 26.0, "France": 19.0, "Argentina": 14.0, "England": 5.0,
        "Brazil": 8.0,
    },
}


def load_engine_predictions() -> dict:
    """加载本引擎预测"""
    path = DATA_OUTPUTS / "synthesizer_report.json"
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    return {team: d["final_probability"] for team, d in data.items()}


def load_market_predictions() -> dict:
    """加载市场赔率"""
    teams = load_teams()
    return {team: d.get("market_implied", 0) * 100 
            for team, d in teams.items() if d.get("group") != "_"}


def compute_disagreement(predictions: dict) -> dict:
    """
    计算各队的"模型分歧度"（标准差）
    """
    teams = set()
    for source_preds in predictions.values():
        teams.update(source_preds.keys())
    
    disagreement = {}
    for team in teams:
        values = []
        for source, preds in predictions.items():
            if team in preds:
                values.append(preds[team])
        if len(values) >= 2:
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            std = variance ** 0.5
            disagreement[team] = {
                "mean": round(mean, 2),
                "std": round(std, 2),
                "min": round(min(values), 2),
                "max": round(max(values), 2),
                "range": round(max(values) - min(values), 2),
                "n_sources": len(values),
            }
    return disagreement


def find_consensus_bias(predictions: dict, market: dict) -> dict:
    """
    识别"共识偏差"：模型组共识 vs 市场的偏差
    （类似 Reference 德国分析法）
    """
    # 计算各模型平均（除 Goldman 太激进单独看）
    main_sources = ["engine", "Reference", "Sophia", "Opta"]
    
    bias = {}
    teams = set()
    for s in main_sources:
        if s in predictions:
            teams.update(predictions[s].keys())
    
    for team in teams:
        values = [predictions[s][team] for s in main_sources 
                  if s in predictions and team in predictions[s]]
        if not values:
            continue
        consensus = sum(values) / len(values)
        market_p = market.get(team, 0)
        if market_p == 0:
            continue
        bias[team] = {
            "consensus": round(consensus, 2),
            "market": round(market_p, 2),
            "bias_pp": round(consensus - market_p, 2),
            "n_models": len(values),
        }
    return bias


def print_comparison_table(predictions: dict, top_n: int = 12):
    """打印多源对比表"""
    # 用本引擎排名作为顺序
    engine = predictions.get("engine", {})
    sorted_teams = sorted(engine.items(), key=lambda x: -x[1])[:top_n]
    
    sources = ["engine", "Reference", "Sophia", "Opta", "Goldman", "Market"]
    
    print("=" * 110)
    print("🌐 多模型预测对比（Top 12 球队）")
    print("=" * 110)
    
    # 表头
    header = f"  {'Rank':<4} {'Team':<14}"
    for s in sources:
        header += f" {s:<10}"
    header += f" {'StdDev':<8}"
    print(header)
    print("  " + "-" * 100)
    
    for rank, (team, _) in enumerate(sorted_teams, 1):
        row = f"  {rank:<4} {team:<14}"
        values = []
        for s in sources:
            if s in predictions and team in predictions[s]:
                v = predictions[s][team]
                row += f" {v:>5.1f}%    "
                values.append(v)
            else:
                row += f"   {'—':<7} "
        # std
        if len(values) >= 2:
            mean = sum(values) / len(values)
            std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
            row += f" {std:>5.2f}"
        print(row)
    
    print("=" * 110)


def print_disagreement_top(disagreement: dict, top_n: int = 10):
    """打印分歧最大的球队"""
    sorted_disagree = sorted(disagreement.items(), key=lambda x: -x[1]["std"])[:top_n]
    
    print(f"\n📊 模型分歧度 Top {top_n}（标准差越大，分歧越大）：\n")
    print(f"  {'Rank':<5} {'Team':<14} {'Std':<8} {'Range':<10} {'Min':<8} {'Mean':<8} {'Max':<8} {'N':<4}")
    print("  " + "-" * 70)
    for rank, (team, d) in enumerate(sorted_disagree, 1):
        print(f"  {rank:<5} {team:<14} {d['std']:>5.2f}    {d['range']:>5.2f}    "
              f"{d['min']:>5.1f}%   {d['mean']:>5.1f}%   {d['max']:>5.1f}%   {d['n_sources']:<4}")


def print_consensus_bias(bias: dict, top_n: int = 10):
    """打印共识偏差 Top（最被低估/高估）"""
    sorted_bias = sorted(bias.items(), key=lambda x: -x[1]["bias_pp"])
    
    print(f"\n🟢 模型组共识 > 市场（被低估）：")
    for team, b in [t for t in sorted_bias if t[1]["bias_pp"] > 0.5][:5]:
        print(f"  {team:<14} 共识 {b['consensus']:>5.1f}% vs 市场 {b['market']:>5.1f}%  "
              f"(+{b['bias_pp']:.2f}pp, {b['n_models']} 个模型)")
    
    print(f"\n🔴 模型组共识 < 市场（被高估）：")
    for team, b in [t for t in sorted_bias if t[1]["bias_pp"] < -0.5][-5:]:
        print(f"  {team:<14} 共识 {b['consensus']:>5.1f}% vs 市场 {b['market']:>5.1f}%  "
              f"({b['bias_pp']:.2f}pp, {b['n_models']} 个模型)")


def main():
    print("📥 加载所有预测来源...")
    
    predictions = {
        "engine": load_engine_predictions(),
        **EXTERNAL_PREDICTIONS,
        "Market": load_market_predictions(),
    }
    
    # 输出对比
    print_comparison_table(predictions)
    
    # 分歧度
    disagreement = compute_disagreement(predictions)
    print_disagreement_top(disagreement)
    
    # 共识偏差
    market = predictions["Market"]
    bias = find_consensus_bias(predictions, market)
    print_consensus_bias(bias)
    
    # 总结
    print(f"\n📈 总结：")
    n_teams_compared = len(disagreement)
    avg_std = sum(d["std"] for d in disagreement.values()) / max(1, n_teams_compared)
    print(f"  对比球队数: {n_teams_compared}")
    print(f"  平均模型分歧度: {avg_std:.2f}pp")
    print(f"  最大分歧球队: {sorted(disagreement.items(), key=lambda x: -x[1]['std'])[0][0]}")
    
    # 保存
    output = {
        "predictions": predictions,
        "disagreement": disagreement,
        "consensus_bias": bias,
        "summary": {
            "n_teams": n_teams_compared,
            "avg_std": round(avg_std, 2),
            "n_sources": len(predictions),
        }
    }
    
    out_path = DATA_OUTPUTS / "multi_model_comparison.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 已保存: {out_path.name}")


if __name__ == "__main__":
    main()
