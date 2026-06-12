"""
爆冷 LLM 归因跑批
==================
扫描 match_bias.json，筛出"候选爆冷场"（规则粗筛），对每场跑 LLM 归因，
输出 5 标签结论存到 upset_llm_analysis.json。

候选规则（B 阈值，比"真爆冷信号"宽松，捕获更多潜在场次给 LLM 裁决）：
  - 弱队模型胜率 ≥ 20%（模型至少有一定看好）
  - 弱队市场胜率 ≥ 5%（市场赔率有效）
  - 模型 - 市场 ≥ 5pp（差距实质）

用法:
  python3 code/models/run_upset_analysis.py        # 跑全候选池
  python3 code/models/run_upset_analysis.py --max 3  # 限制 N 场（测试）
"""
import sys
import json
import time
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.upset_llm_analyzer import analyze_upset
from utils.io import save_output

logging.basicConfig(level=logging.WARNING)

ROOT = Path(__file__).parent.parent.parent


# 候选粗筛阈值（比 match_bias 的"真爆冷信号"更宽，让 LLM 来终审）
MIN_MODEL_PCT = 20       # 模型给关注方 ≥20%
MIN_MARKET_PCT = 5       # 市场赔率有效
MIN_DELTA_PP = 5          # 模型 - 市场差距 ≥5pp


def find_candidates():
    """从 match_bias.json 扫候选场。
    
    候选包含两类：
      1. A 定义经典爆冷：模型预测的负方胜率显著大于市场给该队胜率
      2. 模型 vs 市场胜方分歧：模型推的胜方不是市场推的胜方
    """
    mb_path = ROOT / "data" / "outputs" / "match_bias.json"
    if not mb_path.exists():
        print(f"❌ {mb_path} 不存在，请先跑 match_bias_detector.py")
        sys.exit(1)
    
    mb = json.load(open(mb_path))
    candidates = []
    
    for m in mb["matches"]:
        model = m["model"]
        kalshi = m["kalshi"]
        team_a, team_b = m["team_a"], m["team_b"]
        
        # 模型负方（A 定义关注方）
        if model["p_win_a"] < model["p_win_b"]:
            model_loser = team_a
            loser_model_pct = model["p_win_a"] * 100
            loser_market_pct = kalshi["p_win_a"] * 100
            loser_delta = m["delta_pp"]["p_win_a"]
        else:
            model_loser = team_b
            loser_model_pct = model["p_win_b"] * 100
            loser_market_pct = kalshi["p_win_b"] * 100
            loser_delta = m["delta_pp"]["p_win_b"]
        
        # 模型胜方 vs 市场胜方
        model_winner = team_a if model["p_win_a"] > model["p_win_b"] else team_b
        market_winner = team_a if kalshi["p_win_a"] > kalshi["p_win_b"] else team_b
        winners_diverge = (model_winner != market_winner)
        
        # 触发类型 1：A 定义经典爆冷
        type1 = (
            loser_model_pct >= MIN_MODEL_PCT
            and loser_market_pct >= MIN_MARKET_PCT
            and loser_delta >= MIN_DELTA_PP
        )
        # 触发类型 2：胜方分歧（模型完全反市场）
        # 这种场要给 LLM，让它判是"模型对市场错"还是反过来
        type2 = winners_diverge
        
        if not (type1 or type2):
            continue
        
        # 关注方：A 定义用模型负方；胜方分歧用模型胜方（因为它是"模型反市场"的核心）
        if type1 and not type2:
            focus_team = model_loser
            focus_model_pct = loser_model_pct
            focus_market_pct = loser_market_pct
            focus_delta = loser_delta
            category = "model_loser_upset"
        else:
            # type2 或 type1+type2 同时触发：关注模型胜方
            focus_team = model_winner
            focus_model_pct = model["p_win_a"] * 100 if focus_team == team_a else model["p_win_b"] * 100
            focus_market_pct = kalshi["p_win_a"] * 100 if focus_team == team_a else kalshi["p_win_b"] * 100
            focus_delta = focus_model_pct - focus_market_pct
            category = "model_market_diverge" if type2 and not type1 else "both"
        
        candidates.append({
            "match": m,
            "weak_team": focus_team,           # 复用旧字段名（LLM prompt 还在用）
            "weak_model_pct": round(focus_model_pct, 1),
            "weak_market_pct": round(focus_market_pct, 1),
            "weak_delta_pp": round(focus_delta, 2),
            "category": category,
            "winners_diverge": winners_diverge,
        })
    
    candidates.sort(key=lambda c: -c["weak_delta_pp"])
    return candidates


def main(max_n=None):
    teams = json.load(open(ROOT / "data" / "raw" / "teams.json"))["teams"]
    candidates = find_candidates()
    
    if max_n:
        candidates = candidates[:max_n]
    
    print(f"=== 爆冷 LLM 归因跑批 ===")
    print(f"候选场次：{len(candidates)}")
    print()
    
    for i, c in enumerate(candidates, 1):
        m = c["match"]
        print(f"  [{i}/{len(candidates)}] {m['team_a']} vs {m['team_b']}  "
              f"弱队 {c['weak_team']} 模型 {c['weak_model_pct']}% vs 市场 {c['weak_market_pct']}% "
              f"(差 {c['weak_delta_pp']:+.1f}pp)")
    print()
    
    results = {}
    t0 = time.time()
    
    for i, c in enumerate(candidates, 1):
        m = c["match"]
        print(f"\n[{i}/{len(candidates)}] 跑 {m['team_a']} vs {m['team_b']}...")
        match_t = time.time()
        
        result = analyze_upset(
            m["team_a"], m["team_b"], m["date"],
            m["model"], m["kalshi"], m["delta_pp"],
            c["weak_team"],
            teams[m["team_a"]], teams[m["team_b"]],
        )
        elapsed = time.time() - match_t
        
        key = f"{m['team_a']}|{m['team_b']}|{m['date']}"
        results[key] = {
            "team_a": m["team_a"],
            "team_b": m["team_b"],
            "date": m["date"],
            "weak_team": c["weak_team"],
            "weak_model_pct": c["weak_model_pct"],
            "weak_market_pct": c["weak_market_pct"],
            "weak_delta_pp": c["weak_delta_pp"],
            "model": m["model"],
            "kalshi": m["kalshi"],
            "delta_pp": m["delta_pp"],
            "elapsed_sec": round(elapsed, 1),
            **result,
        }
        
        if "error" in result:
            print(f"  ⚠ 失败：{result['error']} ({elapsed:.1f}s)")
        else:
            print(f"  ✓ {result['label_text']} ({result['confidence']}置信) {elapsed:.1f}s")
            print(f"    {result['rationale']}")
    
    total_elapsed = time.time() - t0
    
    output = {
        "n_candidates": len(candidates),
        "total_elapsed_sec": round(total_elapsed, 1),
        "thresholds": {
            "min_model_pct": MIN_MODEL_PCT,
            "min_market_pct": MIN_MARKET_PCT,
            "min_delta_pp": MIN_DELTA_PP,
        },
        "results": results,
    }
    save_output("upset_llm_analysis.json", output)
    
    print(f"\n=== 完成 ===")
    print(f"总耗时: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
    
    # 标签统计
    from collections import Counter
    labels = Counter(r.get("label", "error") for r in results.values())
    print(f"标签分布: {dict(labels)}")
    
    print(f"\n=== 真爆冷信号（true_upset 标签）===")
    for k, r in results.items():
        if r.get("label") == "true_upset":
            print(f"  🚀 {r['team_a']} vs {r['team_b']} ({r['date']})")
            print(f"     弱队 {r['weak_team']} {r['weak_model_pct']}%vs{r['weak_market_pct']}% (+{r['weak_delta_pp']}pp)")
            print(f"     置信: {r['confidence']}  → {r['rationale']}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=None)
    args = ap.parse_args()
    main(max_n=args.max)
