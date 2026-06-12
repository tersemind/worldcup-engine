"""
跑批：扫描方案 B 的 most_likely_bracket.json，对所有「胶着场」（胜率 35-65%）
跑 5-Agent Byzantine 微调，输出 critical_node_adjustments.json

key 格式：("round", match_id) → adjustment dict
例如：("r16", 5) → {"applied": True, "final_adjustment_pp": +1.5, ...}

设计：
  - 跑前先用 _quick_match_preview 拿到 MC 修正后的 p_win_a（不是原始 MC 频次，
    而是 Elo + Poisson 集成的当前 p_win_a 字段，因为这才是前端实际显示的值）
  - 实际上"胶着"的判定标准也用这个 p_win_a，区间 [0.35, 0.65]
  - 跑完每场记录耗时，可控制总时长

用法：
  python3 code/models/run_critical_nodes.py [--threshold 0.35-0.65] [--max N]

  默认跑所有胶着场（约 15-20 场，~5min）
"""
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.critical_node_swarm import run_critical_node_swarm
from utils.io import load_teams, save_output

# 抑制 LLM 警告
logging.basicConfig(level=logging.WARNING)


DATA_OUTPUTS = Path(__file__).parent.parent.parent / "data" / "outputs"


def get_quick_pred(team_a: str, team_b: str, teams: Dict) -> Dict:
    """复用 tournament_api._quick_match_preview 的核心逻辑（不依赖 web 模块）"""
    from models.elo_engine import match_probabilities
    from models.poisson_model import predict_match_ensemble
    
    ta = teams.get(team_a)
    tb = teams.get(team_b)
    if not ta or not tb:
        return None
    
    elo_p = match_probabilities(ta["elo"], tb["elo"])
    poi = predict_match_ensemble(ta, tb)
    p_a = (elo_p["p_win_a"] + poi["outcome"]["p_win_a"]) / 2
    p_b = (elo_p["p_win_b"] + poi["outcome"]["p_win_b"]) / 2
    p_d = (elo_p["p_draw"] + poi["outcome"]["p_draw"]) / 2
    # 把平局摊到两边的"胜负二元胜率"上（淘汰赛会决出胜方）
    if p_a + p_b > 0:
        p_a_decisive = p_a + p_d * (p_a / (p_a + p_b))
    else:
        p_a_decisive = 0.5
    return {"p_win_a_decisive": round(p_a_decisive, 4)}


def find_critical_matches(ml: Dict, teams: Dict,
                           threshold_low: float = 0.35,
                           threshold_high: float = 0.65) -> List[Tuple]:
    """
    扫描 most_likely_bracket，返回 [(round, match_id, team_a, team_b, p_a_decisive), ...]
    其中 p_a_decisive 在 [threshold_low, threshold_high] 范围内的为胶着场。
    """
    candidates = []
    rounds = ml.get("rounds", {})
    
    for round_name in ("r32", "r16", "qf", "sf"):
        for entry in rounds.get(round_name, []):
            ta, tb = entry["team_a"], entry["team_b"]
            mid = entry["match_id"]
            pred = get_quick_pred(ta, tb, teams)
            if not pred:
                continue
            p = pred["p_win_a_decisive"]
            if threshold_low <= p <= threshold_high:
                candidates.append((round_name, mid, ta, tb, p))
    
    # 决赛
    final = rounds.get("final")
    if final:
        ta, tb = final["team_a"], final["team_b"]
        pred = get_quick_pred(ta, tb, teams)
        if pred and threshold_low <= pred["p_win_a_decisive"] <= threshold_high:
            candidates.append(("final", 1, ta, tb, pred["p_win_a_decisive"]))
    
    return candidates


def main(threshold_low=0.35, threshold_high=0.65, max_matches=None):
    teams = load_teams()
    
    ml_path = DATA_OUTPUTS / "most_likely_bracket.json"
    if not ml_path.exists():
        print(f"❌ {ml_path} 不存在，请先跑 most_likely_bracket.py")
        return
    
    ml = json.load(open(ml_path))
    candidates = find_critical_matches(ml, teams, threshold_low, threshold_high)
    
    if max_matches:
        candidates = candidates[:max_matches]
    
    print(f"=== 关键节点 5-Agent 微调跑批 ===")
    print(f"胶着判定阈值：胜率 ∈ [{threshold_low}, {threshold_high}]")
    print(f"待处理胶着场：{len(candidates)} 场")
    print()
    for r, mid, ta, tb, p in candidates:
        print(f"  [{r}] M{mid}: {ta} {p*100:.1f}% vs {tb} {(1-p)*100:.1f}%")
    print()
    
    if not candidates:
        print("✅ 无胶着场，无需修正")
        return
    
    results = {}
    t0 = time.time()
    
    for i, (round_name, mid, ta, tb, p_a) in enumerate(candidates, 1):
        print(f"\n[{i}/{len(candidates)}] {round_name} M{mid}: {ta} vs {tb} (MC {p_a*100:.1f}%)")
        match_t = time.time()
        
        result = run_critical_node_swarm(
            ta, tb, p_a,
            teams[ta], teams[tb],
        )
        elapsed = time.time() - match_t
        
        # key 序列化为 "round:mid"
        key = f"{round_name}:{mid}"
        results[key] = {
            "round": round_name,
            "match_id": mid,
            "team_a": ta,
            "team_b": tb,
            "mc_p_win_a_decisive": p_a,
            "elapsed_sec": round(elapsed, 1),
            **result,
        }
        
        if result.get("applied"):
            print(f"  ✓ {result['summary']} ({elapsed:.1f}s)")
        else:
            print(f"  ⚠ 未修正: {result.get('reason') or result.get('error')} ({elapsed:.1f}s)")
    
    total_elapsed = time.time() - t0
    
    output = {
        "n_matches_scanned": len(candidates),
        "threshold_low": threshold_low,
        "threshold_high": threshold_high,
        "total_elapsed_sec": round(total_elapsed, 1),
        "adjustments": results,
    }
    save_output("critical_node_adjustments.json", output)
    
    print(f"\n=== 完成 ===")
    print(f"总耗时: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
    n_applied = sum(1 for r in results.values() if r.get("applied"))
    n_disagree = sum(1 for r in results.values() if not r.get("applied") and r.get("reason"))
    n_error = sum(1 for r in results.values() if r.get("error"))
    print(f"有效修正: {n_applied}/{len(results)}")
    print(f"Agent 分歧: {n_disagree}/{len(results)}")
    print(f"LLM 失败: {n_error}/{len(results)}")
    print(f"\n输出: data/outputs/critical_node_adjustments.json")
    
    if n_applied > 0:
        print(f"\n=== 修正最大的 Top 5 ===")
        applied_results = [(k, v) for k, v in results.items() if v.get("applied")]
        applied_results.sort(key=lambda x: -abs(x[1]["final_adjustment_pp"]))
        for k, v in applied_results[:5]:
            print(f"  [{v['round']}] {v['team_a']} vs {v['team_b']}: "
                  f"{v['final_adjustment_pp']:+.2f}pp → 推 {v['consensus_winner']}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--low", type=float, default=0.35)
    ap.add_argument("--high", type=float, default=0.65)
    ap.add_argument("--max", type=int, default=None, help="限制跑批数量（调试用）")
    args = ap.parse_args()
    main(threshold_low=args.low, threshold_high=args.high, max_matches=args.max)
