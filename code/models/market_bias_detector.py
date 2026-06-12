"""
市场偏差检测引擎
================
对齐 分级规范 + §3.6.10：把 Polymarket/Kalshi 市场作为"共识偏差研究变量"，
**不修正模型预测**，只识别偏差并分级触发处理。

输入：
  - teams.json[*].market_implied         （Polymarket+Kalshi 综合夺冠概率）
  - mc_simulation_n100000.json[*].champion （我们模型的夺冠概率）

输出：
  - data/outputs/market_bias_snapshots/snapshot_<timestamp>.json
  - data/outputs/market_bias_latest.json （最新一份的引用）

偏差级别（混合阈值——绝对差或相对差任一触发）：
  - L1 轻度：|Δpp| < 3pp  且  相对差 < 30%
  - L2 中度：3pp ≤ |Δpp| < 6pp  或  30% ≤ 相对差 < 60%
  - L3 高度：6pp ≤ |Δpp| < 10pp  或  60% ≤ 相对差 < 100%
  - L4 极端：|Δpp| ≥ 10pp  或  相对差 ≥ 100%

阈值参考 分级规范，但我们 48 队跨度大，用相对差兜底处理小概率队（如 Curacao）。
"""
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

ROOT = Path(__file__).parent.parent.parent
SNAPSHOT_DIR = ROOT / "data" / "outputs" / "market_bias_snapshots"
LATEST_PATH = ROOT / "data" / "outputs" / "market_bias_latest.json"


def classify_bias(model_pct: float, market_pct: float) -> Dict[str, Any]:
    """
    对单个对比项分级。
    返回 {delta_pp, abs_delta_pp, rel_delta, level, level_text, direction, action}
    """
    delta_pp = model_pct - market_pct
    abs_delta = abs(delta_pp)
    # 相对差：以两者较大值为分母，避免小概率队被绝对差掩盖
    denom = max(model_pct, market_pct, 0.01)  # 防 0
    rel_delta = abs_delta / denom
    
    # 分级：绝对差 OR 相对差任一触发
    # 重要：相对差需要叠加"最小绝对差地板"（0.3pp），避免 0.0% vs 0.1% 这种噪声被算成 L4
    MIN_ABS_FOR_REL = 0.3   # 相对差路径的最小绝对差
    rel_qualifies = abs_delta >= MIN_ABS_FOR_REL
    
    if abs_delta >= 10 or (rel_qualifies and rel_delta >= 1.00):
        level = "L4"
        level_text = "极端分歧"
        action = "暂停定量预测，启用人工审查（分级规范 Level 4）"
    elif abs_delta >= 6 or (rel_qualifies and rel_delta >= 0.60):
        level = "L3"
        level_text = "高度分歧"
        action = "启动多轮辩论，置信度 -50%（分级规范 Level 3）"
    elif abs_delta >= 3 or (rel_qualifies and rel_delta >= 0.30):
        level = "L2"
        level_text = "中度分歧"
        action = "触发情境分析辩论，置信度 -20%（分级规范 Level 2）"
    else:
        level = "L1"
        level_text = "轻度分歧"
        action = "加权平均聚合，置信度不变（分级规范 Level 1）"
    
    direction = (
        "model_higher" if delta_pp > 0
        else "market_higher" if delta_pp < 0
        else "tie"
    )
    
    return {
        "delta_pp": round(delta_pp, 2),
        "abs_delta_pp": round(abs_delta, 2),
        "rel_delta_pct": round(rel_delta * 100, 1),
        "level": level,
        "level_text": level_text,
        "direction": direction,
        "action": action,
    }


def detect_market_bias(save_snapshot: bool = True) -> Dict[str, Any]:
    """
    主入口：跑一次完整偏差检测。
    
    Returns: {
      "timestamp": "2026-06-12T20:30:00",
      "method": "...",
      "n_teams": 48,
      "teams": [{team, elo, model_pct, market_pct, ...classify_bias() 输出...}, ...],
      "summary": {by_level, top_overestimated_by_market, top_underestimated_by_market}
    }
    """
    # 加载数据
    teams_raw = json.load(open(ROOT / "data" / "raw" / "teams.json"))["teams"]
    mc_path = ROOT / "data" / "outputs" / "mc_simulation_n100000.json"
    mc = json.load(open(mc_path)) if mc_path.exists() else {}
    
    rows = []
    for name, info in teams_raw.items():
        if info.get("group") in (None, "_"):
            continue
        model_pct = mc.get(name, {}).get("champion", 0) * 100
        market_pct = info.get("market_implied", 0) * 100
        if model_pct == 0 and market_pct == 0:
            continue
        
        bias = classify_bias(model_pct, market_pct)
        rows.append({
            "team": name,
            "elo": info.get("elo"),
            "group": info.get("group"),
            "model_pct": round(model_pct, 2),
            "market_pct": round(market_pct, 2),
            **bias,
        })
    
    # 按绝对偏差倒序
    rows.sort(key=lambda r: -r["abs_delta_pp"])
    
    # 统计
    by_level = {"L1": 0, "L2": 0, "L3": 0, "L4": 0}
    for r in rows:
        by_level[r["level"]] += 1
    
    # 模型相对市场"高估"和"低估"的 Top（潜在黑马）
    top_model_higher = [r for r in rows if r["direction"] == "model_higher"][:8]
    top_market_higher = [r for r in rows if r["direction"] == "market_higher"][:8]
    
    timestamp = datetime.now().isoformat(timespec="seconds")
    
    output = {
        "timestamp": timestamp,
        "method": (
            "分级规范 偏差分级 + §2.1.1 市场作为共识偏差研究变量。"
            "混合阈值：|Δpp| 或 相对差任一触发。"
        ),
        "n_teams": len(rows),
        "by_level": by_level,
        "top_model_higher": top_model_higher,   # 模型比市场高 → 可能模型抓到了被市场低估的黑马
        "top_market_higher": top_market_higher, # 市场比模型高 → 可能市场过度炒作
        "teams": rows,
    }
    
    if save_snapshot:
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        snap_name = f"snapshot_{timestamp.replace(':', '-')}.json"
        (SNAPSHOT_DIR / snap_name).write_text(json.dumps(output, ensure_ascii=False, indent=2))
        LATEST_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    
    return output


def load_latest_snapshot() -> Dict[str, Any]:
    """加载最新的偏差快照"""
    if LATEST_PATH.exists():
        return json.load(open(LATEST_PATH))
    return detect_market_bias()


def list_snapshots(limit: int = 10) -> List[Dict[str, Any]]:
    """列出最近 N 个快照的元数据，供时间序列展示"""
    if not SNAPSHOT_DIR.exists():
        return []
    files = sorted(SNAPSHOT_DIR.glob("snapshot_*.json"), reverse=True)[:limit]
    out = []
    for f in files:
        d = json.load(open(f))
        out.append({
            "timestamp": d["timestamp"],
            "by_level": d["by_level"],
            "n_teams": d["n_teams"],
            "filename": f.name,
        })
    return out


# ===== CLI 自测 =====
if __name__ == "__main__":
    print("=== 市场偏差检测 ===")
    result = detect_market_bias()
    
    print(f"\n时间: {result['timestamp']}")
    print(f"球队数: {result['n_teams']}")
    print(f"分级统计: {result['by_level']}")
    
    print(f"\n=== Top 10 偏差最大 ===")
    print(f"{'#':<3} {'球队':<14} {'模型%':>7} {'市场%':>7} {'Δpp':>7} {'相对%':>7} {'级别':<6} {'方向':<10}")
    print("-" * 80)
    for i, r in enumerate(result['teams'][:10], 1):
        print(f"{i:<3} {r['team']:<14} {r['model_pct']:>6.2f}% {r['market_pct']:>6.2f}%  "
              f"{r['delta_pp']:+6.2f} {r['rel_delta_pct']:>6.1f}%  {r['level']:<6} {r['direction']:<10}")
    
    print(f"\n=== L2+ 球队（值得关注的偏差）===")
    notable = [r for r in result['teams'] if r['level'] in ('L2', 'L3', 'L4')]
    for r in notable[:15]:
        print(f"  [{r['level']}] {r['team']:14s}  模型 {r['model_pct']:.2f}%  vs  市场 {r['market_pct']:.2f}%  "
              f"({r['delta_pp']:+.2f}pp)  {r['action']}")
    
    print(f"\n=== 潜在黑马（模型 > 市场）Top 8 ===")
    for r in result['top_model_higher']:
        print(f"  {r['team']:14s}  模型 {r['model_pct']:.2f}% vs 市场 {r['market_pct']:.2f}%  ({r['delta_pp']:+.2f}pp, {r['level']})")
    
    print(f"\n=== 市场炒作（市场 > 模型）Top 8 ===")
    for r in result['top_market_higher']:
        print(f"  {r['team']:14s}  市场 {r['market_pct']:.2f}% vs 模型 {r['model_pct']:.2f}%  ({r['delta_pp']:+.2f}pp, {r['level']})")
