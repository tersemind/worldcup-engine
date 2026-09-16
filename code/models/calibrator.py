"""
Brier Score 自动校准器（#3）

功能：
1. 接收实际比赛结果，自动计算 Brier Score
2. 维护 history.csv 战绩库
3. 累积 N 场后自动训练 Platt Scaling 校准
4. 输出校准后概率 + 偏差趋势

Brier Score 公式：
  对单事件：Brier = (P_predicted - O_actual)²
  其中 O_actual = 1（事件发生）或 0（未发生）
  
  完美预测 = 0；随机猜测 ≈ 0.25；越低越好
  
Brier 分解：
  Brier = Reliability - Resolution + Uncertainty
"""
import json
import csv
import sys
import math
import time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.io import DATA_RAW

HISTORY_FILE = Path.home() / ".codebuddy" / "worldcup-predict" / "history.csv"
CALIBRATION_FILE = Path.home() / ".codebuddy" / "worldcup-predict" / "data" / "outputs" / "calibration.json"


def init_history():
    """初始化战绩 CSV"""
    if HISTORY_FILE.exists():
        return
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_FILE, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "predict_date", "event", "team", "predicted_prob",
            "ci_lower", "ci_upper", "market_implied", "bias_pp",
            "confidence", "actual_result", "brier_score", "accuracy_1to5"
        ])


def append_prediction(predict_date: str, event: str, team: str,
                      predicted_prob: float, ci_lower: float = None, ci_upper: float = None,
                      market_implied: float = None, bias_pp: float = None,
                      confidence: str = "low"):
    """追加一条预测到战绩库（赛后用 fill_actual_result 回填）"""
    init_history()
    with open(HISTORY_FILE, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            predict_date, event, team,
            f"{predicted_prob:.4f}",
            f"{ci_lower:.2f}" if ci_lower else "",
            f"{ci_upper:.2f}" if ci_upper else "",
            f"{market_implied:.4f}" if market_implied else "",
            f"{bias_pp:+.2f}" if bias_pp is not None else "",
            confidence, "", "", ""
        ])
    print(f"✅ 预测已记录: {team} {predicted_prob*100:.2f}% (event: {event})")


def fill_actual_result(team: str, event: str, actual_result: str,
                       accuracy: int = None) -> dict:
    """
    赛后回填实际结果，自动计算 Brier Score
    
    Args:
        team: 球队名
        event: 事件（如 "2026世界杯冠军"）
        actual_result: "WIN" / "LOSE" / "ADVANCE_QF" 等
        accuracy: 1-5 准确率主观评分
    """
    if not HISTORY_FILE.exists():
        print("❌ 战绩库不存在")
        return None
    
    # 读取所有记录
    rows = []
    with open(HISTORY_FILE, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # 找到对应记录并更新
    updated = None
    for row in rows:
        if row["team"] == team and row["event"] == event and not row["actual_result"]:
            row["actual_result"] = actual_result
            
            # 计算 Brier
            o_actual = 1.0 if actual_result.upper() == "WIN" else 0.0
            p_pred = float(row["predicted_prob"])
            brier = (p_pred - o_actual) ** 2
            row["brier_score"] = f"{brier:.4f}"
            
            if accuracy is not None:
                row["accuracy_1to5"] = str(accuracy)
            
            updated = dict(row)
            break
    
    if not updated:
        print(f"❌ 未找到匹配记录: {team} - {event}")
        return None
    
    # 写回
    with open(HISTORY_FILE, "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
    
    print(f"✅ 已回填 {team} - {event}")
    print(f"   预测: {float(updated['predicted_prob'])*100:.2f}% → 实际: {actual_result}")
    print(f"   Brier Score: {updated['brier_score']}")
    
    return updated


def compute_brier_decomposition(records: list) -> dict:
    """
    分解 Brier Score 为 Reliability + Resolution + Uncertainty
    
    Reliability：预测概率与实际频率的偏差（越低越好）
    Resolution：预测的"区分度"（越高越好）
    Uncertainty：结果本身的随机性（不可控）
    """
    if not records:
        return {}
    
    n = len(records)
    
    # 按预测概率分箱（10 个箱）
    bins = defaultdict(list)
    for r in records:
        p = float(r["predicted_prob"])
        actual = 1.0 if r["actual_result"].upper() == "WIN" else 0.0
        bin_idx = min(int(p * 10), 9)
        bins[bin_idx].append((p, actual))
    
    # 总平均结果
    total_actual = sum(1.0 if r["actual_result"].upper() == "WIN" else 0.0 for r in records)
    base_rate = total_actual / n
    
    # Reliability + Resolution
    reliability = 0.0
    resolution = 0.0
    for bin_idx, items in bins.items():
        n_bin = len(items)
        avg_p = sum(p for p, _ in items) / n_bin
        avg_actual = sum(a for _, a in items) / n_bin
        reliability += n_bin / n * (avg_p - avg_actual) ** 2
        resolution += n_bin / n * (avg_actual - base_rate) ** 2
    
    uncertainty = base_rate * (1 - base_rate)
    
    # 总 Brier
    brier = sum(float(r["brier_score"]) for r in records) / n
    
    return {
        "n_records": n,
        "mean_brier": round(brier, 4),
        "reliability": round(reliability, 4),  # 低=好
        "resolution": round(resolution, 4),    # 高=好
        "uncertainty": round(uncertainty, 4),
        "base_rate": round(base_rate, 4),
        "interpretation": (
            "Excellent" if brier < 0.10 else
            "Good" if brier < 0.20 else
            "Average" if brier < 0.25 else
            "Poor"
        ),
    }


def compute_calibration_curve(records: list, n_bins: int = 10) -> list:
    """
    校准曲线：每个概率区间的预测概率 vs 实际频率
    完美校准的模型应在 y=x 直线上
    """
    bins = defaultdict(list)
    for r in records:
        p = float(r["predicted_prob"])
        actual = 1.0 if r["actual_result"].upper() == "WIN" else 0.0
        bin_idx = min(int(p * n_bins), n_bins - 1)
        bins[bin_idx].append((p, actual))
    
    curve = []
    for bin_idx in sorted(bins.keys()):
        items = bins[bin_idx]
        avg_p = sum(p for p, _ in items) / len(items)
        avg_actual = sum(a for _, a in items) / len(items)
        curve.append({
            "bin": bin_idx,
            "predicted_prob": round(avg_p, 3),
            "actual_freq": round(avg_actual, 3),
            "n_samples": len(items),
            "calibration_error": round(abs(avg_p - avg_actual), 3),
        })
    return curve


def fit_platt_scaling(records: list) -> dict:
    """
    简化版 Platt Scaling：
      P_calibrated = 1 / (1 + exp(A × P_raw + B))
    
    用 logistic 回归拟合 (A, B) 使 Brier 最小
    """
    if len(records) < 10:
        return {"A": 1.0, "B": 0.0, "n_samples": len(records),
                "warning": "Need ≥10 samples for reliable calibration"}
    
    import numpy as np
    from scipy.optimize import minimize
    
    p_raw = np.array([float(r["predicted_prob"]) for r in records])
    actual = np.array([1.0 if r["actual_result"].upper() == "WIN" else 0.0 for r in records])
    
    # 避免极值
    p_raw = np.clip(p_raw, 1e-6, 1 - 1e-6)
    
    def neg_log_likelihood(params):
        A, B = params
        # logit(p_raw) 变换后线性
        z = -A * p_raw - B
        p_cal = 1.0 / (1.0 + np.exp(z))
        p_cal = np.clip(p_cal, 1e-6, 1 - 1e-6)
        return -np.sum(actual * np.log(p_cal) + (1 - actual) * np.log(1 - p_cal))
    
    result = minimize(neg_log_likelihood, x0=[-1.0, 0.0], method="Nelder-Mead")
    A, B = result.x
    
    return {
        "A": round(float(A), 4),
        "B": round(float(B), 4),
        "n_samples": len(records),
        "converged": result.success,
    }


def apply_calibration(p_raw: float, calibration: dict) -> float:
    """应用 Platt Scaling 校准"""
    A = calibration.get("A", 1.0)
    B = calibration.get("B", 0.0)
    z = -A * p_raw - B
    return 1.0 / (1.0 + math.exp(z))


def report():
    """生成校准报告"""
    if not HISTORY_FILE.exists():
        print("❌ 战绩库为空")
        return
    
    with open(HISTORY_FILE, "r") as f:
        reader = csv.DictReader(f)
        all_records = list(reader)
    
    verified = [r for r in all_records if r["actual_result"]]
    
    print("=" * 70)
    print("📊 Brier Score 校准报告")
    print("=" * 70)
    print(f"\n总预测数: {len(all_records)}")
    print(f"已验证:   {len(verified)}")
    print(f"待验证:   {len(all_records) - len(verified)}")
    
    if len(verified) == 0:
        print("\n⚠️  无已验证记录，无法计算 Brier Score")
        return
    
    # Brier 分解
    decomp = compute_brier_decomposition(verified)
    print(f"\n📈 Brier Score 分解：")
    print(f"  Mean Brier:   {decomp['mean_brier']}  ({decomp['interpretation']})")
    print(f"  Reliability:  {decomp['reliability']}  (低=好)")
    print(f"  Resolution:   {decomp['resolution']}  (高=好)")
    print(f"  Uncertainty:  {decomp['uncertainty']}")
    print(f"  Base rate:    {decomp['base_rate']}")
    
    # 校准曲线
    if len(verified) >= 5:
        print(f"\n📐 校准曲线（预测 vs 实际）：")
        curve = compute_calibration_curve(verified, n_bins=10)
        print(f"  {'Bin':<5} {'Pred':<8} {'Actual':<8} {'Error':<8} {'N':<5}")
        for c in curve:
            print(f"  {c['bin']:<5} {c['predicted_prob']:<8.3f} {c['actual_freq']:<8.3f} {c['calibration_error']:<8.3f} {c['n_samples']:<5}")
    
    # Platt Scaling
    if len(verified) >= 10:
        platt = fit_platt_scaling(verified)
        print(f"\n🎯 Platt Scaling 校准参数：")
        print(f"  A = {platt['A']}, B = {platt['B']}")
        print(f"  Samples: {platt['n_samples']}")
        if not platt.get("converged", True):
            print(f"  ⚠️  Optimization did not converge")
        
        # 保存校准参数供 synthesizer 使用
        CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CALIBRATION_FILE, "w") as f:
            json.dump({
                "platt_scaling": platt,
                "decomposition": decomp,
                "calibration_curve": curve,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, f, indent=2)
        print(f"\n✅ 校准参数已保存: {CALIBRATION_FILE.name}")
    else:
        print(f"\n⏳ 需要 ≥10 条已验证记录才能训练 Platt Scaling（当前 {len(verified)}）")


def add_demo_records():
    """添加 12 条模拟历史预测（用于演示）"""
    init_history()
    demo = [
        # 假设过去做了 12 个比赛预测，含已验证结果
        ("2026-06-09", "2026WC_Group_Match1", "Spain", 0.85, 0.80, 0.90, 0.82, 3.0, "high", "WIN"),
        ("2026-06-09", "2026WC_Group_Match1", "France", 0.78, 0.72, 0.84, 0.75, 3.0, "high", "WIN"),
        ("2026-06-09", "2026WC_Group_Match1", "Argentina", 0.78, 0.72, 0.84, 0.75, 3.0, "high", "WIN"),
        ("2026-06-09", "2026WC_Group_Match1", "Brazil", 0.91, 0.85, 0.97, 0.88, 3.0, "high", "WIN"),
        ("2026-06-09", "2026WC_Group_Match1", "Germany", 0.95, 0.92, 0.98, 0.93, 2.0, "high", "WIN"),
        ("2026-06-09", "2026WC_Group_Match1", "England", 0.75, 0.68, 0.82, 0.72, 3.0, "high", "WIN"),
        ("2026-06-09", "2026WC_Group_Match2", "Croatia", 0.55, 0.45, 0.65, 0.50, 5.0, "med", "LOSE"),
        ("2026-06-09", "2026WC_Group_Match2", "Belgium", 0.65, 0.55, 0.75, 0.60, 5.0, "med", "WIN"),
        ("2026-06-09", "2026WC_Group_Match2", "Japan", 0.40, 0.30, 0.50, 0.45, -5.0, "low", "WIN"),  # 爆冷
        ("2026-06-09", "2026WC_Group_Match2", "Mexico", 0.55, 0.45, 0.65, 0.60, -5.0, "med", "WIN"),
        ("2026-06-09", "2026WC_Group_Match3", "Senegal", 0.45, 0.35, 0.55, 0.42, 3.0, "low", "LOSE"),
        ("2026-06-09", "2026WC_Group_Match3", "USA", 0.55, 0.45, 0.65, 0.50, 5.0, "med", "WIN"),
    ]
    
    with open(HISTORY_FILE, "a", newline="") as f:
        w = csv.writer(f)
        for row in demo:
            brier = (row[3] - (1.0 if row[9] == "WIN" else 0.0)) ** 2
            w.writerow([
                row[0], row[1], row[2], f"{row[3]:.4f}",
                f"{row[4]:.2f}", f"{row[5]:.2f}", f"{row[6]:.4f}",
                f"{row[7]:+.2f}", row[8], row[9], f"{brier:.4f}", ""
            ])
    
    print(f"✅ 已添加 {len(demo)} 条演示记录")


def main():
    if len(sys.argv) == 1 or sys.argv[1] == "report":
        report()
    elif sys.argv[1] == "demo":
        add_demo_records()
        print()
        report()
    elif sys.argv[1] == "fill" and len(sys.argv) >= 5:
        team = sys.argv[2]
        event = sys.argv[3]
        result = sys.argv[4]
        accuracy = int(sys.argv[5]) if len(sys.argv) > 5 else None
        fill_actual_result(team, event, result, accuracy)
    elif sys.argv[1] == "init":
        init_history()
        print(f"✅ 战绩库已初始化: {HISTORY_FILE}")
    else:
        print("用法:")
        print("  python3 calibrator.py report                       # 校准报告")
        print("  python3 calibrator.py fill <team> <event> <WIN/LOSE> [accuracy]  # 回填实际结果")
        print("  python3 calibrator.py demo                         # 添加演示记录并报告")
        print("  python3 calibrator.py init                         # 初始化战绩库")


if __name__ == "__main__":
    main()
