"""
多类校准器（参考报告 2.3.4）
================================

实现三种校准方案对 H/D/A 三分类：

1. Temperature Scaling（Platt 多类扩展）
   p_cal = softmax(z / T), 学一个标量 T
   
2. Vector Scaling（Platt 多类完整版）
   p_cal = softmax(W @ z + b), W: 3×3, b: 3-vec
   
3. Isotonic Regression
   对每个类独立学单调映射 q_cal = isotonic(q_raw)
   再归一化

训练 / 验证流程（避免数据泄露）：
  时序前 80% → 训练校准函数
  时序后 20% → 评估 cal_err / RPS

参考表 2.10 目标：cal_err < 5%
"""
from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression

DATA_OUTPUTS = Path(__file__).parent.parent.parent / "data" / "outputs"


# ============ 评估工具 ============
def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray,
                                  n_bins: int = 10) -> Tuple[float, List[Dict]]:
    """
    多类 ECE（取每条样本的 max prob 作为置信度）
    
    Args:
        probs: (n, 3) 概率矩阵
        y_true: (n,) 0/1/2 真值
    """
    n = len(y_true)
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == y_true).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bins_info = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences >= lo) & (confidences < hi)
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        n_bin = mask.sum()
        if n_bin >= 5:
            avg_conf = confidences[mask].mean()
            avg_acc = correct[mask].mean()
            err = abs(avg_conf - avg_acc)
            ece += (n_bin / n) * err
            bins_info.append({
                "bin": [round(float(lo), 2), round(float(hi), 2)],
                "n": int(n_bin),
                "mean_predicted": round(float(avg_conf), 3),
                "actual_rate": round(float(avg_acc), 3),
                "abs_error": round(float(err), 3),
            })
    return float(ece), bins_info


def brier_multi(probs: np.ndarray, y_true: np.ndarray) -> float:
    """三类 Brier"""
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y_true)), y_true] = 1
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def rps_multi(probs: np.ndarray, y_true: np.ndarray) -> float:
    """有序 RPS（H=0, D=1, A=2）"""
    n = len(y_true)
    onehot = np.zeros_like(probs)
    onehot[np.arange(n), y_true] = 1
    cum_p = np.cumsum(probs[:, :2], axis=1)
    cum_o = np.cumsum(onehot[:, :2], axis=1)
    return float(((cum_p - cum_o) ** 2).sum(axis=1).mean() / 2)


def accuracy(probs: np.ndarray, y_true: np.ndarray) -> float:
    return float((probs.argmax(axis=1) == y_true).mean())


# ============ 1. Temperature Scaling ============
class TemperatureScaling:
    """单参数 T，p_cal = softmax(log(p) / T)。
    
    优化目标可选：NLL（默认）或 ECE（直接最小化校准误差）
    """
    name = "temperature"

    def __init__(self, objective: str = "nll"):
        self.T = 1.0
        self.objective = objective

    def fit(self, probs: np.ndarray, y_true: np.ndarray):
        probs = np.clip(probs, 1e-12, 1.0)
        log_probs = np.log(probs)

        def transform_with_T(T):
            T = max(0.05, T)
            scaled = log_probs / T
            m = scaled.max(axis=1, keepdims=True)
            exp = np.exp(scaled - m)
            return exp / exp.sum(axis=1, keepdims=True)

        if self.objective == "ece":
            # 直接最小化 ECE（不光滑，用网格 + Nelder-Mead 微调）
            best_T, best_ece = 1.0, 1e9
            for T_init in np.linspace(0.5, 2.5, 21):
                p = transform_with_T(T_init)
                ece, _ = expected_calibration_error(p, y_true, n_bins=10)
                if ece < best_ece:
                    best_ece, best_T = ece, T_init
            # 细调
            def ece_loss(params):
                p = transform_with_T(params[0])
                ece, _ = expected_calibration_error(p, y_true, n_bins=10)
                return ece
            res = minimize(ece_loss, x0=[best_T], method="Nelder-Mead",
                            options={"xatol": 1e-3, "fatol": 1e-5})
            self.T = float(max(0.05, res.x[0]))
        else:
            def nll(params):
                T = max(0.05, params[0])
                p = transform_with_T(T)
                p = np.clip(p, 1e-12, 1.0)
                return -np.log(p[np.arange(len(y_true)), y_true]).mean()
            res = minimize(nll, x0=[1.0], method="Nelder-Mead",
                            options={"xatol": 1e-4, "fatol": 1e-6})
            self.T = float(max(0.05, res.x[0]))
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        probs = np.clip(probs, 1e-12, 1.0)
        log_probs = np.log(probs)
        scaled = log_probs / self.T
        m = scaled.max(axis=1, keepdims=True)
        exp = np.exp(scaled - m)
        return exp / exp.sum(axis=1, keepdims=True)

    def to_dict(self):
        return {"name": self.name, "T": self.T, "objective": self.objective}

    @classmethod
    def from_dict(cls, d):
        m = cls(objective=d.get("objective", "nll"))
        m.T = float(d["T"])
        return m


# ============ 2. Vector Scaling ============
class VectorScaling:
    """完整 Platt 多类：p_cal = softmax(W @ log(p) + b)"""
    name = "vector"

    def __init__(self):
        self.W = np.eye(3)
        self.b = np.zeros(3)

    def fit(self, probs: np.ndarray, y_true: np.ndarray):
        probs = np.clip(probs, 1e-12, 1.0)
        log_probs = np.log(probs)

        def nll(params):
            W = params[:9].reshape(3, 3)
            b = params[9:]
            z = log_probs @ W.T + b
            m = z.max(axis=1, keepdims=True)
            exp = np.exp(z - m)
            p = exp / exp.sum(axis=1, keepdims=True)
            p = np.clip(p, 1e-12, 1.0)
            return -np.log(p[np.arange(len(y_true)), y_true]).mean()

        x0 = np.concatenate([np.eye(3).flatten(), np.zeros(3)])
        res = minimize(nll, x0=x0, method="L-BFGS-B",
                        options={"maxiter": 200})
        self.W = res.x[:9].reshape(3, 3)
        self.b = res.x[9:]
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        probs = np.clip(probs, 1e-12, 1.0)
        log_probs = np.log(probs)
        z = log_probs @ self.W.T + self.b
        m = z.max(axis=1, keepdims=True)
        exp = np.exp(z - m)
        return exp / exp.sum(axis=1, keepdims=True)

    def to_dict(self):
        return {"name": self.name, "W": self.W.tolist(), "b": self.b.tolist()}

    @classmethod
    def from_dict(cls, d):
        m = cls()
        m.W = np.array(d["W"])
        m.b = np.array(d["b"])
        return m


# ============ 3. Isotonic Regression（per-class）============
class IsotonicMulti:
    """每类独立学单调映射 + 归一化"""
    name = "isotonic"

    def __init__(self):
        self.regressors: List[IsotonicRegression] = []

    def fit(self, probs: np.ndarray, y_true: np.ndarray):
        self.regressors = []
        n_classes = probs.shape[1]
        for k in range(n_classes):
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(probs[:, k], (y_true == k).astype(float))
            self.regressors.append(iso)
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        n_classes = probs.shape[1]
        cal = np.zeros_like(probs)
        for k in range(n_classes):
            cal[:, k] = self.regressors[k].predict(probs[:, k])
        # 归一化
        sums = cal.sum(axis=1, keepdims=True)
        sums = np.where(sums > 1e-12, sums, 1.0)
        return cal / sums

    def to_dict(self):
        # IsotonicRegression 内部用 X_thresholds_ / y_thresholds_ 表示
        out = {"name": self.name, "regressors": []}
        for r in self.regressors:
            out["regressors"].append({
                "x": r.X_thresholds_.tolist(),
                "y": r.y_thresholds_.tolist(),
            })
        return out

    @classmethod
    def from_dict(cls, d):
        m = cls()
        for r in d["regressors"]:
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.X_thresholds_ = np.array(r["x"])
            iso.y_thresholds_ = np.array(r["y"])
            iso.X_min_ = float(iso.X_thresholds_.min())
            iso.X_max_ = float(iso.X_thresholds_.max())
            iso.increasing_ = True
            # 显式构建插值函数（sklearn 旧版用 f_）
            from scipy.interpolate import interp1d
            iso.f_ = interp1d(iso.X_thresholds_, iso.y_thresholds_,
                               kind="linear", bounds_error=False,
                               fill_value=(float(iso.y_thresholds_[0]),
                                           float(iso.y_thresholds_[-1])))
            m.regressors.append(iso)
        return m


# ============ 统一 fit / evaluate ============
def time_split(probs: np.ndarray, y_true: np.ndarray, ratio: float = 0.8):
    """时序 split，前 ratio 训练，后 1-ratio 验证"""
    n = len(y_true)
    s = int(n * ratio)
    return (probs[:s], y_true[:s]), (probs[s:], y_true[s:])


def evaluate(name: str, probs: np.ndarray, y_true: np.ndarray) -> Dict:
    ece, bins = expected_calibration_error(probs, y_true)
    return {
        "model": name,
        "n": int(len(y_true)),
        "accuracy": round(accuracy(probs, y_true), 4),
        "rps": round(rps_multi(probs, y_true), 4),
        "brier": round(brier_multi(probs, y_true), 4),
        "ece": round(ece, 4),
        "bins": bins,
    }


def fit_all_calibrators(probs_train: np.ndarray, y_train: np.ndarray) -> Dict:
    """fit 4 种校准器（含 ECE-objective temperature）"""
    out = {}
    out["temperature"] = TemperatureScaling(objective="nll").fit(probs_train, y_train)
    out["temperature_ece"] = TemperatureScaling(objective="ece").fit(probs_train, y_train)
    out["vector"] = VectorScaling().fit(probs_train, y_train)
    out["isotonic"] = IsotonicMulti().fit(probs_train, y_train)
    return out


def benchmark(probs: np.ndarray, y_true: np.ndarray,
                ratio: float = 0.8) -> Dict[str, Any]:
    """
    完整 benchmark：
      1. 时序 split
      2. 在训练集 fit 三种校准器
      3. 在验证集上对比 raw / temperature / vector / isotonic
    
    Returns:
        {
            "uncalibrated": {...},
            "temperature": {...},
            "vector": {...},
            "isotonic": {...},
            "calibrators": {...},
            "n_train": int, "n_test": int,
        }
    """
    (p_tr, y_tr), (p_te, y_te) = time_split(probs, y_true, ratio)
    calibrators = fit_all_calibrators(p_tr, y_tr)

    results = {
        "uncalibrated": evaluate("uncalibrated", p_te, y_te),
        "n_train": len(y_tr),
        "n_test": len(y_te),
    }
    for name, cal in calibrators.items():
        p_cal = cal.transform(p_te)
        results[name] = evaluate(name, p_cal, y_te)
    results["calibrators"] = {n: c.to_dict() for n, c in calibrators.items()}
    return results


def save(result: Dict, path: Optional[Path] = None) -> Path:
    path = path or (DATA_OUTPUTS / "calibration_multiclass.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    # 不写 bins 细节进文件，太长
    light = {}
    for k, v in result.items():
        if isinstance(v, dict) and "bins" in v:
            v = {kk: vv for kk, vv in v.items() if kk != "bins"}
        light[k] = v
    path.write_text(json.dumps(light, ensure_ascii=False, indent=2))
    return path


# ============ CLI 入口 ============
def collect_catboost_predictions(include_tournaments: Optional[List[str]] = None,
                                    min_year: int = 2010
                                    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    用 CatBoost + pi-ratings 跑预测，返回 (probs, y_true) 用作校准训练数据
    
    Args:
        include_tournaments: 比赛过滤；None=所有；
            "wc_only" → 仅 FIFA World Cup
            "major"   → WC + Euro + Copa + 大陆杯
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from backtest.walk_forward import _load_history, PiCatBoostModel, actual_outcome

    if include_tournaments == "wc_only":
        tournaments = {"FIFA World Cup"}
    elif include_tournaments == "major":
        tournaments = {"FIFA World Cup", "FIFA World Cup qualification",
                        "UEFA Euro", "UEFA Euro qualification",
                        "Copa América", "Africa Cup of Nations",
                        "AFC Asian Cup", "UEFA Nations League",
                        "CONCACAF Gold Cup"}
    else:
        tournaments = None  # 全部

    df = _load_history(min_year=1990)
    model = PiCatBoostModel()

    rows = []
    for _, row in df.iterrows():
        home, away = row["home_team"], row["away_team"]
        try:
            hs, as_ = int(row["home_score"]), int(row["away_score"])
        except Exception:
            continue
        neutral = bool(row.get("neutral", False))
        tour = row.get("tournament", "Friendly")
        is_eval = (row["date"].year >= min_year
                    and (tournaments is None or tour in tournaments))
        if is_eval:
            p_h, p_d, p_a = model.predict(home=home, away=away, neutral=neutral)
            actual = actual_outcome(hs, as_)
            y = {"H": 0, "D": 1, "A": 2}[actual]
            rows.append((p_h, p_d, p_a, y, row["date"]))
        model.update(home=home, away=away, home_score=hs, away_score=as_,
                      tournament=tour, neutral=neutral)

    rows.sort(key=lambda r: r[4])
    probs = np.array([[r[0], r[1], r[2]] for r in rows])
    y = np.array([r[3] for r in rows])
    return probs, y


def main():
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "benchmark"
    
    if cmd == "benchmark":
        # 训练用 major，验证只看 WC 子集（更接近真实预测场景）
        train_scope = sys.argv[2] if len(sys.argv) > 2 else "major"
        print(f"📥 训练样本（scope={train_scope}）...")
        probs_all, y_all = collect_catboost_predictions(include_tournaments=train_scope)
        print(f"   共 {len(y_all)} 场样本")
        
        print("\n🎯 校准 benchmark（80% 训练 / 20% 验证）—— 全样本对比")
        result = benchmark(probs_all, y_all, ratio=0.8)
        save(result)

        names = ["uncalibrated", "temperature", "temperature_ece", "vector", "isotonic"]
        print(f"\n📊 全样本验证集 n={result['n_test']}:")
        print(f"  {'方法':<18} {'ACC':<8} {'RPS':<8} {'Brier':<8} {'ECE':<8}")
        for name in names:
            r = result[name]
            print(f"  {name:<18} {r['accuracy']:<8.4f} {r['rps']:<8.4f} "
                  f"{r['brier']:<8.4f} {r['ece']:<8.4f}")

        # === 额外：WC-only 子集评估（真实场景）===
        print("\n🌍 WC-only 子集评估（用 major 训练的校准器在 WC 上的表现）")
        wc_probs, wc_y = collect_catboost_predictions(include_tournaments="wc_only")
        print(f"   WC 样本 n={len(wc_y)}")

        # 用 major 训出来的校准器
        calibrators_dict = result["calibrators"]
        from_dict = {
            "temperature": TemperatureScaling.from_dict,
            "temperature_ece": TemperatureScaling.from_dict,
            "vector": VectorScaling.from_dict,
            "isotonic": IsotonicMulti.from_dict,
        }
        wc_results = {"uncalibrated": evaluate("uncalibrated", wc_probs, wc_y)}
        cal_names = ["temperature", "temperature_ece", "vector", "isotonic"]
        for name in cal_names:
            cal = from_dict[name](calibrators_dict[name])
            p_cal = cal.transform(wc_probs)
            wc_results[name] = evaluate(name, p_cal, wc_y)

        print(f"\n  {'方法':<18} {'ACC':<8} {'RPS':<8} {'Brier':<8} {'ECE':<8}")
        for name in ["uncalibrated"] + cal_names:
            r = wc_results[name]
            print(f"  {name:<18} {r['accuracy']:<8.4f} {r['rps']:<8.4f} "
                  f"{r['brier']:<8.4f} {r['ece']:<8.4f}")

        # 最佳 (in WC scope)
        best = min(["uncalibrated"] + cal_names,
                    key=lambda n: wc_results[n]["ece"])
        print(f"\n🏆 在 WC 上最佳: {best} (ECE = {wc_results[best]['ece']})")
        target_met = wc_results[best]['ece'] < 0.05
        print(f"   {'✓' if target_met else '✗'} 精度目标 ECE<5%: {target_met}")

        # 把 WC-only 结果也写入
        result["wc_only_eval"] = wc_results
        save(result)
        print(f"\n📄 完整报告：data/outputs/calibration_multiclass.json")
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
