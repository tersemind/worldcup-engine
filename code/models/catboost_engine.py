"""
CatBoost 3 分类引擎（参考报告模型 10）
=======================================

用 pi-ratings 7 维特征 + 时序 walk-forward 训练 3 分类（H/D/A）

训练流程：
  1. 增量计算每场比赛**赛前**的 pi-ratings 快照（避免数据泄露）
  2. 喂入 CatBoost Classifier（3 分类）
  3. 时序 80/20 split（早 80% 训练，晚 20% 验证）
  4. 输出 RPS / Brier 评估

参考：参考报告 2.2.4 节
  - learning_rate=0.03
  - depth=6
  - iterations=1000 (Early Stopping)
  - l2_leaf_reg=3.0
"""
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.pi_ratings import (
    update_step, expected_goal_diff, build_features,
    LAMBDA, GAMMA, PSI, TOURNAMENT_WEIGHTS,
    train_pi_ratings, save_ratings, load_ratings,
)
from collections import defaultdict, deque


DATA_HIST = Path(__file__).parent.parent.parent / "data" / "historical"
DATA_OUTPUTS = Path(__file__).parent.parent.parent / "data" / "outputs"


# ============ 1. 时序快照训练数据 ============
def build_training_dataset(csv_path: Optional[Path] = None,
                            since_year: int = 1990,
                            burn_in_year: int = 1995) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    构造时序 walk-forward 数据集：
      对每场比赛，先取**赛前**快照特征，再在赛后增量更新评级。
    
    burn_in_year 之前的比赛仅用于初始化评级，不进训练集。
    
    Returns:
        (X_df, y) 形状 (n_matches, 8) / (n_matches,)
        y: 0=主胜, 1=平局, 2=客胜
    """
    csv_path = csv_path or (DATA_HIST / "results.csv")
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"].dt.year >= since_year].sort_values("date").reset_index(drop=True)

    R_H: Dict[str, float] = {}
    R_A: Dict[str, float] = {}
    recent_results = defaultdict(lambda: deque(maxlen=5))

    rows = []
    print(f"⏳ 扫 {len(df):,} 场，构造时序训练集（burn-in={burn_in_year}）...")
    t0 = time.time()
    for idx, row in df.iterrows():
        home, away = row["home_team"], row["away_team"]
        try:
            hs, ascore = int(row["home_score"]), int(row["away_score"])
        except Exception:
            continue
        tournament = row.get("tournament", "Friendly")
        weight = TOURNAMENT_WEIGHTS.get(tournament, 0.8)
        neutral = bool(row.get("neutral", False))

        # 1. 赛前快照特征
        if row["date"].year >= burn_in_year:
            form_5 = {
                home: _form_score(recent_results[home]),
                away: _form_score(recent_results[away]),
            }
            feats = build_features(R_H, R_A, home, away, neutral, form_5)
            # 标签
            if hs > ascore:
                y = 0
            elif hs < ascore:
                y = 2
            else:
                y = 1
            feats["_label"] = y
            feats["_date"] = row["date"]
            feats["_tournament_weight"] = weight
            rows.append(feats)

        # 2. 赛后增量更新
        update_step(R_H, R_A, home, away, hs, ascore,
                     weight=weight, neutral=neutral)
        if hs > ascore:
            recent_results[home].append(("W", hs - ascore))
            recent_results[away].append(("L", ascore - hs))
        elif hs < ascore:
            recent_results[home].append(("L", hs - ascore))
            recent_results[away].append(("W", ascore - hs))
        else:
            recent_results[home].append(("D", 0))
            recent_results[away].append(("D", 0))

        if (idx + 1) % 5000 == 0:
            print(f"  {idx+1:,}/{len(df):,}  ({time.time()-t0:.1f}s)")

    print(f"✓ {len(rows):,} 条训练样本，耗时 {time.time()-t0:.1f}s")

    out_df = pd.DataFrame(rows)
    y = out_df["_label"].values
    X = out_df.drop(columns=["_label", "_date", "_tournament_weight"])
    return out_df, X, y


def _form_score(results_deque) -> Dict:
    """从 deque 计算单队 form_score"""
    if not results_deque:
        return {"form_score": 0.5}
    wins = sum(1 for r, _ in results_deque if r == "W")
    draws = sum(1 for r, _ in results_deque if r == "D")
    return {
        "wins": wins, "draws": draws,
        "form_score": (wins * 3 + draws) / (3 * len(results_deque)),
    }


# ============ 2. 训练 CatBoost ============
def train_catboost(X: pd.DataFrame, y: np.ndarray,
                    weights: Optional[np.ndarray] = None,
                    test_ratio: float = 0.2) -> Dict:
    """
    时序 split + CatBoost 3 分类
    
    Returns: {
        "model": CatBoostClassifier,
        "test_metrics": {"rps", "brier", "accuracy"},
        "feature_importance": {...}
    }
    """
    from catboost import CatBoostClassifier

    n = len(X)
    split = int(n * (1 - test_ratio))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y[:split], y[split:]
    w_train = weights[:split] if weights is not None else None

    print(f"⏳ CatBoost 训练 ({split:,} 训练 / {n - split:,} 验证)...")
    model = CatBoostClassifier(
        iterations=1000,
        depth=6,
        learning_rate=0.03,
        l2_leaf_reg=3.0,
        loss_function="MultiClass",
        early_stopping_rounds=50,
        verbose=0,
        random_seed=42,
    )
    model.fit(X_train, y_train, sample_weight=w_train,
               eval_set=(X_test, y_test))

    # 评估
    proba = model.predict_proba(X_test)  # (n, 3)
    metrics = compute_metrics(y_test, proba)

    # 特征重要性
    fi = dict(zip(X.columns, model.get_feature_importance()))
    fi_sorted = dict(sorted(fi.items(), key=lambda kv: -kv[1]))

    return {
        "model": model,
        "test_metrics": metrics,
        "feature_importance": fi_sorted,
        "n_train": split,
        "n_test": n - split,
    }


def compute_metrics(y_true: np.ndarray, proba: np.ndarray) -> Dict:
    """RPS（Ranked Probability Score）+ Brier + Accuracy"""
    n = len(y_true)
    # one-hot
    onehot = np.zeros_like(proba)
    onehot[np.arange(n), y_true] = 1

    # Brier (multiclass)
    brier = np.mean(np.sum((proba - onehot) ** 2, axis=1))

    # RPS for ordinal H/D/A (lower better)
    cum_pred = np.cumsum(proba[:, :2], axis=1)   # 累积到第二类
    cum_true = np.cumsum(onehot[:, :2], axis=1)
    rps = np.mean(np.sum((cum_pred - cum_true) ** 2, axis=1))

    # Accuracy
    pred = np.argmax(proba, axis=1)
    acc = float(np.mean(pred == y_true))

    return {
        "rps": round(float(rps), 4),
        "brier": round(float(brier), 4),
        "accuracy": round(acc, 4),
    }


# ============ 3. 持久化 ============
def save_model(result: Dict, path: Optional[Path] = None) -> Path:
    path = path or (DATA_OUTPUTS / "catboost_match.cbm")
    result["model"].save_model(str(path))
    # 同时保存 metrics
    meta_path = path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps({
        "test_metrics": result["test_metrics"],
        "feature_importance": result["feature_importance"],
        "n_train": result["n_train"],
        "n_test": result["n_test"],
    }, ensure_ascii=False, indent=2))
    return path


def load_model(path: Optional[Path] = None):
    from catboost import CatBoostClassifier
    path = path or (DATA_OUTPUTS / "catboost_match.cbm")
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} 不存在，先 train")
    model = CatBoostClassifier()
    model.load_model(str(path))
    return model


# ============ 4. 在线预测 ============
def predict_match(home: str, away: str, neutral: bool = False,
                   R_H: Optional[Dict] = None, R_A: Optional[Dict] = None,
                   form_5: Optional[Dict] = None,
                   model=None) -> Dict:
    """
    给定主客队名，返回 {p_home, p_draw, p_away}
    """
    if R_H is None or R_A is None:
        R_H, R_A, meta = load_ratings()
        form_5 = form_5 or meta.get("form_5", {})

    if model is None:
        model = load_model()

    feats = build_features(R_H, R_A, home, away, neutral, form_5)
    X = pd.DataFrame([feats])

    proba = model.predict_proba(X)[0]
    return {
        "p_home": round(float(proba[0]), 4),
        "p_draw": round(float(proba[1]), 4),
        "p_away": round(float(proba[2]), 4),
        "features": feats,
    }


# ============ CLI ============
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "predict", "metrics"])
    ap.add_argument("--home", default="Spain")
    ap.add_argument("--away", default="France")
    ap.add_argument("--neutral", action="store_true")
    args = ap.parse_args()

    if args.cmd == "train":
        out_df, X, y = build_training_dataset()
        # 给较新的比赛更高样本权重（衰减半衰期 ~3 年）
        years = out_df["_date"].dt.year.values
        max_year = years.max()
        weights = 0.5 ** ((max_year - years) / 3.0)

        result = train_catboost(X, y, weights=weights)
        save_model(result)

        print("\n📊 测试集表现:")
        for k, v in result["test_metrics"].items():
            print(f"  {k}: {v}")
        print("\n🔍 Top 5 特征重要性:")
        for k, v in list(result["feature_importance"].items())[:5]:
            print(f"  {k:<20} {v:.2f}")
        print(f"\n📄 模型写入 data/outputs/catboost_match.cbm")

    elif args.cmd == "predict":
        result = predict_match(args.home, args.away, args.neutral)
        print(f"\n{args.home} vs {args.away} (neutral={args.neutral}):")
        print(f"  P(主胜)  = {result['p_home']:.1%}")
        print(f"  P(平局)  = {result['p_draw']:.1%}")
        print(f"  P(客胜)  = {result['p_away']:.1%}")

    elif args.cmd == "metrics":
        meta = json.load(open(DATA_OUTPUTS / "catboost_match.meta.json"))
        print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
