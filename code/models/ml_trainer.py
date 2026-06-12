"""
基于真实历史数据的 ML 模型训练器

训练数据：32,359 场国际比赛（1990-2026）
模型：Logistic Regression（基础）+ 可扩展到 XGBoost
特征：近期形态 + 历史交锋 + 主客场 + 大洲对阵
目标：3 分类（主胜/平局/客胜）
"""
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.historical_loader import load_results, get_team_form, get_h2h
from utils.io import ROOT


HIST_DIR = ROOT / "data" / "historical"
MODEL_DIR = ROOT / "data" / "outputs"


# 大洲分类（与 venues.json 一致）
TEAM_CONTINENT = {
    # Europe
    "Spain": "Europe", "France": "Europe", "Germany": "Europe", "Italy": "Europe",
    "England": "Europe", "Portugal": "Europe", "Netherlands": "Europe", "Belgium": "Europe",
    "Croatia": "Europe", "Switzerland": "Europe", "Sweden": "Europe", "Norway": "Europe",
    "Austria": "Europe", "Czech Republic": "Europe", "Czechia": "Europe", "Denmark": "Europe",
    "Poland": "Europe", "Russia": "Europe", "Ukraine": "Europe", "Turkey": "Europe", 
    "Serbia": "Europe", "Romania": "Europe", "Greece": "Europe", "Hungary": "Europe",
    "Republic of Ireland": "Europe", "Wales": "Europe", "Scotland": "Europe",
    "Northern Ireland": "Europe", "Bosnia and Herzegovina": "Europe", "Iceland": "Europe",
    "Finland": "Europe", "Slovakia": "Europe", "Bulgaria": "Europe", "Albania": "Europe",
    # South America
    "Brazil": "South America", "Argentina": "South America", "Uruguay": "South America",
    "Colombia": "South America", "Chile": "South America", "Peru": "South America",
    "Ecuador": "South America", "Paraguay": "South America", "Bolivia": "South America",
    "Venezuela": "South America",
    # CONCACAF
    "Mexico": "CONCACAF", "United States": "CONCACAF", "Canada": "CONCACAF",
    "Costa Rica": "CONCACAF", "Honduras": "CONCACAF", "Panama": "CONCACAF",
    "Jamaica": "CONCACAF", "Haiti": "CONCACAF", "Trinidad and Tobago": "CONCACAF",
    # Asia
    "Japan": "Asia", "South Korea": "Asia", "Australia": "Asia", "Iran": "Asia",
    "Saudi Arabia": "Asia", "China PR": "Asia", "Iraq": "Asia", "Qatar": "Asia",
    "United Arab Emirates": "Asia", "Uzbekistan": "Asia", "Jordan": "Asia",
    # Africa
    "Morocco": "Africa", "Egypt": "Africa", "Senegal": "Africa", "Nigeria": "Africa",
    "Ghana": "Africa", "Cameroon": "Africa", "Algeria": "Africa", "Tunisia": "Africa",
    "Ivory Coast": "Africa", "South Africa": "Africa",
    # Oceania
    "New Zealand": "Oceania",
}


CONTINENT_H2H_BIAS = {
    ("Europe", "Asia"): 0.30,
    ("Europe", "Africa"): 0.20,
    ("Europe", "CONCACAF"): 0.25,
    ("Europe", "Oceania"): 0.40,
    ("South America", "Asia"): 0.35,
    ("South America", "Africa"): 0.15,
    ("South America", "Europe"): 0.05,
    ("South America", "CONCACAF"): 0.30,
    ("South America", "Oceania"): 0.45,
    ("CONCACAF", "Asia"): 0.10,
    ("CONCACAF", "Africa"): -0.05,
    ("Africa", "Asia"): 0.10,
    ("Asia", "Oceania"): 0.20,
}


def get_continent_bias(team_a: str, team_b: str) -> float:
    ca = TEAM_CONTINENT.get(team_a, "Europe")
    cb = TEAM_CONTINENT.get(team_b, "Europe")
    if ca == cb:
        return 0.0
    if (ca, cb) in CONTINENT_H2H_BIAS:
        return CONTINENT_H2H_BIAS[(ca, cb)]
    if (cb, ca) in CONTINENT_H2H_BIAS:
        return -CONTINENT_H2H_BIAS[(cb, ca)]
    return 0.0


def build_features_fast(df: pd.DataFrame, sample_size: int = None) -> pd.DataFrame:
    """
    快速构建特征矩阵（用 vectorized 操作）
    """
    print(f"📊 构建特征矩阵...")
    
    # 仅用 1995+ 比赛（更快）
    df = df[df["year"] >= 1995].copy().reset_index(drop=True)
    
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42).sort_values("date").reset_index(drop=True)
    
    print(f"   样本数: {len(df):,}")
    
    # 预计算每队的"截至 D 日的"累计战绩（rolling）
    # 这里用一个简化版本：直接基于 Elo 差异作为代理
    
    features = []
    
    # 给每队一个简化 Elo（基于历史胜率累积）
    print("   计算每队累积胜率...")
    team_stats = {}  # {team: {"matches": [...], "wins": ...}}
    
    # 排序按日期
    df_sorted = df.sort_values("date").reset_index(drop=True)
    
    # 遍历每场比赛，构建特征
    rolling_form = {}  # team -> deque of recent results (1=win, 0=draw, -1=loss)
    
    for idx, m in df_sorted.iterrows():
        ht, at = m["home_team"], m["away_team"]
        
        # 取近 5 场战绩（PPG）
        h_recent = rolling_form.get(ht, [])[-5:]
        a_recent = rolling_form.get(at, [])[-5:]
        h_ppg = (sum(3 if r == 1 else 1 if r == 0.5 else 0 for r in h_recent) / max(1, len(h_recent))) if h_recent else 1.5
        a_ppg = (sum(3 if r == 1 else 1 if r == 0.5 else 0 for r in a_recent) / max(1, len(a_recent))) if a_recent else 1.5
        
        # 大洲对阵
        cont_bias = get_continent_bias(ht, at)
        
        # 主场标记
        home_adv = 0 if m["neutral"] else 1
        
        # 标签
        label = 0 if m["result"] == "H" else 1 if m["result"] == "D" else 2
        
        features.append({
            "form_diff": h_ppg - a_ppg,
            "home_advantage": home_adv,
            "continent_bias": cont_bias,
            "tournament_wc": 1 if "World Cup" in m["tournament"] else 0,
            "label": label,
        })
        
        # 更新 rolling form
        if m["result"] == "H":
            rolling_form.setdefault(ht, []).append(1)
            rolling_form.setdefault(at, []).append(-1)
        elif m["result"] == "A":
            rolling_form.setdefault(ht, []).append(-1)
            rolling_form.setdefault(at, []).append(1)
        else:
            rolling_form.setdefault(ht, []).append(0.5)
            rolling_form.setdefault(at, []).append(0.5)
    
    feat_df = pd.DataFrame(features)
    return feat_df, df_sorted


def train_logistic(X: np.ndarray, y: np.ndarray, n_classes: int = 3) -> dict:
    """
    手工训练多分类 Logistic Regression（不依赖 sklearn）
    用 numpy + 梯度下降
    """
    n_samples, n_features = X.shape
    
    # one-hot
    Y = np.zeros((n_samples, n_classes))
    Y[np.arange(n_samples), y] = 1
    
    # 添加 bias
    X_bias = np.hstack([np.ones((n_samples, 1)), X])
    n_features_ext = n_features + 1
    
    # 初始化参数
    np.random.seed(42)
    W = np.random.randn(n_features_ext, n_classes) * 0.01
    
    lr = 0.1
    n_iter = 500
    
    for i in range(n_iter):
        # softmax
        logits = X_bias @ W
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        
        # 交叉熵梯度
        grad = (probs - Y).T @ X_bias / n_samples
        W -= lr * grad.T
        
        # L2 正则
        W *= (1 - lr * 0.001)
    
    # 计算训练 loss
    logits = X_bias @ W
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    
    # accuracy
    y_pred = probs.argmax(axis=1)
    acc = (y_pred == y).mean()
    
    # 类别准确率
    class_acc = {}
    for c in range(n_classes):
        mask = y == c
        if mask.sum() > 0:
            class_acc[["Home Win", "Draw", "Away Win"][c]] = (y_pred[mask] == c).mean()
    
    return {
        "weights": W.tolist(),
        "accuracy": float(acc),
        "class_accuracy": {k: float(v) for k, v in class_acc.items()},
        "n_iter": n_iter,
        "n_features": n_features,
    }


def train_and_save():
    """完整训练流程"""
    print("=" * 70)
    print("🤖 ML Trainer v0.1 — 基于历史数据训练")
    print("=" * 70)
    
    t0 = time.time()
    
    # 加载数据
    print("\n📥 加载历史数据...")
    df = load_results(min_year=1995)
    df = df[df["tournament"] != "Friendly"].copy()  # 只用正赛
    print(f"   过滤友谊赛后: {len(df):,} 场")
    
    # 构建特征
    feat_df, _ = build_features_fast(df)
    
    print(f"\n✅ 特征矩阵: {feat_df.shape}")
    print(f"   特征列: {[c for c in feat_df.columns if c != 'label']}")
    print(f"\n   类别分布:")
    print(f"     主胜: {(feat_df['label']==0).mean()*100:.1f}%")
    print(f"     平局: {(feat_df['label']==1).mean()*100:.1f}%")
    print(f"     客胜: {(feat_df['label']==2).mean()*100:.1f}%")
    
    # 训练
    print("\n🎯 训练 Logistic Regression...")
    feature_cols = ["form_diff", "home_advantage", "continent_bias", "tournament_wc"]
    X = feat_df[feature_cols].values.astype(float)
    y = feat_df["label"].values.astype(int)
    
    # train/test split (80/20)
    n_test = int(len(X) * 0.2)
    np.random.seed(42)
    idx = np.random.permutation(len(X))
    X_train, X_test = X[idx[n_test:]], X[idx[:n_test]]
    y_train, y_test = y[idx[n_test:]], y[idx[:n_test]]
    
    print(f"   训练集: {len(X_train):,}")
    print(f"   测试集: {len(X_test):,}")
    
    model = train_logistic(X_train, y_train)
    
    print(f"\n📈 训练准确率: {model['accuracy']*100:.1f}%")
    print(f"   各类别准确率:")
    for cls, acc in model['class_accuracy'].items():
        print(f"     {cls:<12} {acc*100:.1f}%")
    
    # 测试集评估
    W = np.array(model["weights"])
    X_test_bias = np.hstack([np.ones((len(X_test), 1)), X_test])
    logits = X_test_bias @ W
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    test_acc = (probs.argmax(axis=1) == y_test).mean()
    
    # Brier score
    Y_test = np.zeros((len(y_test), 3))
    Y_test[np.arange(len(y_test)), y_test] = 1
    brier_test = ((probs - Y_test) ** 2).sum(axis=1).mean()
    
    print(f"\n📊 测试集表现:")
    print(f"   准确率: {test_acc*100:.1f}%")
    print(f"   Brier Score: {brier_test:.3f}")
    
    # 基准对比
    base_acc = max(
        (y_test == 0).mean(),
        (y_test == 1).mean(),
        (y_test == 2).mean(),
    )
    print(f"   基准准确率（永远预测多数类）: {base_acc*100:.1f}%")
    print(f"   提升: +{(test_acc - base_acc)*100:.1f}pp")
    
    elapsed = time.time() - t0
    print(f"\n⏱️  总耗时: {elapsed:.1f} 秒")
    
    # 保存
    model["feature_columns"] = feature_cols
    model["test_accuracy"] = float(test_acc)
    model["test_brier"] = float(brier_test)
    model["baseline_accuracy"] = float(base_acc)
    model["trained_on"] = f"{len(X_train)} matches (1995+, non-friendly)"
    model["trained_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    
    out_path = MODEL_DIR / "ml_model.json"
    with open(out_path, "w") as f:
        json.dump(model, f, indent=2)
    print(f"\n✅ 模型已保存: {out_path.name}")


if __name__ == "__main__":
    train_and_save()
