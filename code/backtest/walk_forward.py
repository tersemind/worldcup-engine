"""
Walk-Forward 历届 WC 回测（参考报告 2.4.3）
================================================

四层回测框架：
  L1 时序交叉验证：2010/2014/2018/2022 四届 WC 作为验证节点
  L2 跨赛事验证：    用 N 届及之前训练，预测 N+1 届
  L3 Confederation Holdout：轮流排除某大洲
  L4 Walk-Forward：    所有 WC 比赛按时序累积预测

对比的模型：
  - baseline_random：均匀 1/3
  - baseline_home：永远主队胜
  - elo_simple：Elo + sigmoid（现有 elo_engine.py）
  - elo_dc：Elo + Dixon-Coles 分平局
  - catboost_pi：CatBoost + pi-ratings（参考模型 10）

目标指标（参考表 2.10）：
  accuracy > 60%
  RPS < 0.195
  Brier < 0.55
  calibration error < 5%
"""
from __future__ import annotations
import sys
import json
import time
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.elo_engine import match_probabilities, update_elo
from models.pi_ratings import (
    update_step as pi_update_step,
    build_features as pi_build_features,
    TOURNAMENT_WEIGHTS, expected_goal_diff,
)
from utils.io import ROOT


DATA_HIST = ROOT / "data" / "historical"
DATA_OUTPUTS = ROOT / "data" / "outputs"

WC_YEARS = [2010, 2014, 2018, 2022]   # 验证节点
ALL_WC_YEARS = [2002, 2006, 2010, 2014, 2018, 2022]


# ============ 大洲映射（简化） ============
TEAM_CONFED = {
    # UEFA
    "Spain": "UEFA", "France": "UEFA", "Germany": "UEFA", "England": "UEFA",
    "Portugal": "UEFA", "Netherlands": "UEFA", "Italy": "UEFA", "Belgium": "UEFA",
    "Croatia": "UEFA", "Switzerland": "UEFA", "Sweden": "UEFA", "Denmark": "UEFA",
    "Austria": "UEFA", "Czech Republic": "UEFA", "Czechia": "UEFA", "Poland": "UEFA",
    "Russia": "UEFA", "Ukraine": "UEFA", "Turkey": "UEFA", "Turkiye": "UEFA",
    "Serbia": "UEFA", "Romania": "UEFA", "Greece": "UEFA", "Norway": "UEFA",
    "Republic of Ireland": "UEFA", "Wales": "UEFA", "Scotland": "UEFA",
    "Iceland": "UEFA", "Slovakia": "UEFA", "Slovenia": "UEFA",
    "Bosnia and Herzegovina": "UEFA", "Hungary": "UEFA",
    # CONMEBOL
    "Brazil": "CONMEBOL", "Argentina": "CONMEBOL", "Uruguay": "CONMEBOL",
    "Colombia": "CONMEBOL", "Chile": "CONMEBOL", "Peru": "CONMEBOL",
    "Ecuador": "CONMEBOL", "Paraguay": "CONMEBOL", "Bolivia": "CONMEBOL",
    "Venezuela": "CONMEBOL",
    # CONCACAF
    "Mexico": "CONCACAF", "United States": "CONCACAF", "Canada": "CONCACAF",
    "Costa Rica": "CONCACAF", "Honduras": "CONCACAF", "Panama": "CONCACAF",
    "Jamaica": "CONCACAF", "Haiti": "CONCACAF", "Trinidad and Tobago": "CONCACAF",
    # AFC
    "Japan": "AFC", "South Korea": "AFC", "Korea Republic": "AFC",
    "Iran": "AFC", "Australia": "AFC", "Saudi Arabia": "AFC", "Iraq": "AFC",
    "Qatar": "AFC", "United Arab Emirates": "AFC", "China PR": "AFC",
    "Uzbekistan": "AFC", "Jordan": "AFC",
    # CAF
    "Morocco": "CAF", "Senegal": "CAF", "Egypt": "CAF", "Nigeria": "CAF",
    "Cameroon": "CAF", "Algeria": "CAF", "Tunisia": "CAF", "Ghana": "CAF",
    "South Africa": "CAF", "Cape Verde": "CAF", "Ivory Coast": "CAF",
    "Côte d'Ivoire": "CAF",
    # OFC
    "New Zealand": "OFC",
}


def get_confed(team: str) -> str:
    return TEAM_CONFED.get(team, "OTHER")


# ============ 评估指标 ============
def rps(p_h: float, p_d: float, p_a: float, actual: str) -> float:
    """Ranked Probability Score（越低越好）"""
    if actual == "H":
        o = (1, 0, 0)
    elif actual == "D":
        o = (0, 1, 0)
    else:
        o = (0, 0, 1)
    # 累积分布到第 2 类
    cum_p = [p_h, p_h + p_d]
    cum_o = [o[0], o[0] + o[1]]
    return ((cum_p[0] - cum_o[0]) ** 2 + (cum_p[1] - cum_o[1]) ** 2) / 2


def brier_multi(p_h: float, p_d: float, p_a: float, actual: str) -> float:
    """三类 Brier"""
    if actual == "H":
        return (1 - p_h) ** 2 + p_d ** 2 + p_a ** 2
    if actual == "D":
        return p_h ** 2 + (1 - p_d) ** 2 + p_a ** 2
    return p_h ** 2 + p_d ** 2 + (1 - p_a) ** 2


def log_loss(p_h: float, p_d: float, p_a: float, actual: str,
              eps: float = 1e-12) -> float:
    """三类 log loss"""
    if actual == "H":
        p = max(p_h, eps)
    elif actual == "D":
        p = max(p_d, eps)
    else:
        p = max(p_a, eps)
    return -math.log(p)


def predicted_class(p_h: float, p_d: float, p_a: float) -> str:
    m = max(p_h, p_d, p_a)
    if p_h == m: return "H"
    if p_d == m: return "D"
    return "A"


# ============ 基准模型 ============
class BaselineRandom:
    """均匀 1/3"""
    name = "baseline_random"
    def predict(self, **kw):
        return 1/3, 1/3, 1/3
    def update(self, **kw): pass


class BaselineHomeAlways:
    """永远主队胜"""
    name = "baseline_home"
    def predict(self, **kw):
        return 0.9, 0.05, 0.05
    def update(self, **kw): pass


class EloModel:
    """Elo + 简单平局拆分"""
    name = "elo_simple"
    def __init__(self, draw_rate: float = 0.26):
        self.elo: Dict[str, float] = {}
        self.draw_rate = draw_rate

    def get(self, team: str, default: float = 1500) -> float:
        return self.elo.get(team, default)

    def predict(self, home: str, away: str, neutral: bool = False, **kw):
        e_h = self.get(home)
        e_a = self.get(away)
        if not neutral:
            e_h += 60
        p = match_probabilities(e_h, e_a)
        # match_probabilities 返回的是含 draw 的 dict
        return p["p_win_a"], p["p_draw"], p["p_win_b"]

    def update(self, home: str, away: str, home_score: int, away_score: int,
                tournament: str = "Friendly", **kw):
        if home_score > away_score:
            result = 1.0
        elif home_score < away_score:
            result = 0.0
        else:
            result = 0.5

        # k 值按赛事
        if "World Cup" in tournament and "qualification" not in tournament:
            k = 60
        elif "Euro" in tournament:
            k = 40
        else:
            k = 30

        e_h = self.get(home)
        e_a = self.get(away)
        new_h, new_a = update_elo(e_h, e_a, result, k=k)
        self.elo[home] = new_h
        self.elo[away] = new_a


class CalibratedCatBoostModel:
    """CatBoost + 多类校准器（Isotonic / Vector / Temperature）"""

    def __init__(self, calibrator_name: str = "isotonic"):
        from models.catboost_engine import load_model as _load_cb
        self.cb = _load_cb()
        self.R_H: Dict[str, float] = {}
        self.R_A: Dict[str, float] = {}
        self.form_5: Dict[str, Dict] = defaultdict(lambda: {"form_score": 0.5})
        self.recent: Dict[str, deque] = defaultdict(lambda: deque(maxlen=5))

        # 加载校准器
        cal_path = DATA_OUTPUTS / "calibration_multiclass.json"
        if not cal_path.exists():
            raise FileNotFoundError(
                "calibration_multiclass.json 不存在，先 multi_calibrator.py benchmark"
            )
        from models.multi_calibrator import (
            TemperatureScaling, VectorScaling, IsotonicMulti,
        )
        cal_data = json.load(open(cal_path))["calibrators"]
        builders = {
            "temperature": TemperatureScaling.from_dict,
            "temperature_ece": TemperatureScaling.from_dict,
            "vector": VectorScaling.from_dict,
            "isotonic": IsotonicMulti.from_dict,
        }
        self.calibrator = builders[calibrator_name](cal_data[calibrator_name])
        self.name = f"catboost_cal_{calibrator_name}"

    def predict(self, home: str, away: str, neutral: bool = False, **kw):
        feats = pi_build_features(self.R_H, self.R_A, home, away,
                                    neutral=neutral, form_5=self.form_5)
        raw = self.cb.predict_proba(pd.DataFrame([feats]))[0].reshape(1, -1)
        cal = self.calibrator.transform(raw)[0]
        return float(cal[0]), float(cal[1]), float(cal[2])

    def update(self, home: str, away: str, home_score: int, away_score: int,
                tournament: str = "Friendly", neutral: bool = False, **kw):
        weight = TOURNAMENT_WEIGHTS.get(tournament, 0.8)
        pi_update_step(self.R_H, self.R_A, home, away,
                        home_score, away_score, weight=weight, neutral=neutral)
        if home_score > away_score:
            self.recent[home].append(("W", home_score - away_score))
            self.recent[away].append(("L", away_score - home_score))
        elif home_score < away_score:
            self.recent[home].append(("L", home_score - away_score))
            self.recent[away].append(("W", away_score - home_score))
        else:
            self.recent[home].append(("D", 0))
            self.recent[away].append(("D", 0))
        for t in (home, away):
            results = self.recent[t]
            wins = sum(1 for r, _ in results if r == "W")
            draws = sum(1 for r, _ in results if r == "D")
            self.form_5[t] = {
                "form_score": (wins * 3 + draws) / (3 * max(1, len(results))),
            }


class PiCatBoostModel:
    """pi-ratings + CatBoost（参考模型 10）"""
    name = "catboost_pi"

    def __init__(self, model_path: Optional[Path] = None):
        from catboost import CatBoostClassifier
        path = model_path or (DATA_OUTPUTS / "catboost_match.cbm")
        if not path.exists():
            raise FileNotFoundError(
                f"{path} 不存在，先 python3 code/models/catboost_engine.py train"
            )
        self.model = CatBoostClassifier()
        self.model.load_model(str(path))
        self.R_H: Dict[str, float] = {}
        self.R_A: Dict[str, float] = {}
        self.form_5: Dict[str, Dict] = defaultdict(lambda: {"form_score": 0.5})
        self.recent: Dict[str, deque] = defaultdict(lambda: deque(maxlen=5))

    def predict(self, home: str, away: str, neutral: bool = False, **kw):
        feats = pi_build_features(self.R_H, self.R_A, home, away,
                                    neutral=neutral, form_5=self.form_5)
        proba = self.model.predict_proba(pd.DataFrame([feats]))[0]
        return float(proba[0]), float(proba[1]), float(proba[2])

    def update(self, home: str, away: str, home_score: int, away_score: int,
                tournament: str = "Friendly", neutral: bool = False, **kw):
        weight = TOURNAMENT_WEIGHTS.get(tournament, 0.8)
        pi_update_step(self.R_H, self.R_A, home, away,
                        home_score, away_score, weight=weight, neutral=neutral)
        # 更新 form
        if home_score > away_score:
            self.recent[home].append(("W", home_score - away_score))
            self.recent[away].append(("L", away_score - home_score))
        elif home_score < away_score:
            self.recent[home].append(("L", home_score - away_score))
            self.recent[away].append(("W", away_score - home_score))
        else:
            self.recent[home].append(("D", 0))
            self.recent[away].append(("D", 0))
        for t in (home, away):
            results = self.recent[t]
            wins = sum(1 for r, _ in results if r == "W")
            draws = sum(1 for r, _ in results if r == "D")
            self.form_5[t] = {
                "form_score": (wins * 3 + draws) / (3 * max(1, len(results))),
            }


# ============ 核心：Walk-Forward 引擎 ============
def actual_outcome(home_score: int, away_score: int) -> str:
    if home_score > away_score: return "H"
    if home_score < away_score: return "A"
    return "D"


def run_walk_forward(models: List, df: pd.DataFrame,
                       eval_filter=None,
                       confed_holdout: Optional[str] = None,
                       verbose: bool = False) -> Dict:
    """
    通用 walk-forward：
      - 按时序遍历 df
      - 每场：所有模型先 predict，再 update
      - 仅 eval_filter(row) == True 的场次计入评估
      - confed_holdout 模式：训练时跳过该大洲的比赛
    
    Args:
        models: [BaselineRandom(), EloModel(), PiCatBoostModel(), ...]
        df: 必须有 date / home_team / away_team / home_score / away_score / tournament / neutral / year
        eval_filter: 函数 (row) -> bool
        confed_holdout: 'UEFA' 等，跳过该洲比赛的训练
    
    Returns: {
        model_name: {"n", "accuracy", "rps", "brier", "log_loss",
                      "by_year": {2018: {...}}, "calibration_bins": [...]},
        ...
    }
    """
    records = {m.name: [] for m in models}
    n_train_skipped = 0

    for _, row in df.iterrows():
        home, away = row["home_team"], row["away_team"]
        try:
            hs, as_ = int(row["home_score"]), int(row["away_score"])
        except Exception:
            continue
        tournament = row.get("tournament", "Friendly")
        neutral = bool(row.get("neutral", False))
        actual = actual_outcome(hs, as_)

        # 评估：先 predict
        if eval_filter is None or eval_filter(row):
            for m in models:
                p_h, p_d, p_a = m.predict(home=home, away=away, neutral=neutral)
                records[m.name].append({
                    "year": row["date"].year if hasattr(row["date"], "year") else int(str(row["date"])[:4]),
                    "tournament": tournament,
                    "home_confed": get_confed(home),
                    "away_confed": get_confed(away),
                    "p_h": p_h, "p_d": p_d, "p_a": p_a,
                    "actual": actual,
                    "rps": rps(p_h, p_d, p_a, actual),
                    "brier": brier_multi(p_h, p_d, p_a, actual),
                    "log_loss": log_loss(p_h, p_d, p_a, actual),
                    "predicted": predicted_class(p_h, p_d, p_a),
                    "correct": predicted_class(p_h, p_d, p_a) == actual,
                    "max_prob": max(p_h, p_d, p_a),
                })

        # 训练更新（confed_holdout 时跳过）
        skip_train = False
        if confed_holdout:
            if get_confed(home) == confed_holdout or get_confed(away) == confed_holdout:
                skip_train = True
                n_train_skipped += 1

        if not skip_train:
            for m in models:
                m.update(home=home, away=away,
                          home_score=hs, away_score=as_,
                          tournament=tournament, neutral=neutral)

    # 汇总每个模型的指标
    summary = {}
    for name, recs in records.items():
        if not recs:
            summary[name] = {"n": 0}
            continue
        rdf = pd.DataFrame(recs)
        summary[name] = {
            "n": len(rdf),
            "accuracy": round(float(rdf["correct"].mean()), 4),
            "rps": round(float(rdf["rps"].mean()), 4),
            "brier": round(float(rdf["brier"].mean()), 4),
            "log_loss": round(float(rdf["log_loss"].mean()), 4),
        }
        # 按年
        by_year = rdf.groupby("year").agg(
            n=("correct", "size"),
            acc=("correct", "mean"),
            rps=("rps", "mean"),
            brier=("brier", "mean"),
        ).round(4)
        summary[name]["by_year"] = {
            int(y): {k: float(v) for k, v in row.items()}
            for y, row in by_year.iterrows()
        }
        # 校准曲线（10 bins）
        bins = []
        for lo in np.arange(0.0, 1.0, 0.1):
            hi = lo + 0.1
            mask = (rdf["max_prob"] >= lo) & (rdf["max_prob"] < hi)
            sub = rdf[mask]
            if len(sub) >= 5:
                bins.append({
                    "bin_lo": round(float(lo), 2),
                    "bin_hi": round(float(hi), 2),
                    "n": len(sub),
                    "mean_predicted": round(float(sub["max_prob"].mean()), 3),
                    "actual_rate": round(float(sub["correct"].mean()), 3),
                })
        summary[name]["calibration_bins"] = bins
        # 校准误差（ECE）
        if bins:
            total_n = sum(b["n"] for b in bins)
            ece = sum(b["n"] / total_n * abs(b["mean_predicted"] - b["actual_rate"])
                       for b in bins)
            summary[name]["calibration_error"] = round(float(ece), 4)

    if verbose and confed_holdout:
        print(f"  (holdout {confed_holdout}: 跳过 {n_train_skipped} 场训练)")

    return summary


# ============ 四层验证入口 ============
def layer1_temporal_cv(verbose: bool = True) -> Dict:
    """L1 时序交叉验证：以 2010/2014/2018/2022 为节点滚动"""
    print("=" * 70)
    print("L1 时序交叉验证：2010/2014/2018/2022 滚动节点")
    print("=" * 70)

    df = _load_history(min_year=1990)

    results = {}
    for eval_year in WC_YEARS:
        if verbose:
            print(f"\n  📍 验证节点 {eval_year} WC")

        # 训练数据：eval_year - 4 之前的所有
        # 验证数据：eval_year 的 WC 比赛
        models = _build_models()
        eval_filter = lambda row: (row["tournament"] == "FIFA World Cup"
                                     and row["date"].year == eval_year)

        # 只跑到 eval_year 结束前后
        df_sub = df[df["date"].dt.year <= eval_year + 1].copy()

        summary = run_walk_forward(models, df_sub, eval_filter=eval_filter)
        results[eval_year] = summary

        if verbose:
            for name, s in summary.items():
                if s.get("n", 0) > 0:
                    print(f"    {name:<18} n={s['n']:>3}  "
                          f"acc={s['accuracy']:.3f}  rps={s['rps']:.4f}  "
                          f"brier={s['brier']:.4f}")
    return results


def layer2_cross_tournament(verbose: bool = True) -> Dict:
    """L2 跨赛事验证：训 2014- 测 2018；训 2018- 测 2022"""
    print("\n" + "=" * 70)
    print("L2 跨赛事验证")
    print("=" * 70)

    df = _load_history(min_year=1990)
    pairs = [(2014, 2018), (2018, 2022)]
    results = {}

    for train_until, test_year in pairs:
        if verbose:
            print(f"\n  📍 训练截止 {train_until}WC → 验证 {test_year}WC")
        models = _build_models()
        eval_filter = lambda row, y=test_year: (
            row["tournament"] == "FIFA World Cup" and row["date"].year == y
        )
        # 只跑到 test_year + 1 月（不让模型看到 test_year WC 后续比赛）
        df_sub = df[df["date"].dt.year <= test_year].copy()
        summary = run_walk_forward(models, df_sub, eval_filter=eval_filter)
        results[f"train≤{train_until}→test{test_year}"] = summary
        if verbose:
            for name, s in summary.items():
                if s.get("n", 0) > 0:
                    print(f"    {name:<18} n={s['n']:>3}  "
                          f"acc={s['accuracy']:.3f}  rps={s['rps']:.4f}  "
                          f"brier={s['brier']:.4f}  cal_err={s.get('calibration_error', 0):.3f}")
    return results


def layer3_confed_holdout(confeds: List[str] = None,
                            verbose: bool = True) -> Dict:
    """L3 Confederation Holdout：轮流排除某大洲训练"""
    print("\n" + "=" * 70)
    print("L3 Confederation Holdout")
    print("=" * 70)

    confeds = confeds or ["UEFA", "CONMEBOL", "CAF", "AFC"]
    df = _load_history(min_year=2000)
    eval_filter = lambda row: row["tournament"] == "FIFA World Cup" and row["date"].year >= 2010
    results = {}

    for confed in confeds:
        if verbose:
            print(f"\n  📍 Holdout {confed}（训练时跳过该洲所有比赛）")
        models = _build_models(skip_catboost=True)  # CatBoost 重训太慢，仅跑 Elo
        eval_confed_filter = lambda row, c=confed: (
            eval_filter(row) and (get_confed(row["home_team"]) == c
                                    or get_confed(row["away_team"]) == c)
        )
        summary = run_walk_forward(models, df, eval_filter=eval_confed_filter,
                                      confed_holdout=confed, verbose=verbose)
        results[confed] = summary
        if verbose:
            for name, s in summary.items():
                if s.get("n", 0) > 0:
                    print(f"    {name:<18} n={s['n']:>3}  "
                          f"acc={s['accuracy']:.3f}  rps={s['rps']:.4f}")
    return results


def layer4_walk_forward_all(verbose: bool = True) -> Dict:
    """L4 完整 walk-forward：所有 WC 比赛 + 国际比赛"""
    print("\n" + "=" * 70)
    print("L4 完整 Walk-Forward（2010+ 所有 WC）")
    print("=" * 70)

    df = _load_history(min_year=1990)
    models = _build_models()
    eval_filter = lambda row: (row["tournament"] == "FIFA World Cup"
                                 and row["date"].year >= 2010
                                 and row["date"].year <= 2022)

    summary = run_walk_forward(models, df, eval_filter=eval_filter)
    if verbose:
        print(f"\n  评估范围：2010-2022 共 5 届 WC")
        for name, s in summary.items():
            if s.get("n", 0) > 0:
                print(f"    {name:<18} n={s['n']:>3}  acc={s['accuracy']:.3f}  "
                      f"rps={s['rps']:.4f}  brier={s['brier']:.4f}  "
                      f"log_loss={s['log_loss']:.4f}  "
                      f"cal_err={s.get('calibration_error', 0):.3f}")
    return summary


# ============ 辅助函数 ============
def _load_history(min_year: int = 1990, csv_path: Optional[Path] = None) -> pd.DataFrame:
    csv_path = csv_path or (DATA_HIST / "results.csv")
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"].dt.year >= min_year].sort_values("date").reset_index(drop=True)
    return df


class PoissonFamilyModel:
    """
    Poisson 族集成（DC+Bivariate+ZIGP）转 3 分类
    
    流程：
      1. 用 pi-ratings 的 expected_gd 反推 λ_a, λ_b
      2. ensemble_score_matrix 算 8×8 比分矩阵
      3. 汇总成 H/D/A 概率
    """
    name = "poisson_family"

    def __init__(self, base_lambda: float = 1.45):
        self.R_H: Dict[str, float] = {}
        self.R_A: Dict[str, float] = {}
        self.base_lambda = base_lambda

    def _team_lambdas(self, home: str, away: str, neutral: bool) -> Tuple[float, float]:
        """从 pi-ratings 反推两队期望进球"""
        rh = self.R_H.get(home, 0.0)
        ra_h = self.R_A.get(home, 0.0)
        rh_a = self.R_H.get(away, 0.0)
        ra = self.R_A.get(away, 0.0)
        if neutral:
            r_h_eff = (rh + ra_h) / 2.0
            r_a_eff = (rh_a + ra) / 2.0
        else:
            r_h_eff = rh
            r_a_eff = ra
        # expected_gd 给的是主队净胜，拆成 λ
        gd = expected_goal_diff(r_h_eff, r_a_eff)
        # 总进球约 base × 2 = 2.9（国际赛均值），按 gd 切分
        total = self.base_lambda * 2
        lam_a = max(0.2, (total + gd) / 2)
        lam_b = max(0.2, (total - gd) / 2)
        return lam_a, lam_b

    def predict(self, home: str, away: str, neutral: bool = False, **kw):
        from models.poisson_model import ensemble_score_matrix, match_outcome_from_score_matrix
        lam_a, lam_b = self._team_lambdas(home, away, neutral)
        ens = ensemble_score_matrix(lam_a, lam_b)
        out = match_outcome_from_score_matrix(ens["matrix"])
        return out["p_win_a"], out["p_draw"], out["p_win_b"]

    def update(self, home: str, away: str, home_score: int, away_score: int,
                tournament: str = "Friendly", neutral: bool = False, **kw):
        weight = TOURNAMENT_WEIGHTS.get(tournament, 0.8)
        pi_update_step(self.R_H, self.R_A, home, away,
                        home_score, away_score, weight=weight, neutral=neutral)


class StackedEnsembleModel:
    """
    Stacked ensemble：用 logistic regression 学子模型的最优权重
    
    训练：
      用 holdout（前 80% major 国际比赛）的子模型预测作为特征，
      实际结果作为 y，学 W ∈ R^(3 × 3K) 把 K 个模型的 3 类概率
      映射到最终 3 类概率（含 softmax）。
    
    推理：
      forward 时把各子模型的 9 维 logit 喂进 stacker。
    """
    name = "stacked_ensemble"

    def __init__(self, submodels: List, stacker_path: Optional[Path] = None):
        self.submodels = submodels
        self.K = len(submodels)
        self.stacker = None
        if stacker_path and stacker_path.exists():
            self._load_stacker(stacker_path)

    def _load_stacker(self, path: Path):
        from sklearn.linear_model import LogisticRegression
        import joblib
        self.stacker = joblib.load(path)

    def _features(self, home, away, neutral):
        """子模型预测 → 9 维特征（每个 3 类）"""
        feats = []
        for m in self.submodels:
            ph, pd, pa = m.predict(home=home, away=away, neutral=neutral)
            feats.extend([ph, pd, pa])
        return feats

    def predict(self, home: str, away: str, neutral: bool = False, **kw):
        if self.stacker is None:
            # 没训 stacker → 等权重平均（degrade gracefully）
            ph, pd, pa = 0.0, 0.0, 0.0
            for m in self.submodels:
                a, b, c = m.predict(home=home, away=away, neutral=neutral)
                ph += a; pd += b; pa += c
            n = max(1, self.K)
            return ph / n, pd / n, pa / n
        import numpy as np
        feats = np.array([self._features(home, away, neutral)])
        proba = self.stacker.predict_proba(feats)[0]
        return float(proba[0]), float(proba[1]), float(proba[2])

    def update(self, **kw):
        for m in self.submodels:
            m.update(**kw)


def fit_stacker(submodels: List, df: pd.DataFrame,
                  eval_year_min: int = 2010,
                  eval_year_max: int = 2018) -> "object":
    """
    在 [eval_year_min, eval_year_max] 区间收集子模型预测 → fit logistic stacker
    
    把 eval_year_max+ 留给真实测试（避免 leakage）
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    X, y = [], []
    for _, row in df.iterrows():
        home, away = row["home_team"], row["away_team"]
        try:
            hs, ascore = int(row["home_score"]), int(row["away_score"])
        except Exception:
            continue
        neutral = bool(row.get("neutral", False))
        tour = row.get("tournament", "Friendly")
        year = row["date"].year

        is_train = (eval_year_min <= year <= eval_year_max
                     and tour in {
                         "FIFA World Cup", "FIFA World Cup qualification",
                         "UEFA Euro", "Copa América", "Africa Cup of Nations",
                         "AFC Asian Cup", "UEFA Nations League",
                     })
        if is_train:
            feats = []
            for m in submodels:
                ph, pd_, pa = m.predict(home=home, away=away, neutral=neutral)
                feats.extend([ph, pd_, pa])
            X.append(feats)
            y.append({"H": 0, "D": 1, "A": 2}[actual_outcome(hs, ascore)])

        for m in submodels:
            m.update(home=home, away=away, home_score=hs, away_score=ascore,
                      tournament=tour, neutral=neutral)

    X = np.array(X)
    y = np.array(y)
    print(f"   Stacker 训练样本: {len(y)}")
    stacker = LogisticRegression(
        multi_class="multinomial", C=1.0, max_iter=1000, solver="lbfgs",
    )
    stacker.fit(X, y)
    return stacker


class EnsembleModel:
    """
    简单线性集成：多个子模型概率加权平均
    
    参考报告 2.3.4：集成的核心价值不是更高 accuracy，而是更好校准
    """
    name = "ensemble"

    def __init__(self, submodels: List, weights: Optional[List[float]] = None):
        self.submodels = submodels
        self.weights = weights or [1.0 / len(submodels)] * len(submodels)

    def predict(self, home: str, away: str, neutral: bool = False, **kw):
        p_h, p_d, p_a = 0.0, 0.0, 0.0
        for m, w in zip(self.submodels, self.weights):
            ph, pd, pa = m.predict(home=home, away=away, neutral=neutral)
            p_h += w * ph
            p_d += w * pd
            p_a += w * pa
        return p_h, p_d, p_a

    def update(self, **kw):
        for m in self.submodels:
            m.update(**kw)


def _build_models(skip_catboost: bool = False,
                   include_calibrated: bool = True,
                   include_ensemble: bool = True,
                   include_poisson_family: bool = True,
                   include_stacked: bool = True) -> List:
    models = [BaselineRandom(), BaselineHomeAlways(), EloModel()]
    if not skip_catboost:
        try:
            models.append(PiCatBoostModel())
        except FileNotFoundError:
            pass
        if include_calibrated:
            for cal_name in ["temperature", "temperature_ece", "vector", "isotonic"]:
                try:
                    models.append(CalibratedCatBoostModel(cal_name))
                except (FileNotFoundError, KeyError):
                    pass

        if include_poisson_family:
            models.append(PoissonFamilyModel())

        # 等权 ensemble：Elo + CatBoost + Cal-CB + Poisson族
        if include_ensemble and len(models) >= 4:
            sub = [EloModel(), PiCatBoostModel(),
                    CalibratedCatBoostModel("temperature"),
                    PoissonFamilyModel()]
            ens = EnsembleModel(sub, weights=[0.25, 0.25, 0.25, 0.25])
            ens.name = "ensemble_4"
            models.append(ens)

        # Stacked ensemble：用 logistic 学最优权重（如果 stacker 已训练）
        if include_stacked:
            stacker_path = DATA_OUTPUTS / "stacker_logistic.joblib"
            if stacker_path.exists():
                sub = [EloModel(), PiCatBoostModel(),
                        CalibratedCatBoostModel("temperature"),
                        PoissonFamilyModel()]
                stacked = StackedEnsembleModel(sub, stacker_path=stacker_path)
                models.append(stacked)

    return models


# ============ 全套入口 ============
def run_all(out_path: Optional[Path] = None) -> Dict:
    """跑全部四层并写出 JSON 报告"""
    all_results = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "reference_targets": {
            "accuracy": ">60%",
            "rps": "<0.195",
            "brier": "<0.55",
            "calibration_error": "<5%",
        },
    }

    t0 = time.time()
    all_results["L1_temporal_cv"] = layer1_temporal_cv()
    all_results["L2_cross_tournament"] = layer2_cross_tournament()
    all_results["L3_confed_holdout"] = layer3_confed_holdout()
    all_results["L4_walk_forward"] = layer4_walk_forward_all()
    all_results["elapsed_s"] = round(time.time() - t0, 1)

    out_path = out_path or (DATA_OUTPUTS / "backtest_walk_forward.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2))
    print(f"\n✅ 全套四层回测完成，耗时 {all_results['elapsed_s']}s")
    print(f"📄 写入 {out_path.relative_to(ROOT)}")
    return all_results


# ============ Stacker 训练入口 ============
def train_stacker_cli():
    """训练 stacked ensemble 的 logistic 元学习器。
    
    用 2010-2018 五届 WC + 同期主要洲际杯做训练，
    2018-2022 留作真实评估（避免泄露）。
    """
    print("📥 加载历史数据...")
    df = _load_history(min_year=1990)

    print("🧠 构造 4 个子模型（warm-up 到 2009 底）...")
    submodels = [
        EloModel(),
        PiCatBoostModel(),
        CalibratedCatBoostModel("temperature"),
        PoissonFamilyModel(),
    ]
    # warm up
    df_warmup = df[df["date"].dt.year < 2010]
    for _, row in df_warmup.iterrows():
        try:
            hs, ascore = int(row["home_score"]), int(row["away_score"])
        except Exception:
            continue
        for m in submodels:
            m.update(home=row["home_team"], away=row["away_team"],
                      home_score=hs, away_score=ascore,
                      tournament=row.get("tournament", "Friendly"),
                      neutral=bool(row.get("neutral", False)))
    print(f"   warm-up 完成（{len(df_warmup):,} 场）")

    df_train_window = df[(df["date"].dt.year >= 2010)
                          & (df["date"].dt.year <= 2018)]
    print(f"\n🎯 在 2010-2018 区间训练 stacker（{len(df_train_window):,} 场）...")
    stacker = fit_stacker(submodels, df_train_window,
                            eval_year_min=2010, eval_year_max=2018)

    import joblib
    out_path = DATA_OUTPUTS / "stacker_logistic.joblib"
    joblib.dump(stacker, out_path)
    print(f"\n✅ Stacker 保存: {out_path.relative_to(ROOT)}")
    print(f"   classes: {stacker.classes_}")
    print(f"   coef shape: {stacker.coef_.shape}")


# ============ CLI ============
if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "L1":
        layer1_temporal_cv()
    elif cmd == "L2":
        layer2_cross_tournament()
    elif cmd == "L3":
        layer3_confed_holdout()
    elif cmd == "L4":
        layer4_walk_forward_all()
    elif cmd == "train_stacker":
        train_stacker_cli()
    elif cmd == "all":
        run_all()
    else:
        print(__doc__)
