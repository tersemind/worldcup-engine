"""
分化信号离线验证（standalone，零生产影响）
==========================================
目标：验证两个"根因型"信号能否降低历史 WC 预测误差——**绝不向市场收敛**，
只补 ELO 自身的结构盲区。对应用户提出的两类分化：

  1. confederation 强度盲区（USA/Australia 类）
     ELO 把"在 CONCACAF/AFC 刷分"和"在 UEFA/CONMEBOL 刷分"等价对待。
     历史上跨洲 WC 对决里 AFC/CONCACAF 系统性低于其 ELO 预期。
     → 从历史"跨洲 WC 实际结果 vs ELO 预期"反推每洲 ELO 偏移，walk-forward 注入。
     → 支持 recency-decay：近期跨洲结果权重更高，自动反映"洲际差距随时间收敛"。

  2. motivation / dead-rubber（晋级形势盲区）
     小组赛末轮已出线队轮换、必胜队超水平发挥。ELO 完全看不到。
     → 由独立脚本（signal 2）验证，本文件聚焦信号 1。

方法学（严格无泄漏）：
  - 按时间顺序 walk-forward 累积 ELO（与 walk_forward.EloModel 同款 K/update）
  - **每届 WC 在其首场比赛前冻结一次 confed 偏移快照**，整届复用该快照
    → 该届自身比赛绝不进入其偏移估计，杜绝赛中泄漏
  - 评估集：2010-2022 共 4 届 WC 的所有比赛（results.csv 截至 2022）
  - 指标：RPS / Brier / log-loss / accuracy（复用 walk_forward 的实现）
  - 切片：全部 / 跨洲 / 含 AFC-CONCACAF 跨洲（USA/Australia 类）
  - 每届分解 + 超参敏感性扫描（证明非过拟合到单一超参或单一届）

输出：data/outputs/divergence_validation.json
运行：python3 code/backtest/validate_divergence_signals.py
"""
from __future__ import annotations
import sys
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.elo_engine import match_probabilities, update_elo
from backtest.walk_forward import (
    get_confed, rps, brier_multi, log_loss, predicted_class, _load_history,
)
from utils.io import ROOT

DATA_OUTPUTS = ROOT / "data" / "outputs"
EVAL_YEARS = {2010, 2014, 2018, 2022}   # 评估用 WC（results.csv 截至 2022）
HOME_ADV_ELO = 60.0                     # 与 EloModel 一致


# ---------------------------------------------------------------------------
# ELO walk-forward 状态（与 walk_forward.EloModel 同款 K/update，独立持有以免污染）
# ---------------------------------------------------------------------------
class EloState:
    def __init__(self, default: float = 1500.0):
        self.elo: Dict[str, float] = {}
        self.default = default

    def get(self, team: str) -> float:
        return self.elo.get(team, self.default)

    def update_match(self, home: str, away: str, hs: int, a_s: int, tournament: str):
        result = 1.0 if hs > a_s else (0.0 if hs < a_s else 0.5)
        if "World Cup" in tournament and "qualification" not in tournament:
            k = 60.0
        elif "Euro" in tournament:
            k = 40.0
        else:
            k = 30.0
        nh, na = update_elo(self.get(home), self.get(away), result, k=k)
        self.elo[home] = nh
        self.elo[away] = na


# ---------------------------------------------------------------------------
# 信号 1：confederation 偏移（只用历史跨洲 WC 样本，无泄漏）
# ---------------------------------------------------------------------------
class ConfedOffsetEstimator:
    """
    增量累积每洲的"WC 实际表现 - ELO 预期"残差，转成 ELO 偏移。

    对一场跨洲 WC 比赛 (home=confed_X, away=confed_Y)：
      expected_home = ELO 期望胜率（含中立场处理）
      actual_home   = 1/0.5/0
      residual = actual - expected  → 归到 confed_X(+), confed_Y(-)

    decay < 1.0 时启用**指数衰减**：每次新观测前先把已有累计乘以 decay，
    使近期跨洲结果权重更高 → 自动反映"洲际差距随时间收敛"（解决 2022 上翘）。
    decay = 1.0 退化为等权长期均值。
    """
    def __init__(self, elo_per_winrate: float = 600.0, scale: float = 1.0,
                 decay: float = 1.0):
        # 累计：每洲的 (Σweighted_residual, Σweight)
        self.acc: Dict[str, List[float]] = defaultdict(lambda: [0.0, 0.0])
        self.elo_per_winrate = elo_per_winrate
        self.scale = scale
        self.decay = decay

    def observe(self, confed_h: str, confed_a: str,
                exp_home: float, actual_home: float):
        if confed_h == confed_a:
            return  # 只学跨洲信号
        res = actual_home - exp_home
        if self.decay < 1.0:
            # 衰减全体已有累计（近似按"观测序"指数衰减；跨洲 WC 样本稀疏，足够平滑）
            for c in self.acc:
                self.acc[c][0] *= self.decay
                self.acc[c][1] *= self.decay
        self.acc[confed_h][0] += res
        self.acc[confed_h][1] += 1
        self.acc[confed_a][0] += (-res)
        self.acc[confed_a][1] += 1

    def offset_elo(self, confed: str, min_n: float = 8) -> float:
        s, n = self.acc.get(confed, [0.0, 0.0])
        if n < min_n:
            return 0.0
        mean_res = s / n                       # 加权平均胜率残差（+ = 强于 ELO 预期）
        # 胜率残差 → 等价 ELO：用 expected_score 的局部斜率近似（s=600 时 1 胜率≈600 ELO）
        return mean_res * self.elo_per_winrate * self.scale


# ---------------------------------------------------------------------------
# 预测：baseline vs treatment
# ---------------------------------------------------------------------------
def predict_baseline(elo: EloState, home: str, away: str, neutral: bool) -> Tuple[float, float, float]:
    e_h = elo.get(home) + (0 if neutral else HOME_ADV_ELO)
    e_a = elo.get(away)
    p = match_probabilities(e_h, e_a)
    return p["p_win_a"], p["p_draw"], p["p_win_b"]


def predict_treatment(elo: EloState, frozen_off: Dict[str, float],
                      home: str, away: str, neutral: bool,
                      confed_cap: float = 80.0) -> Tuple[float, float, float]:
    """用**冻结的**每洲偏移（赛前快照，杜绝赛中泄漏）预测。"""
    off_h = max(-confed_cap, min(confed_cap, frozen_off.get(get_confed(home), 0.0)))
    off_a = max(-confed_cap, min(confed_cap, frozen_off.get(get_confed(away), 0.0)))
    e_h = elo.get(home) + off_h + (0 if neutral else HOME_ADV_ELO)
    e_a = elo.get(away) + off_a
    p = match_probabilities(e_h, e_a)
    return p["p_win_a"], p["p_draw"], p["p_win_b"]


# ---------------------------------------------------------------------------
# 指标累加器
# ---------------------------------------------------------------------------
class Metrics:
    def __init__(self):
        self.n = 0
        self.rps = 0.0
        self.brier = 0.0
        self.ll = 0.0
        self.correct = 0

    def add(self, p_h, p_d, p_a, actual):
        self.n += 1
        self.rps += rps(p_h, p_d, p_a, actual)
        self.brier += brier_multi(p_h, p_d, p_a, actual)
        self.ll += log_loss(p_h, p_d, p_a, actual)
        if predicted_class(p_h, p_d, p_a) == actual:
            self.correct += 1

    def summary(self) -> Dict:
        if self.n == 0:
            return {"n": 0}
        return {
            "n": self.n,
            "rps": round(self.rps / self.n, 4),
            "brier": round(self.brier / self.n, 4),
            "log_loss": round(self.ll / self.n, 4),
            "accuracy": round(self.correct / self.n, 4),
        }


def _actual(hs: int, a_s: int) -> str:
    return "H" if hs > a_s else ("A" if hs < a_s else "D")


# ---------------------------------------------------------------------------
# 主流程：单次 walk-forward（给定超参），返回各切片 + 各届指标
# ---------------------------------------------------------------------------
def _walk_forward(df: pd.DataFrame, scale: float = 1.0,
                  confed_cap: float = 80.0, min_n: float = 8,
                  decay: float = 1.0) -> Dict:
    elo = EloState()
    est = ConfedOffsetEstimator(scale=scale, decay=decay)

    WEAK_CONFEDS = {"AFC", "CONCACAF"}
    slices = ("all", "cross", "afc_concacaf")
    m_base = {s: Metrics() for s in slices}
    m_treat = {s: Metrics() for s in slices}
    m_base_year = {y: Metrics() for y in EVAL_YEARS}
    m_treat_year = {y: Metrics() for y in EVAL_YEARS}

    # 每届的冻结偏移快照：在该届首场 WC 比赛出现前，用"此前所有历史"冻结一次
    frozen: Dict[int, Dict[str, float]] = {}
    offsets_snapshot: Dict[int, Dict[str, float]] = {}

    for _, row in df.iterrows():
        home, away = row["home_team"], row["away_team"]
        if pd.isna(row["home_score"]) or pd.isna(row["away_score"]):
            continue
        hs, a_s = int(row["home_score"]), int(row["away_score"])
        tournament = row.get("tournament", "Friendly")
        neutral = bool(row.get("neutral", False))
        year = row["date"].year
        is_wc = ("World Cup" in tournament) and ("qualification" not in tournament)
        is_eval = is_wc and year in EVAL_YEARS

        confed_h, confed_a = get_confed(home), get_confed(away)
        known = confed_h != "OTHER" and confed_a != "OTHER"
        cross = (confed_h != confed_a) and known
        actual = _actual(hs, a_s)

        # ===== 评估前：若该届尚未冻结偏移，则用"截至目前"的 est 冻结一次 =====
        # 冻结发生在处理该届任何一场之前 → 该届自身比赛不进入其偏移估计 → 无赛中泄漏
        if is_eval and year not in frozen:
            frozen[year] = {c: est.offset_elo(c, min_n=min_n) for c in est.acc}
            offsets_snapshot[year] = {c: round(v, 1) for c, v in frozen[year].items()}

        # ===== 评估（预测在 update 之前；偏移用冻结快照）=====
        if is_eval and known:
            fo = frozen[year]
            bh, bd, ba = predict_baseline(elo, home, away, neutral)
            th, td, ta = predict_treatment(elo, fo, home, away, neutral, confed_cap)

            m_base["all"].add(bh, bd, ba, actual)
            m_treat["all"].add(th, td, ta, actual)
            if cross:
                m_base["cross"].add(bh, bd, ba, actual)
                m_treat["cross"].add(th, td, ta, actual)
                m_base_year[year].add(bh, bd, ba, actual)
                m_treat_year[year].add(th, td, ta, actual)
                if WEAK_CONFEDS & {confed_h, confed_a}:
                    m_base["afc_concacaf"].add(bh, bd, ba, actual)
                    m_treat["afc_concacaf"].add(th, td, ta, actual)

        # ===== 学习 confed 偏移（用预测前 ELO 期望）=====
        # 评估届自身比赛也会更新 est，但只影响**更晚的届**的冻结快照，
        # 不影响当前届（当前届已冻结），故对当前届评估仍无泄漏。
        if is_wc and cross:
            e_h = elo.get(home) + (0 if neutral else HOME_ADV_ELO)
            exp_home = match_probabilities(e_h, elo.get(away))["p_win_a"]
            actual_home_wr = 1.0 if actual == "H" else (0.5 if actual == "D" else 0.0)
            est.observe(confed_h, confed_a, exp_home, actual_home_wr)

        elo.update_match(home, away, hs, a_s, tournament)

    def _delta(b, t):
        d = {}
        if b.get("n", 0) > 0 and t.get("n", 0) > 0:
            for key in ("rps", "brier", "log_loss", "accuracy"):
                d[key] = round(t[key] - b[key], 4)
        return d

    out = {
        "params": {"scale": scale, "confed_cap": confed_cap, "min_n": min_n,
                   "decay": decay},
        "offsets_frozen_per_year": offsets_snapshot,
        "slices": {},
        "per_year_cross": {},
    }
    for s in slices:
        b, t = m_base[s].summary(), m_treat[s].summary()
        out["slices"][s] = {"baseline": b, "treatment": t, "delta": _delta(b, t)}
    for y in sorted(EVAL_YEARS):
        b, t = m_base_year[y].summary(), m_treat_year[y].summary()
        out["per_year_cross"][y] = {"baseline": b, "treatment": t, "delta": _delta(b, t)}
    return out


def run(min_year: int = 1990, verbose: bool = True,
        sweep: bool = True, decay: float = 1.0) -> Dict:
    df = _load_history(min_year=min_year)

    # 主结果：默认超参（无赛中泄漏 + 每届冻结 + 每届分解）
    main = _walk_forward(df, scale=1.0, confed_cap=80.0, min_n=8, decay=decay)

    out = {
        "_method": ("confed-offset walk-forward · 无市场 · 无泄漏（每届赛前冻结偏移）· "
                    "eval=WC 2010-2022 · 含每届分解 + 超参敏感性扫描"),
        "_eval_years": sorted(EVAL_YEARS),
        "main": main,
    }

    # 敏感性扫描：证明增益不是"调出来的运气点"，并捕捉每届表现
    if sweep:
        grid = []
        for sc in (0.5, 0.75, 1.0, 1.25, 1.5):
            for cap in (40.0, 80.0, 120.0):
                for dc in (1.0, 0.9):
                    r = _walk_forward(df, scale=sc, confed_cap=cap, min_n=8, decay=dc)
                    cross = r["slices"]["cross"]
                    rec = {
                        "scale": sc, "cap": cap, "decay": dc,
                        "cross_rps_delta": cross["delta"].get("rps"),
                        "cross_brier_delta": cross["delta"].get("brier"),
                        "cross_acc_delta": cross["delta"].get("accuracy"),
                    }
                    for y in sorted(EVAL_YEARS):
                        rec[f"rps_delta_{y}"] = r["per_year_cross"][y]["delta"].get("rps")
                    grid.append(rec)
        out["sensitivity_sweep"] = grid

    DATA_OUTPUTS.mkdir(parents=True, exist_ok=True)
    (DATA_OUTPUTS / "divergence_validation.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2)
    )

    if verbose:
        _print_report(out)
    return out


def _print_report(out: Dict):
    main = out["main"]
    print("\n" + "=" * 82)
    print("分化信号离线验证 · confederation 偏移（无市场 · 每届赛前冻结偏移 · 无泄漏）")
    print("=" * 82)

    snap = main["offsets_frozen_per_year"]
    if snap:
        last_year = max(snap.keys(), key=lambda y: int(y))
        print(f"\n冻结于 {last_year} 届赛前的各洲 ELO 偏移（+ 强于 ELO 预期 / - 弱于）：")
        for c, v in sorted(snap[last_year].items(), key=lambda x: -x[1]):
            print(f"  {c:<10} {v:+6.1f} ELO")

    labels = {
        "all": "全部 WC 评估场",
        "cross": "跨洲场次",
        "afc_concacaf": "含 AFC/CONCACAF 跨洲（USA/Australia 类）",
    }
    for slc, label in labels.items():
        s = main["slices"][slc]
        b, t, d = s["baseline"], s["treatment"], s["delta"]
        if b.get("n", 0) == 0 or not d:
            continue
        print(f"\n── {label}  (n={b['n']}) ──")
        print(f"{'指标':<10}{'baseline':>12}{'treatment':>12}{'Δ':>10}  (rps/brier/ll 负Δ=改善)")
        for key in ("rps", "brier", "log_loss", "accuracy"):
            arrow = "✅" if ((key == "accuracy" and d[key] > 0) or
                            (key != "accuracy" and d[key] < 0)) else \
                    ("➖" if d[key] == 0 else "⚠️")
            print(f"{key:<10}{b[key]:>12}{t[key]:>12}{d[key]:>+10}  {arrow}")

    print("\n── 每届 WC 分解（跨洲切片，RPS Δ / 准确率 Δ）──")
    print(f"{'届':<8}{'n':>5}{'RPS Δ':>12}{'Acc Δ':>12}")
    for y in sorted(main["per_year_cross"].keys(), key=lambda x: int(x)):
        pe = main["per_year_cross"][y]
        b, d = pe["baseline"], pe["delta"]
        if b.get("n", 0) == 0 or not d:
            continue
        rps_arrow = "✅" if d["rps"] < 0 else ("➖" if d["rps"] == 0 else "⚠️")
        print(f"{y:<8}{b['n']:>5}{d['rps']:>+12}{d['accuracy']:>+12}  {rps_arrow}")

    if "sensitivity_sweep" in out:
        print("\n── 超参敏感性扫描（跨洲切片 RPS Δ；全为负=对超参稳健）──")
        print(f"{'scale':>7}{'cap':>7}{'decay':>7}{'RPS Δ':>11}{'2022 Δ':>10}")
        all_neg = True
        for g in out["sensitivity_sweep"]:
            if g["cross_rps_delta"] is None:
                continue
            if g["cross_rps_delta"] >= 0:
                all_neg = False
            print(f"{g['scale']:>7}{g['cap']:>7.0f}{g['decay']:>7}"
                  f"{g['cross_rps_delta']:>+11}{g['rps_delta_2022']:>+10}")
        print(f"\n  → 所有超参组合 RPS 均改善: {'是 ✅' if all_neg else '否 ⚠️'}")

    print("\n输出已写入 data/outputs/divergence_validation.json")
    print("注：motivation/dead-rubber 信号需小组积分流，单独脚本验证（signal 2）。")


if __name__ == "__main__":
    run()
