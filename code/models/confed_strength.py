"""
Confederation 强度偏移引擎（生产模块）
=======================================
解决"模型 vs 市场分化"的根因之一：ELO 把不同大洲的刷分等价对待，
但历史跨洲 WC 对决证明 AFC/CONCACAF/CAF 系统性低于 ELO 预期。

本模块从 data/historical/results.csv **运行时动态计算**每洲 ELO 偏移
（不硬编码，随历史更新自动演进），供 ai_confed 通道在 _quick_match_preview 注入。

验证依据：code/backtest/validate_divergence_signals.py
  - 跨洲切片 RPS −0.0063 / Brier −0.0132（WC 2010-2022 walk-forward，无泄漏）
  - AFC/CONCACAF 切片增益最大；30 组超参池化全改善
  - 已知边界：2022 爆冷年轻微劣化，靠 cap=80 限幅（信号内禀，已接受）

⚠ 任何失败都返回空偏移（全 0），绝不破坏主预测流程。
缓存：results.csv mtime 不变则复用，避免每次预测重算。
"""
from __future__ import annotations
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.io import ROOT

DATA_HIST = ROOT / "data" / "historical"

# 与验证脚本一致的超参（cap=80 是已接受的安全限幅）
ELO_PER_WINRATE = 600.0
CONFED_CAP = 80.0
MIN_N = 8
HOME_ADV_ELO = 60.0

# 与 backtest/walk_forward.py 同款大洲映射（运行时从该模块导入，避免重复维护）
_CONFED_CACHE: Dict[str, object] = {"offsets": None, "mtime": 0.0}


def _confed_of(team: str) -> str:
    try:
        from backtest.walk_forward import get_confed
        return get_confed(team)
    except Exception:
        return "OTHER"


def _compute_offsets() -> Dict[str, float]:
    """从 results.csv 全历史跨洲 WC 比赛反推每洲 ELO 偏移。失败返 {}。"""
    try:
        import pandas as pd
        from models.elo_engine import match_probabilities, update_elo

        csv = DATA_HIST / "results.csv"
        df = pd.read_csv(csv)
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"].dt.year >= 1990].sort_values("date")
        df = df[~(df["home_score"].isna() | df["away_score"].isna())]

        elo: Dict[str, float] = {}

        def g(t):
            return elo.get(t, 1500.0)

        acc = defaultdict(lambda: [0.0, 0.0])  # confed -> [Σresid, n]
        for _, r in df.iterrows():
            h, a = r["home_team"], r["away_team"]
            hs, a_s = r["home_score"], r["away_score"]
            tour = r.get("tournament", "Friendly")
            neutral = bool(r.get("neutral", False))
            is_wc = ("World Cup" in tour) and ("qualification" not in tour)
            ch, ca = _confed_of(h), _confed_of(a)

            if is_wc and ch != ca and "OTHER" not in (ch, ca):
                e_h = g(h) + (0 if neutral else HOME_ADV_ELO)
                exp_h = match_probabilities(e_h, g(a))["p_win_a"]
                act_h = 1.0 if hs > a_s else (0.5 if hs == a_s else 0.0)
                res = act_h - exp_h
                acc[ch][0] += res
                acc[ch][1] += 1
                acc[ca][0] += (-res)
                acc[ca][1] += 1

            # walk-forward 更新 ELO
            result = 1.0 if hs > a_s else (0.0 if hs < a_s else 0.5)
            if is_wc:
                k = 60.0
            elif "Euro" in tour:
                k = 40.0
            else:
                k = 30.0
            nh, na = update_elo(g(h), g(a), result, k=k)
            elo[h] = nh
            elo[a] = na

        offsets = {}
        for confed, (s, n) in acc.items():
            if n >= MIN_N:
                raw = (s / n) * ELO_PER_WINRATE
                offsets[confed] = max(-CONFED_CAP, min(CONFED_CAP, raw))
        return offsets
    except Exception:
        return {}


def get_confed_offsets() -> Dict[str, float]:
    """返回 {confed: elo_offset}，带 results.csv mtime 缓存。失败返 {}。"""
    try:
        csv = DATA_HIST / "results.csv"
        if not csv.exists():
            return {}
        mtime = csv.stat().st_mtime
        if _CONFED_CACHE["offsets"] is not None and _CONFED_CACHE["mtime"] == mtime:
            return _CONFED_CACHE["offsets"]  # type: ignore
        offsets = _compute_offsets()
        _CONFED_CACHE["offsets"] = offsets
        _CONFED_CACHE["mtime"] = mtime
        return offsets
    except Exception:
        return {}


def get_team_confed_offset(team: str) -> float:
    """单队 ELO 偏移（已限幅）。未知/失败返 0.0。"""
    return get_confed_offsets().get(_confed_of(team), 0.0)


if __name__ == "__main__":
    offs = get_confed_offsets()
    print("=== Confederation ELO 偏移（运行时从 results.csv 计算，cap ±80）===")
    for c, v in sorted(offs.items(), key=lambda x: -x[1]):
        print(f"  {c:<10} {v:+6.1f} ELO")
    if not offs:
        print("  （空——计算失败或数据缺失，预测将退化为无偏移）")
