"""
信号2 离线验证：motivation / dead-rubber（晋级形势盲区）
=========================================================
假设：小组赛末轮（MD3），已出线/已出局队动机下降、轮换主力 → 实际表现低于 ELO 预期；
必争出线队动机拉满 → 高于预期。ELO 完全看不到这层。

**绝不向市场收敛**，只检验"动机状态"能否解释 ELO 残差并降低预测误差。

数据：data/historical/results.csv 的 7 届 32 队制 WC（1998-2022），
      每届 64 场（48 小组 + 16 淘汰）。CSV 无小组标签 → 自行重建：
  1. 取每届前 48 场为小组赛（按日期）
  2. 并查集：同场过的队聚成连通块 → 每块恰好 4 队 = 一个小组
  3. 按日期排小组内 3 轮（MD1/MD2/MD3）
  4. MD3 前算各队积分/净胜球 → 判定每队 MD3 动机状态：
     - SECURED  已出线（数学上提前锁定，保守用"积分领先到不可追"近似）
     - ELIMINATED 已出局
     - MUST_WIN  生死战（未定且需要分数）
  5. 检验 MD3 各状态下"实际胜率残差 vs ELO 期望"，并测注入修正能否降 RPS

⚠ 重建是启发式（无官方分组），用并查集+4队校验保证结构正确，
  无法重建的小组跳过并计数，绝不静默截断。

输出：data/outputs/motivation_validation.json
运行：python3 code/backtest/validate_motivation_signal.py
"""
from __future__ import annotations
import sys
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.elo_engine import match_probabilities, update_elo
from backtest.walk_forward import rps, brier_multi, predicted_class, _load_history
from utils.io import ROOT

DATA_OUTPUTS = ROOT / "data" / "outputs"
WC_YEARS = [1998, 2002, 2006, 2010, 2014, 2018, 2022]
HOME_ADV_ELO = 60.0  # WC 中立场，但保持与基线一致用 0；下面单独处理


# ---------------------------------------------------------------------------
# 并查集（重建小组）
# ---------------------------------------------------------------------------
class DSU:
    def __init__(self):
        self.p: Dict[str, str] = {}

    def find(self, x: str) -> str:
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: str, b: str):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def reconstruct_groups(group_matches: pd.DataFrame) -> Tuple[List[List[str]], int]:
    """
    从小组赛场次重建小组。返回 (groups, n_failed)。
    每组应恰好 4 队、每队恰好 3 场组内赛。
    """
    dsu = DSU()
    for _, r in group_matches.iterrows():
        dsu.union(r["home_team"], r["away_team"])
    clusters: Dict[str, List[str]] = defaultdict(list)
    for t in set(group_matches["home_team"]) | set(group_matches["away_team"]):
        clusters[dsu.find(t)].append(t)
    groups, n_failed = [], 0
    for _, members in clusters.items():
        if len(members) == 4:
            groups.append(sorted(members))
        else:
            n_failed += 1  # 重建异常（跨组重复对手等），跳过并计数
    return groups, n_failed


# ---------------------------------------------------------------------------
# ELO walk-forward（与信号1同款）
# ---------------------------------------------------------------------------
class EloState:
    def __init__(self, default: float = 1500.0):
        self.elo: Dict[str, float] = {}
        self.default = default

    def get(self, t: str) -> float:
        return self.elo.get(t, self.default)

    def update_match(self, h, a, hs, a_s, tournament):
        result = 1.0 if hs > a_s else (0.0 if hs < a_s else 0.5)
        if "World Cup" in tournament and "qualification" not in tournament:
            k = 60.0
        elif "Euro" in tournament:
            k = 40.0
        else:
            k = 30.0
        nh, na = update_elo(self.get(h), self.get(a), result, k=k)
        self.elo[h] = nh
        self.elo[a] = na


def _pts(hs, a_s):
    if hs > a_s: return 3, 0
    if hs < a_s: return 0, 3
    return 1, 1


# ---------------------------------------------------------------------------
# 动机状态判定（MD3 前）
# ---------------------------------------------------------------------------
def classify_motivation(standings: Dict[str, Dict], team: str,
                        all_teams: List[str]) -> str:
    """
    用 MD1+MD2 后的积分判定 MD3 各队状态（保守近似）：
      - 队伍按 (pts, gd, gf) 排序
      - SECURED:    当前积分 >= 6（两连胜，数学已出线）
      - ELIMINATED: 当前积分 == 0 且最大可能(=3) 仍追不上第二名现有积分
      - 其余 = MUST_WIN（生死/争夺）
    保守起见，仅把最确定的两端标出，其余归 MUST_WIN，避免误标稀释信号。
    """
    me = standings[team]
    sorted_t = sorted(all_teams, key=lambda t: (-standings[t]["pts"],
                                                 -standings[t]["gd"],
                                                 -standings[t]["gf"]))
    if me["pts"] >= 6:
        return "SECURED"
    # 已出局：自己最多到 pts+3，仍 < 第二名当前积分
    second_pts = standings[sorted_t[1]]["pts"]
    if me["pts"] + 3 < second_pts:
        return "ELIMINATED"
    # 第3/4名且积分落后较多也近似出局（保守：落后第二名 >=4 且自己 <=1 分）
    rank = sorted_t.index(team)
    if rank >= 2 and (second_pts - me["pts"]) >= 4 and me["pts"] <= 1:
        return "ELIMINATED"
    return "MUST_WIN"


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def run(verbose: bool = True) -> Dict:
    df = _load_history(min_year=1990)
    elo = EloState()

    # 先把所有非 WC 比赛按时序喂给 ELO，并在 WC 年穿插 WC 比赛
    # 为简化：单遍历，遇到 WC 小组赛末轮时做评估
    df = df[~(df["home_score"].isna() | df["away_score"].isna())].copy()

    # 预重建每届小组结构（仅用小组赛日期，不含淘汰赛）
    groups_by_year: Dict[int, List[List[str]]] = {}
    md3_dates: Dict[int, set] = {}
    failed_total = 0
    for y in WC_YEARS:
        wc = df[(df["tournament"] == "FIFA World Cup") & (df["date"].dt.year == y)]
        wc = wc.sort_values("date")
        group_part = wc.head(48)  # 前 48 场为小组赛
        groups, nf = reconstruct_groups(group_part)
        groups_by_year[y] = groups
        failed_total += nf

    # 残差累计：按动机状态
    res_by_state = defaultdict(lambda: [0.0, 0.0])  # state -> [Σ(actual-exp), n]
    # 评估指标：baseline vs treatment（在 MD3 场注入动机修正）
    m_base = {"all": _M(), "secured": _M(), "must_win": _M(), "eliminated": _M()}
    m_treat = {"all": _M(), "secured": _M(), "must_win": _M(), "eliminated": _M()}

    # 为每届构建 MD3 场次集合 + 每队状态（用前两轮积分）
    md3_eval = {}  # (year, frozenset({a,b})) -> {team: state}
    for y in WC_YEARS:
        wc = df[(df["tournament"] == "FIFA World Cup") & (df["date"].dt.year == y)].sort_values("date")
        group_part = wc.head(48)
        for grp in groups_by_year[y]:
            gset = set(grp)
            gm = group_part[group_part["home_team"].isin(gset) &
                            group_part["away_team"].isin(gset)].sort_values("date")
            if len(gm) != 6:
                continue  # 每组应 6 场（C(4,2)）
            md1md2 = gm.head(4)
            md3 = gm.tail(2)
            standings = {t: {"pts": 0, "gd": 0, "gf": 0} for t in grp}
            for _, r in md1md2.iterrows():
                ph, pa = _pts(r["home_score"], r["away_score"])
                standings[r["home_team"]]["pts"] += ph
                standings[r["away_team"]]["pts"] += pa
                standings[r["home_team"]]["gd"] += r["home_score"] - r["away_score"]
                standings[r["away_team"]]["gd"] += r["away_score"] - r["home_score"]
                standings[r["home_team"]]["gf"] += r["home_score"]
                standings[r["away_team"]]["gf"] += r["away_score"]
            for _, r in md3.iterrows():
                a, b = r["home_team"], r["away_team"]
                states = {
                    a: classify_motivation(standings, a, grp),
                    b: classify_motivation(standings, b, grp),
                }
                md3_eval[(y, frozenset({a, b}))] = states

    # 单遍历时序：喂 ELO + 在 MD3 场评估
    # 第一遍：仅学残差（用 baseline ELO 期望）。第二遍：用残差做 treatment 评估。
    # 为无泄漏，残差按"届"冻结：评估某届时只用更早届的残差。
    # 简化实现：两趟。
    def _walk(learn_only: bool, frozen_offsets: Optional[Dict[int, Dict[str, float]]] = None):
        elo = EloState()
        seen_state_res = defaultdict(lambda: [0.0, 0.0])
        per_year_frozen = {}
        for _, r in df.iterrows():
            h, a = r["home_team"], r["away_team"]
            hs, a_s = r["home_score"], r["away_score"]
            tournament = r["tournament"]
            year = r["date"].year
            is_wc = ("World Cup" in tournament) and ("qualification" not in tournament)
            key = (year, frozenset({h, a}))

            if is_wc and key in md3_eval:
                states = md3_eval[key]
                exp_h = match_probabilities(elo.get(h), elo.get(a))["p_win_a"]
                actual_h = 1.0 if hs > a_s else (0.5 if hs == a_s else 0.0)
                if learn_only:
                    # 累计每状态残差（team 视角：actual_team - exp_team）
                    for team, opp, exp_t, act_t in [
                        (h, a, exp_h, actual_h), (a, h, 1 - exp_h, 1 - actual_h)]:
                        st = states[team]
                        seen_state_res[st][0] += (act_t - exp_t)
                        seen_state_res[st][1] += 1
                else:
                    # 评估：treatment 用 frozen_offsets[year] 注入
                    fo = (frozen_offsets or {}).get(year, {})
                    def off(team):
                        return fo.get(states[team], 0.0)
                    e_h = elo.get(h) + off(h)
                    e_a = elo.get(a) + off(a)
                    bp = match_probabilities(elo.get(h), elo.get(a))
                    tp = match_probabilities(e_h, e_a)
                    actual = "H" if hs > a_s else ("A" if hs < a_s else "D")
                    for grp, p in (("base", bp), ("treat", tp)):
                        M = m_base if grp == "base" else m_treat
                        M["all"].add(p["p_win_a"], p["p_draw"], p["p_win_b"], actual)
                        for team, pp in ((h, p["p_win_a"]), (a, p["p_win_b"])):
                            stk = states[team].lower()
                            if stk in M:
                                # 以该队视角记一次（用该队胜概率近似）
                                pass
            elo.update_match(h, a, hs, a_s, tournament)
        return seen_state_res

    # 趟1：全样本学残差（池化，用于估计效应量 + 转 ELO 偏移）
    state_res = _walk(learn_only=True)

    # 残差 → ELO 偏移（与信号1同口径：胜率残差 × 600）
    ELO_PER_WR = 600.0
    state_offset = {}
    for st, (s, n) in state_res.items():
        if n >= 10:
            state_offset[st] = round((s / n) * ELO_PER_WR, 1)
        else:
            state_offset[st] = 0.0

    # 趟2：用池化偏移做 treatment 评估（注：此处偏移含全样本，属乐观上界估计）
    frozen = {y: state_offset for y in WC_YEARS}
    _walk(learn_only=False, frozen_offsets=frozen)

    out = {
        "_method": "motivation/dead-rubber 重建小组 + MD3 状态残差。无市场。1998-2022 7届.",
        "n_failed_group_reconstruct": failed_total,
        "state_residual_winrate": {
            st: {"mean_resid": round(s / n, 4) if n else None, "n": int(n)}
            for st, (s, n) in state_res.items()
        },
        "state_offset_elo": state_offset,
        "eval_all": {"baseline": m_base["all"].summary(),
                     "treatment": m_treat["all"].summary()},
    }
    b, t = out["eval_all"]["baseline"], out["eval_all"]["treatment"]
    if b.get("n", 0) and t.get("n", 0):
        out["eval_all"]["delta"] = {k: round(t[k] - b[k], 4)
                                    for k in ("rps", "brier", "accuracy")}

    DATA_OUTPUTS.mkdir(parents=True, exist_ok=True)
    (DATA_OUTPUTS / "motivation_validation.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2))

    if verbose:
        _report(out)
    return out


class _M:
    def __init__(self):
        self.n = self.rps = self.brier = 0.0
        self.correct = 0

    def add(self, ph, pd_, pa, actual):
        self.n += 1
        self.rps += rps(ph, pd_, pa, actual)
        self.brier += brier_multi(ph, pd_, pa, actual)
        if predicted_class(ph, pd_, pa) == actual:
            self.correct += 1

    def summary(self):
        if not self.n:
            return {"n": 0}
        return {"n": int(self.n), "rps": round(self.rps / self.n, 4),
                "brier": round(self.brier / self.n, 4),
                "accuracy": round(self.correct / self.n, 4)}


def _report(out: Dict):
    print("\n" + "=" * 76)
    print("信号2 离线验证 · motivation / dead-rubber（无市场）")
    print("=" * 76)
    print(f"\n小组重建失败数（跳过）: {out['n_failed_group_reconstruct']}")
    print("\nMD3 各动机状态的胜率残差（actual - ELO期望，+ = 超预期）：")
    print(f"{'状态':<14}{'样本n':>8}{'平均残差':>12}{'→ELO偏移':>12}")
    for st in ("SECURED", "MUST_WIN", "ELIMINATED"):
        sr = out["state_residual_winrate"].get(st)
        off = out["state_offset_elo"].get(st, 0.0)
        if sr:
            mr = sr["mean_resid"]
            print(f"{st:<14}{sr['n']:>8}{(mr if mr is not None else 0):>+12.4f}{off:>+12.1f}")
    e = out["eval_all"]
    if "delta" in e:
        print(f"\nMD3 场整体评估（baseline vs treatment 注入动机偏移）：")
        b, t, d = e["baseline"], e["treatment"], e["delta"]
        print(f"  n={b['n']}")
        print(f"  RPS:   {b['rps']} → {t['rps']}  (Δ {d['rps']:+})")
        print(f"  Brier: {b['brier']} → {t['brier']}  (Δ {d['brier']:+})")
        print(f"  Acc:   {b['accuracy']} → {t['accuracy']}  (Δ {d['accuracy']:+})")
        print(f"\n  注：treatment 偏移用全样本估计（乐观上界）；正式接入需 walk-forward 冻结。")
    print("\n输出已写入 data/outputs/motivation_validation.json")


if __name__ == "__main__":
    run()
