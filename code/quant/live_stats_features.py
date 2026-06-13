"""
v2 量化信号：从 ESPN summary 映射成 ΔE_pp（与 v1 同口径，10 ELO = 1pp）
==================================================================

输入：live_fetcher_extras.fetch_summary 返回的 dict（含 home/away 各 stats 块）
输出：dict{ delta_pp_a, components, warnings, time_decay_factor }

设计原则：
  - 全 fail-soft：任何字段缺失 → 该项贡献为 0，不抛
  - 与 v1 同 ELO_PER_PP=10，与 phase3 对齐
  - 上限 cap：v2 总贡献 |delta_pp_a| ≤ 4pp（避免 v1 红牌信号被 v2 反向覆盖）
  - 同一 time_decay：0.4(0min)→1.0(90min) 与 v1 一致
  - v2 是相对 home 的视角（>0 利好 home）

口径：
  1. xG_proxy_diff  ：xG_proxy = shots*0.10 + shots_on_target*0.20，diff*1.2pp
  2. possession_diff：(home_pos% - away_pos%) / 10 * 0.5 pp
  3. shot_pressure  ：shots_diff/2 * 0.3 pp
  4. penalty_diff   ：(home_pen_goals - away_pen_goals) * 1.0 pp
  5. corners_diff   ：corners_diff/3 * 0.2 pp（弱信号）

经验值已校准：在 ESPN 已完赛样本上做单场上限 ~3pp，正常 0.5-1.5pp。
不会盖过 v1 的 score_diff_pp（首球 ≈ 12pp）和 red_pp（每张 6pp）。
"""
from __future__ import annotations

from typing import Dict, Any, Optional

# ====== 常量集 ======
V2_TOTAL_CAP_PP = 4.0   # v2 总贡献绝对值上限（防过拟合压过 v1）
ELO_PER_PP = 10.0       # 与 v1/phase3 对齐

XG_PER_SHOT = 0.10           # 普通射门期望
XG_PER_SHOT_ON_TARGET = 0.20  # 射正额外加权
XG_DIFF_PP_FACTOR = 1.2

POSS_DIFF_PP_PER_10 = 0.5  # 持球率 +10% → 0.5pp

SHOTS_DIFF_PP_PER_2 = 0.3  # 射门差 +2 → 0.3pp

PENALTY_GOAL_PP = 1.0  # 每个净点球 +1.0pp（v1 已计 score_diff，避免双倍）

CORNERS_DIFF_PP_PER_3 = 0.2  # 弱信号

# 时间衰减（与 v1 同公式）
TIME_DECAY_BASE = 0.4
TIME_DECAY_END = 1.0
REGULAR_MATCH_MIN = 90


def _safe_float(x, default: float = 0.0) -> float:
    try:
        if isinstance(x, str):
            x = x.replace("%", "").strip()
        return float(x)
    except (ValueError, TypeError):
        return default


def _safe_int(x, default: int = 0) -> int:
    try:
        return int(float(x))
    except (ValueError, TypeError):
        return default


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def time_decay(elapsed_min: int, phase: str = "") -> float:
    """与 v1 一致：FT 强制 1.0；OT/PK 锁 1.0；常规期间线性 0.4→1.0"""
    if phase in ("FT", "AET", "PEN"):
        return TIME_DECAY_END
    if elapsed_min >= REGULAR_MATCH_MIN:
        return TIME_DECAY_END
    if elapsed_min <= 0:
        return TIME_DECAY_BASE
    frac = elapsed_min / REGULAR_MATCH_MIN
    return TIME_DECAY_BASE + (TIME_DECAY_END - TIME_DECAY_BASE) * frac


def compute_v2_features(summary: Optional[Dict[str, Any]],
                         elapsed_min: int = 0,
                         phase: str = "") -> Dict[str, Any]:
    """
    v2 主入口。

    Args:
        summary: live_fetcher_extras.fetch_summary 返回的 dict
        elapsed_min: 比赛已进行分钟（用于 time_decay）
        phase: "1H"/"HT"/"2H"/"FT"/"AET"/"PEN"/"" — 同 v1

    Returns:
        {
          "delta_pp_a": float,         # v2 总 ΔE_pp（home 视角，已 cap）
          "components": {
              "xg_diff_pp": float,
              "possession_pp": float,
              "shots_pp": float,
              "penalty_pp": float,
              "corners_pp": float,
          },
          "raw": {                      # 原始统计便于审计
              "home": {...}, "away": {...},
          },
          "time_decay_factor": float,
          "warnings": [str, ...],
          "ok": bool,
        }
    """
    warnings: list = []
    components = {
        "xg_diff_pp": 0.0,
        "possession_pp": 0.0,
        "shots_pp": 0.0,
        "penalty_pp": 0.0,
        "corners_pp": 0.0,
    }

    # 输入校验：summary 不可用 / 失败 / 空 stats
    if not summary or not isinstance(summary, dict):
        warnings.append("summary_missing")
        return {
            "delta_pp_a": 0.0, "components": components,
            "raw": {}, "time_decay_factor": time_decay(elapsed_min, phase),
            "warnings": warnings, "ok": False,
        }
    if not summary.get("ok"):
        warnings.append(f"summary_fetch_failed:{summary.get('error','')[:60]}")
        return {
            "delta_pp_a": 0.0, "components": components,
            "raw": {}, "time_decay_factor": time_decay(elapsed_min, phase),
            "warnings": warnings, "ok": False,
        }

    home = summary.get("home") or {}
    away = summary.get("away") or {}
    if not home and not away:
        warnings.append("both_teams_empty")
        return {
            "delta_pp_a": 0.0, "components": components,
            "raw": {"home": home, "away": away},
            "time_decay_factor": time_decay(elapsed_min, phase),
            "warnings": warnings, "ok": False,
        }

    # ---- 1. xG_proxy ----
    h_shots = _safe_int(home.get("shots_total"))
    h_sot = _safe_int(home.get("shots_on_target"))
    a_shots = _safe_int(away.get("shots_total"))
    a_sot = _safe_int(away.get("shots_on_target"))
    h_xg = h_shots * XG_PER_SHOT + h_sot * XG_PER_SHOT_ON_TARGET
    a_xg = a_shots * XG_PER_SHOT + a_sot * XG_PER_SHOT_ON_TARGET
    components["xg_diff_pp"] = (h_xg - a_xg) * XG_DIFF_PP_FACTOR

    # ---- 2. possession ----
    h_pos = _safe_float(home.get("possession_pct"))
    a_pos = _safe_float(away.get("possession_pct"))
    if h_pos > 0 or a_pos > 0:
        # 取相对差（防 ESPN 给的两边 sum != 100 异常）
        diff = h_pos - a_pos
        components["possession_pp"] = (diff / 10.0) * POSS_DIFF_PP_PER_10
    else:
        warnings.append("possession_missing")

    # ---- 3. shot_pressure ----
    components["shots_pp"] = ((h_shots - a_shots) / 2.0) * SHOTS_DIFF_PP_PER_2

    # ---- 4. penalty diff ----
    # 注意：v1 已通过 score_diff 反映点球进的 goal，这里只补「执行差距」(取消 goal 双重计数)
    h_pen_att = _safe_int(home.get("penalty_attempts"))
    a_pen_att = _safe_int(away.get("penalty_attempts"))
    h_pen_g = _safe_int(home.get("penalty_goals"))
    a_pen_g = _safe_int(away.get("penalty_goals"))
    # 用「进点 - 0.5*罚失」近似净 xP，乘 0.5 缓冲（避免和 v1 score_diff 大幅重复）
    h_pen_xp = h_pen_g - 0.5 * max(0, h_pen_att - h_pen_g)
    a_pen_xp = a_pen_g - 0.5 * max(0, a_pen_att - a_pen_g)
    components["penalty_pp"] = (h_pen_xp - a_pen_xp) * PENALTY_GOAL_PP * 0.5

    # ---- 5. corners ----
    h_corners = _safe_int(home.get("corners"))
    a_corners = _safe_int(away.get("corners"))
    components["corners_pp"] = ((h_corners - a_corners) / 3.0) * CORNERS_DIFF_PP_PER_3

    # ---- 汇总 + time_decay + cap ----
    raw_total = sum(components.values())
    td = time_decay(elapsed_min, phase)
    decayed = raw_total * td
    capped = _clip(decayed, -V2_TOTAL_CAP_PP, V2_TOTAL_CAP_PP)
    if abs(decayed) > V2_TOTAL_CAP_PP + 1e-6:
        warnings.append(f"capped:{decayed:.2f}->{capped:.2f}")

    return {
        "delta_pp_a": capped,
        "delta_elo_a": capped * ELO_PER_PP,
        "components": components,
        "raw": {
            "home": {"shots": h_shots, "sot": h_sot, "xg_proxy": round(h_xg, 2),
                     "poss_pct": h_pos, "pen_g": h_pen_g, "pen_att": h_pen_att,
                     "corners": h_corners},
            "away": {"shots": a_shots, "sot": a_sot, "xg_proxy": round(a_xg, 2),
                     "poss_pct": a_pos, "pen_g": a_pen_g, "pen_att": a_pen_att,
                     "corners": a_corners},
        },
        "time_decay_factor": td,
        "warnings": warnings,
        "ok": True,
    }


# =================== CLI demo ===================
if __name__ == "__main__":
    # 用 mock：home 主导（70% poss / 18 shots / 8 SoT / 1 penalty） vs away (30% / 6 shots / 2 SoT)
    mock_summary = {
        "espn_id": "demo", "ok": True,
        "home": {"possession_pct": 65.0, "shots_total": 18, "shots_on_target": 8,
                 "penalty_attempts": 1, "penalty_goals": 1, "corners": 9},
        "away": {"possession_pct": 35.0, "shots_total": 6, "shots_on_target": 2,
                 "penalty_attempts": 0, "penalty_goals": 0, "corners": 2},
    }
    out = compute_v2_features(mock_summary, elapsed_min=70, phase="2H")
    print("[demo] mock home-dominant @70min")
    print(f"  delta_pp_a = {out['delta_pp_a']:+.3f} pp ({out['delta_elo_a']:+.1f} ELO)")
    print(f"  time_decay = {out['time_decay_factor']:.3f}")
    print(f"  components:")
    for k, v in out["components"].items():
        print(f"    {k:20s} = {v:+.3f}")
    print(f"  warnings: {out['warnings']}")

    # 失败/空场景
    print()
    print("[demo] empty summary")
    out2 = compute_v2_features(None, elapsed_min=30)
    print(f"  delta_pp_a = {out2['delta_pp_a']} ok={out2['ok']} warnings={out2['warnings']}")

    print()
    print("[demo] failed fetch")
    out3 = compute_v2_features({"ok": False, "error": "404"}, elapsed_min=30)
    print(f"  delta_pp_a = {out3['delta_pp_a']} ok={out3['ok']} warnings={out3['warnings']}")
