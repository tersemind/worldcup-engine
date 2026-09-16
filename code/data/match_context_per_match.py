"""
按场计算 H2H / referee / weather / lineups 的 (δA_pp, δB_pp)
================================================================

阶段 2 专用：替代 match_context_adjustments 的"队级聚合"，把同样的单场算法
保留 per-match 粒度，输出供 simulate_match 直接消费的 ΔElo 输入。

设计：
- 完全复用 match_context_adjustments 内已有的 cap、阈值、判定逻辑
- 不修改原文件，避免影响 synthesizer 主流程
- 失败一律吞掉，返回 {}

输出格式：
    {(team_a, team_b): (pp_a, pp_b), ...}
    pp_a/pp_b 可正可负（pp 单位，正=对该队有利）
    注意：以 raw 文件中实际的 (team_a, team_b) 顺序为 key，
    上层 lookup 时需要尝试反向 (B,A) 并交换 (pp_b, pp_a)。
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.io import DATA_RAW
from data.match_context_adjustments import (
    H2H_MAX_PP, REFEREE_MAX_PP, WEATHER_MAX_PP, LINEUP_MAX_PP,
    _parse_record, _h2h_pp_for_team_a, _referee_severity,
    ABSENCE_KEYWORDS, _team_has_concerning_changes,
)

PerMatchKey = Tuple[str, str]
PerMatchVal = Tuple[float, float]


def _add(out: Dict[PerMatchKey, list], key: PerMatchKey, dpa: float, dpb: float):
    """累加单场 ΔppA, ΔppB 到 out[key]（list of [pp_a, pp_b]）"""
    if key not in out:
        out[key] = [0.0, 0.0]
    out[key][0] += dpa
    out[key][1] += dpb


def _h2h_per_match(out: Dict[PerMatchKey, list]):
    path = DATA_RAW / "h2h.json"
    if not path.exists():
        return
    try:
        data = json.load(open(path))
    except Exception:
        return
    for key, entry in data.items():
        if key.startswith("_") or not isinstance(entry, dict):
            continue
        if entry.get("confidence") == "低":
            continue
        team_a = entry.get("team_a"); team_b = entry.get("team_b")
        if not team_a or not team_b:
            continue
        record = _parse_record(entry.get("summary_record", ""))
        n = sum(record)
        pp_a = _h2h_pp_for_team_a(record, n)
        if pp_a == 0:
            continue
        # cap 单场（已在 _h2h_pp_for_team_a 里限制）
        _add(out, (team_a, team_b), pp_a, -pp_a)


def _referee_per_match(out: Dict[PerMatchKey, list], teams_data: Optional[dict]):
    path = DATA_RAW / "referee.json"
    if not path.exists():
        return
    try:
        data = json.load(open(path))
    except Exception:
        return

    elo_median = 1800.0
    if teams_data:
        elos = [d.get("elo", 0) for d in teams_data.values()
                if isinstance(d, dict) and d.get("group") != "_" and d.get("elo")]
        if elos:
            elo_median = sorted(elos)[len(elos) // 2]

    for key, entry in data.items():
        if key.startswith("_") or not isinstance(entry, dict):
            continue
        if entry.get("confidence") == "低":
            continue
        severity = _referee_severity(entry)
        if severity == 0:
            continue
        team_a = entry.get("team_a"); team_b = entry.get("team_b")
        if not team_a or not team_b:
            continue
        elo_a = teams_data.get(team_a, {}).get("elo", elo_median) if teams_data else elo_median
        elo_b = teams_data.get(team_b, {}).get("elo", elo_median) if teams_data else elo_median
        sign_a = -1 if elo_a > elo_median else +1
        sign_b = -1 if elo_b > elo_median else +1
        pp_a = severity * sign_a * 0.2
        pp_b = severity * sign_b * 0.2
        # cap
        pp_a = max(-REFEREE_MAX_PP, min(REFEREE_MAX_PP, pp_a))
        pp_b = max(-REFEREE_MAX_PP, min(REFEREE_MAX_PP, pp_b))
        _add(out, (team_a, team_b), pp_a, pp_b)


def _weather_per_match(out: Dict[PerMatchKey, list], teams_data: Optional[dict]):
    path = DATA_RAW / "weather.json"
    if not path.exists():
        return
    try:
        data = json.load(open(path))
    except Exception:
        return

    elo_median = 1800.0
    if teams_data:
        elos = [d.get("elo", 0) for d in teams_data.values()
                if isinstance(d, dict) and d.get("group") != "_" and d.get("elo")]
        if elos:
            elo_median = sorted(elos)[len(elos) // 2]

    risk_severity = {"extreme": 2, "high": 1, "medium": 0, "low": 0}

    for m in data.get("matches", []):
        we = m.get("weather", {})
        severity = risk_severity.get(we.get("risk_level", "low"), 0)
        if severity == 0:
            continue
        team_a = m.get("team_a"); team_b = m.get("team_b")
        if not team_a or not team_b:
            continue
        elo_a = teams_data.get(team_a, {}).get("elo", elo_median) if teams_data else elo_median
        elo_b = teams_data.get(team_b, {}).get("elo", elo_median) if teams_data else elo_median
        sign_a = -1 if elo_a > elo_median else +1
        sign_b = -1 if elo_b > elo_median else +1
        pp_a = severity * sign_a * 0.15
        pp_b = severity * sign_b * 0.15
        pp_a = max(-WEATHER_MAX_PP, min(WEATHER_MAX_PP, pp_a))
        pp_b = max(-WEATHER_MAX_PP, min(WEATHER_MAX_PP, pp_b))
        _add(out, (team_a, team_b), pp_a, pp_b)


def _lineups_per_match(out: Dict[PerMatchKey, list]):
    path = DATA_RAW / "lineups.json"
    if not path.exists():
        return
    try:
        data = json.load(open(path))
    except Exception:
        return

    for key, entry in data.get("lineups", {}).items():
        if key.startswith("_") or not isinstance(entry, dict):
            continue
        team_a_side = entry.get("team_a") or {}
        team_b_side = entry.get("team_b") or {}
        ta = team_a_side.get("team") if isinstance(team_a_side, dict) else None
        tb = team_b_side.get("team") if isinstance(team_b_side, dict) else None
        if not ta or not tb:
            continue
        pp_a = pp_b = 0.0
        if isinstance(team_a_side, dict) and team_a_side.get("confidence") != "低":
            hit, _ = _team_has_concerning_changes(team_a_side)
            if hit:
                pp_a -= 0.3
        if isinstance(team_b_side, dict) and team_b_side.get("confidence") != "低":
            hit, _ = _team_has_concerning_changes(team_b_side)
            if hit:
                pp_b -= 0.3
        pp_a = max(-LINEUP_MAX_PP, min(LINEUP_MAX_PP, pp_a))
        pp_b = max(-LINEUP_MAX_PP, min(LINEUP_MAX_PP, pp_b))
        if pp_a == 0 and pp_b == 0:
            continue
        _add(out, (ta, tb), pp_a, pp_b)


def compute_per_match_pp(
    teams_data: Optional[dict] = None,
) -> Dict[PerMatchKey, PerMatchVal]:
    """
    汇总 H2H + referee + weather + lineups 4 项 per-match pp 调整

    Returns:
        {(team_a, team_b): (total_pp_a, total_pp_b)}

    teams_data: 用于判断 elo 强弱（referee/weather 需要）；
                None 时使用中位数兜底（仍能跑，只是判定不精确）
    """
    accum: Dict[PerMatchKey, list] = {}
    try:
        _h2h_per_match(accum)
    except Exception:
        pass
    try:
        _referee_per_match(accum, teams_data)
    except Exception:
        pass
    try:
        _weather_per_match(accum, teams_data)
    except Exception:
        pass
    try:
        _lineups_per_match(accum)
    except Exception:
        pass

    return {k: (round(v[0], 3), round(v[1], 3)) for k, v in accum.items()}


# ============== CLI 自测 ==============
if __name__ == "__main__":
    try:
        from utils.io import load_teams
        teams = load_teams()
    except Exception:
        teams = None

    pm = compute_per_match_pp(teams_data=teams)
    print(f"=== Per-match pp ({len(pm)} 场) ===")
    sorted_pm = sorted(pm.items(),
                        key=lambda x: -(abs(x[1][0]) + abs(x[1][1])))
    for (a, b), (pa, pb) in sorted_pm[:30]:
        print(f"  {a:<18} vs {b:<18}  Δpp_a={pa:+5.2f}  Δpp_b={pb:+5.2f}")
