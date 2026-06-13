"""
AI-weighted baseline (阶段 1)
=============================

把 synthesizer 的 8 项 adj_*（pp 单位，后置加在 mc_baseline 上）
反求成每队的等价 Elo 加成 ΔE，让 MC 在抽样阶段就吃到 AI 修正。

核心思路：
- 当前 synthesizer 计算 final_probability = mc_baseline + Σ adj_i (pp)
- 这 8 项 adj 都是 "夺冠概率口径" 的修正，没有反作用回单场胜率
- 阶段 1 用经验系数线性映射：每 1pp 等价 ELO_PER_PP Elo（默认 8）
- 阶段 2 会用反求解器拟合实际斜率（基于 MC 重采样观察）

设计要点：
1. **read-only 调用**: 不修改 synthesizer_report.json，只读
2. **失败回退**: synth 缺失时返回空 dict，让 MC 走原 baseline
3. **clip 边界**: ΔE 限制在 [-80, +80] 防止异常 adj 把队拉到极端
4. **保留可调**: ELO_PER_PP 可以从环境变量覆盖，便于阶段 2 调参

未来扩展（阶段 2）：
- 把 h2h/referee/weather/lineups 4 项改为按场次叠加（match-level shifts）
- 用 MC 重采样反求 ELO_PER_PP 的真实值（不同 mc_baseline 处斜率不同）
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Dict, Tuple, Optional

DATA_OUTPUTS = Path(__file__).parent.parent.parent / "data" / "outputs"
SYNTH_FILE = DATA_OUTPUTS / "synthesizer_report.json"

# 经验系数：每 1pp final_probability 等价的 Elo 加成
# 默认 10.0：基于实证回测（+50 Elo → champion +4.85pp，实测 100k MC，2026-06-13）
# 历史值：8.0 (理论粗估)；阶段 1 验收选定 10.0
# 可通过环境变量 AI_WEIGHTED_ELO_PER_PP 覆盖（用于阶段 2 调参）
#
# 阶段 1 验收数据（100k AI-MC，所有 48 队）：
#   ELO_PER_PP=8  → MAE=0.93pp, Max=3.38pp (Germany)
#   ELO_PER_PP=10 → MAE=0.88pp, Max=3.06pp (Germany)
# 残差主要来自 bracket 半区结构（Germany 在硬半区被 Spain/France 抵消加成），
# 线性反求无法消除，需要阶段 2 逐场叠加才能进一步降低。
DEFAULT_ELO_PER_PP = float(os.environ.get("AI_WEIGHTED_ELO_PER_PP", "10.0"))

# ΔE 限制范围，防止异常 adj 把球队拉到极端（如 health=-15pp 会变成 -120 Elo）
ELO_SHIFT_CLIP = (-80.0, 80.0)


def _load_synth_safe() -> Optional[dict]:
    """读 synthesizer_report.json；不存在或解析失败返回 None"""
    try:
        if not SYNTH_FILE.exists():
            return None
        with open(SYNTH_FILE) as f:
            return json.load(f)
    except Exception:
        return None


def compute_elo_shifts(
    synth_data: Optional[dict] = None,
    elo_per_pp: float = DEFAULT_ELO_PER_PP,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    从 synthesizer 报告计算每队等价 Elo 加成

    Args:
        synth_data: synthesizer_report.json 内容；为 None 时自动读
        elo_per_pp: 经验系数，默认 8 Elo/pp
        verbose: 打印每队的 ΔE 明细

    Returns:
        {team: delta_elo, ...}  每队应在原 Elo 上加多少分
        若 synth 不可用，返回 {}（让上层走原 MC，不破坏主流程）

    注意：
    - 8 项 adj 的总和 = final_probability - mc_baseline（pp）
    - 直接用 synth 字段，不重新计算（避免和 synthesizer.py 实现漂移）
    """
    if synth_data is None:
        synth_data = _load_synth_safe()
    if not synth_data:
        return {}

    shifts: Dict[str, float] = {}
    if verbose:
        print(f"{'Team':<18} {'mc_base%':>9} {'final%':>8} {'Δpp':>7} {'ΔElo':>7}")
        print("-" * 55)

    for team, info in synth_data.items():
        if not isinstance(info, dict):
            continue
        mc_base = info.get("mc_baseline")
        final_p = info.get("final_probability")
        if mc_base is None or final_p is None:
            continue

        # Δpp = 8 项 adj 的总和（synthesizer 已经合好了）
        delta_pp = float(final_p) - float(mc_base)
        delta_elo = delta_pp * elo_per_pp
        # clip 边界
        delta_elo = max(ELO_SHIFT_CLIP[0], min(ELO_SHIFT_CLIP[1], delta_elo))
        shifts[team] = round(delta_elo, 2)

        if verbose:
            print(f"{team:<18} {mc_base:>8.2f}% {final_p:>7.2f}% "
                  f"{delta_pp:>+6.2f} {delta_elo:>+7.1f}")

    return shifts


def explain_shifts(shifts: Dict[str, float], top_n: int = 16) -> str:
    """格式化 ΔE 排行榜（用于报告）"""
    if not shifts:
        return "(无数据：synthesizer_report.json 不可读)"
    sorted_s = sorted(shifts.items(), key=lambda x: -x[1])
    lines = [f"{'Rank':<5} {'Team':<18} {'ΔElo':>8}"]
    lines.append("-" * 35)
    for i, (t, e) in enumerate(sorted_s[:top_n], 1):
        lines.append(f"{i:<5} {t:<18} {e:>+7.1f}")
    if len(sorted_s) > top_n:
        lines.append(f"... ({len(sorted_s) - top_n} more)")
        lines.append("最低者：")
        for t, e in sorted_s[-3:]:
            lines.append(f"      {t:<18} {e:>+7.1f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 阶段 2：分离 "事件级 ΔE（队级）" 与 "比赛级 Δpp（per-match）"
# ---------------------------------------------------------------------------
#
# synthesizer 8 项 adj 中：
#   - 事件级（队级聚合即可）: adj_health, adj_squad_value, adj_psych, adj_context
#   - 比赛级（应当 per-match）: adj_h2h, adj_referee, adj_weather, adj_lineups
#
# 阶段 2 把后 4 项从队级聚合改成按场注入：
#   * 事件级：用 synth 报告里的 4 项 → ΔE 注入所有比赛
#   * 比赛级：从 raw 文件按场重算 (team_a, team_b, date) → (δA_pp, δB_pp)，
#            再 ×ELO_PER_PP 得到该场专属 ΔElo，仅在该场注入 simulate_match
#
# 注意：此处避免与阶段 1 的 compute_elo_shifts 双重计数。
#
EVENT_LEVEL_KEYS = ("adj_health", "adj_squad_value", "adj_psych", "adj_context")
MATCH_LEVEL_KEYS = ("adj_h2h", "adj_referee", "adj_weather", "adj_lineups")
# 注意：adj_bracket 不进 EVENT_LEVEL_KEYS。
# 原因：bracket 阻力本来就在 mc_baseline 里（pure-MC 已经模拟了 bracket）。
# adj_bracket 是给 synth 补的"它本来不知道的 bracket 信息"。
# Phase 2/3 让 MC 贴 synth 时，bracket 部分自然由 MC 自身的模拟提供，
# 不需要再用 ΔE 注入，否则会 double-count。


def compute_event_level_elo_shifts(
    synth_data: Optional[dict] = None,
    elo_per_pp: float = DEFAULT_ELO_PER_PP,
) -> Dict[str, float]:
    """阶段 2 用：仅累加 4 项事件级 adj（剔除比赛级以免双重计数）"""
    if synth_data is None:
        synth_data = _load_synth_safe()
    if not synth_data:
        return {}
    shifts: Dict[str, float] = {}
    for team, info in synth_data.items():
        if not isinstance(info, dict):
            continue
        delta_pp = sum(float(info.get(k, 0.0) or 0.0) for k in EVENT_LEVEL_KEYS)
        delta_elo = delta_pp * elo_per_pp
        delta_elo = max(ELO_SHIFT_CLIP[0], min(ELO_SHIFT_CLIP[1], delta_elo))
        shifts[team] = round(delta_elo, 2)
    return shifts


def compute_match_level_shifts(
    teams_data: Optional[dict] = None,
    elo_per_pp: float = DEFAULT_ELO_PER_PP,
) -> Dict[Tuple[str, str], Tuple[float, float]]:
    """
    阶段 2 用：从 raw H2H/referee/weather/lineups 文件按场算 (δA_elo, δB_elo)

    Returns:
        {(team_a, team_b): (delta_elo_a, delta_elo_b)}
        只用 (A, B) 作为 key（不含 date），因为 simulate_match 只看队对队，
        小组赛阶段每对球队最多一场。淘汰赛阶段 simulate_match 也用 (A,B) 查询，
        若两队历史上 group 赛已对过，会沿用同一 shift（合理近似：都是同一段时间的天气/裁判风格）。

    上层用法：simulate_match(team_a, team_b, ..., match_shifts=shifts)
              内部 lookup (A,B) 或 (B,A) 反向。
    """
    import sys as _sys
    _here = Path(__file__).parent.parent
    if str(_here) not in _sys.path:
        _sys.path.insert(0, str(_here))
    try:
        from data.match_context_per_match import compute_per_match_pp
    except Exception as e:
        # 模块没创建 → 阶段 2 fallback：返回 {}（上层走纯事件级路径）
        return {}

    per_match_pp = compute_per_match_pp(teams_data=teams_data)
    out: Dict[Tuple[str, str], Tuple[float, float]] = {}
    for (a, b), (pp_a, pp_b) in per_match_pp.items():
        elo_a = max(ELO_SHIFT_CLIP[0], min(ELO_SHIFT_CLIP[1], pp_a * elo_per_pp))
        elo_b = max(ELO_SHIFT_CLIP[0], min(ELO_SHIFT_CLIP[1], pp_b * elo_per_pp))
        out[(a, b)] = (round(elo_a, 2), round(elo_b, 2))
    return out


# ---------------------------------------------------------------------------
# 阶段 3：分段 ELO_PER_PP（非线性反求）
# ---------------------------------------------------------------------------
#
# Phase 2 残差诊断（100k AI-MC，排除 floor=0.10% 的 9 队）：
#   高段 mc_base≥8pp     : MAE=1.20pp, 4 队全为负残差（MC 没把 boost 吃满）
#   中段 mc_base 3-8pp   : MAE=1.02pp, 残差方向均衡
#   低段 mc_base<3pp     : MAE=0.77pp, 残差以正为主（MC 给低段过度升降）
#
# 推论：固定 ELO_PER_PP=10 在不同 mc_baseline 处的边际斜率 dP/dE 不同：
#   高段 baseline 处更平（top 队 ΔP/ΔE 较小，需要更大 ELO_PER_PP 才能推到目标）
#   低段 baseline 处更陡（弱队 ΔP/ΔE 较大，10 过于激进，应当减小）
#
# 阶段 3 方案：分段拟合 ELO_PER_PP（默认值基于诊断方向定调，可由
# AI_WEIGHTED_ELO_PER_PP_HIGH/MID/LOW 环境变量覆盖）
#
# Floor 处理：synth final≈0.10% 的队（synthesizer 内部 floor）
# 残差不应被纳入 ΔE 反求 —— 这些队在阶段 3 走 LOW 段即可，floor 误差由 synth
# 决定，与 MC 无关。
SEG_HIGH_THRESHOLD = 8.0  # mc_baseline ≥ 8pp 算高段
SEG_LOW_THRESHOLD = 3.0   # mc_baseline < 3pp 算低段

DEFAULT_ELO_PER_PP_HIGH = float(os.environ.get("AI_WEIGHTED_ELO_PER_PP_HIGH", "12.5"))
DEFAULT_ELO_PER_PP_MID = float(os.environ.get("AI_WEIGHTED_ELO_PER_PP_MID", "10.0"))
DEFAULT_ELO_PER_PP_LOW = float(os.environ.get("AI_WEIGHTED_ELO_PER_PP_LOW", "8.0"))


def _segment_elo_per_pp(mc_base: Optional[float],
                        high: float = DEFAULT_ELO_PER_PP_HIGH,
                        mid: float = DEFAULT_ELO_PER_PP_MID,
                        low: float = DEFAULT_ELO_PER_PP_LOW) -> float:
    """根据 mc_baseline 返回对应的 ELO_PER_PP；mc_base=None 走 mid"""
    if mc_base is None:
        return mid
    if mc_base >= SEG_HIGH_THRESHOLD:
        return high
    if mc_base < SEG_LOW_THRESHOLD:
        return low
    return mid


def compute_event_level_elo_shifts_segmented(
    synth_data: Optional[dict] = None,
    elo_per_pp_high: float = DEFAULT_ELO_PER_PP_HIGH,
    elo_per_pp_mid: float = DEFAULT_ELO_PER_PP_MID,
    elo_per_pp_low: float = DEFAULT_ELO_PER_PP_LOW,
) -> Dict[str, float]:
    """阶段 3 用：按 mc_baseline 分段使用不同 ELO_PER_PP（仅累加 4 项事件级 adj）"""
    if synth_data is None:
        synth_data = _load_synth_safe()
    if not synth_data:
        return {}
    shifts: Dict[str, float] = {}
    for team, info in synth_data.items():
        if not isinstance(info, dict):
            continue
        delta_pp = sum(float(info.get(k, 0.0) or 0.0) for k in EVENT_LEVEL_KEYS)
        mc_base = info.get("mc_baseline")
        epp = _segment_elo_per_pp(mc_base, elo_per_pp_high, elo_per_pp_mid, elo_per_pp_low)
        delta_elo = delta_pp * epp
        delta_elo = max(ELO_SHIFT_CLIP[0], min(ELO_SHIFT_CLIP[1], delta_elo))
        shifts[team] = round(delta_elo, 2)
    return shifts


def compute_match_level_shifts_segmented(
    teams_data: Optional[dict] = None,
    synth_data: Optional[dict] = None,
    elo_per_pp_high: float = DEFAULT_ELO_PER_PP_HIGH,
    elo_per_pp_mid: float = DEFAULT_ELO_PER_PP_MID,
    elo_per_pp_low: float = DEFAULT_ELO_PER_PP_LOW,
) -> Dict[Tuple[str, str], Tuple[float, float]]:
    """
    阶段 3 用：比赛级 ΔE 也按队的 mc_baseline 分段
    每队 (a, b) 各自查 synth.mc_baseline，分别决定 ELO_PER_PP_a 和 ELO_PER_PP_b
    """
    import sys as _sys
    _here = Path(__file__).parent.parent
    if str(_here) not in _sys.path:
        _sys.path.insert(0, str(_here))
    try:
        from data.match_context_per_match import compute_per_match_pp
    except Exception:
        return {}
    if synth_data is None:
        synth_data = _load_synth_safe() or {}

    per_match_pp = compute_per_match_pp(teams_data=teams_data)
    out: Dict[Tuple[str, str], Tuple[float, float]] = {}
    for (a, b), (pp_a, pp_b) in per_match_pp.items():
        mc_a = synth_data.get(a, {}).get("mc_baseline") if isinstance(synth_data.get(a), dict) else None
        mc_b = synth_data.get(b, {}).get("mc_baseline") if isinstance(synth_data.get(b), dict) else None
        epp_a = _segment_elo_per_pp(mc_a, elo_per_pp_high, elo_per_pp_mid, elo_per_pp_low)
        epp_b = _segment_elo_per_pp(mc_b, elo_per_pp_high, elo_per_pp_mid, elo_per_pp_low)
        elo_a = max(ELO_SHIFT_CLIP[0], min(ELO_SHIFT_CLIP[1], pp_a * epp_a))
        elo_b = max(ELO_SHIFT_CLIP[0], min(ELO_SHIFT_CLIP[1], pp_b * epp_b))
        out[(a, b)] = (round(elo_a, 2), round(elo_b, 2))
    return out


# ============== 自测 ==============
if __name__ == "__main__":
    print(f"=== AI-weighted Elo shifts (ELO_PER_PP={DEFAULT_ELO_PER_PP}) ===\n")
    shifts = compute_elo_shifts(verbose=True)
    print(f"\n共计算 {len(shifts)} 队\n")
    print(explain_shifts(shifts))

    print("\n=== 阶段 2: 事件级 ΔElo（剔除比赛级避免双计） ===")
    event_shifts = compute_event_level_elo_shifts()
    print(explain_shifts(event_shifts, top_n=10))

    print("\n=== 阶段 2: 比赛级 ΔElo（per-match） ===")
    match_shifts = compute_match_level_shifts()
    print(f"共计算 {len(match_shifts)} 场比赛级 shift")
    # 取绝对值最大的 8 场展示
    sorted_ms = sorted(match_shifts.items(),
                        key=lambda x: -(abs(x[1][0]) + abs(x[1][1])))
    for (a, b), (ea, eb) in sorted_ms[:8]:
        print(f"  {a:<18} vs {b:<18}  ΔE_a={ea:+6.1f}  ΔE_b={eb:+6.1f}")

    print(f"\n=== 阶段 3: 分段 ELO_PER_PP "
          f"(HIGH={DEFAULT_ELO_PER_PP_HIGH} MID={DEFAULT_ELO_PER_PP_MID} LOW={DEFAULT_ELO_PER_PP_LOW}) ===")
    seg_shifts = compute_event_level_elo_shifts_segmented()
    print(explain_shifts(seg_shifts, top_n=10))
    seg_match = compute_match_level_shifts_segmented()
    print(f"\n比赛级（分段）共 {len(seg_match)} 场")
    sorted_sms = sorted(seg_match.items(),
                        key=lambda x: -(abs(x[1][0]) + abs(x[1][1])))
    for (a, b), (ea, eb) in sorted_sms[:6]:
        print(f"  {a:<18} vs {b:<18}  ΔE_a={ea:+6.1f}  ΔE_b={eb:+6.1f}")
