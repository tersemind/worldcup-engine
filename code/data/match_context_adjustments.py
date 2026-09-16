"""
从 h2h.json / referee.json 聚合出 per-team 的概率调整百分点。

供 synthesizer.py 调用，思路与 injuries_fetcher.get_health_adjustments() 同构。

设计原则：
- h2h: 心理压制。近 5 战胜率显著高 → 给优势方 +pp，劣势方 -pp。
       仅当 confidence != "低" 且至少 3 场样本时启用。
- referee: 裁判风格。严判（黄牌/犯规多）→ 历史上利于排名低的球队（裁判"压制力量型"）；
           样本太薄（per_match 字段 null）时返回中性 0。

⚠ 失败必须吞掉 → 返回空 dict，保证不破坏主合成流程。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.io import DATA_RAW


# ---------------------------------------------------------------------------
# H2H 心理压制
# ---------------------------------------------------------------------------

# 调整曲线（保守）：
#   3-4 场样本：满压制（W-0-0）= ±0.5pp
#   5+ 场样本：满压制（W-0-0）= ±0.8pp
# cap ±1.0pp（不让单一信号反客为主）
H2H_MAX_PP = 1.0


def _parse_record(summary: str) -> tuple:
    """'Brazil 3-1-1' → (3, 1, 1)；解析失败返回 (0,0,0)"""
    if not summary:
        return (0, 0, 0)
    parts = summary.strip().split()
    if len(parts) < 2:
        return (0, 0, 0)
    nums = parts[-1].split("-")
    if len(nums) != 3:
        return (0, 0, 0)
    try:
        return (int(nums[0]), int(nums[1]), int(nums[2]))
    except ValueError:
        return (0, 0, 0)


def _h2h_pp_for_team_a(record: tuple, n_matches: int) -> float:
    """正数=对 team_a 有利，负数=对 team_a 不利"""
    w, d, loss = record
    total = w + d + loss
    if total < 3:
        return 0.0
    win_rate = w / total
    # 中性 0.5；偏离 0.5 即压制/被压制
    delta = win_rate - 0.5
    cap = 0.5 if total <= 4 else 0.8
    pp = delta * 2 * cap  # delta=0.5 → ±cap
    return max(-H2H_MAX_PP, min(H2H_MAX_PP, pp))


def get_h2h_adjustments() -> Dict[str, float]:
    """
    返回 {team: psych_pp}。一支球队的多场 h2h 加总（按 cap 内裁剪），
    适合 synthesizer 当作 psych 附加项使用。
    """
    path = DATA_RAW / "h2h.json"
    if not path.exists():
        return {}
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception:
        return {}

    out: Dict[str, float] = {}
    for key, entry in data.items():
        if key.startswith("_"):
            continue
        if not isinstance(entry, dict):
            continue
        if entry.get("confidence") == "低":
            continue
        team_a = entry.get("team_a")
        team_b = entry.get("team_b")
        if not team_a or not team_b:
            continue
        record = _parse_record(entry.get("summary_record", ""))
        n = sum(record)
        pp_a = _h2h_pp_for_team_a(record, n)
        if pp_a == 0:
            continue
        out[team_a] = out.get(team_a, 0.0) + pp_a
        out[team_b] = out.get(team_b, 0.0) - pp_a

    # 再次裁剪到 cap（多场叠加不应超过单场 cap 太多）
    for t in out:
        out[t] = max(-H2H_MAX_PP, min(H2H_MAX_PP, out[t]))
    return out


# ---------------------------------------------------------------------------
# 裁判风格
# ---------------------------------------------------------------------------

# 严判（黄牌 ≥ 5.5/场 或风格含 "strict")：
#   - 利于低排名/防守反击型（保守 +0.2pp）
#   - 不利于持球高位强队（-0.2pp）
# 当 per_match 字段为 null 且 style 不明 → 中性 0
REFEREE_MAX_PP = 0.4

STRICT_KEYWORDS = ("strict", "严", "harsh")
LENIENT_KEYWORDS = ("lenient", "宽")


def _referee_severity(entry: dict) -> int:
    """+1=严判, -1=宽松, 0=中性/未知"""
    yellow = entry.get("yellow_per_match")
    style = (entry.get("style") or "").lower()
    if yellow is not None:
        if yellow >= 5.5:
            return 1
        if yellow <= 3.0:
            return -1
    if any(k in style for k in STRICT_KEYWORDS):
        return 1
    if any(k in style for k in LENIENT_KEYWORDS):
        return -1
    return 0


def get_referee_adjustments(teams_data: Dict[str, dict] | None = None) -> Dict[str, float]:
    """
    返回 {team: context_pp}。
    teams_data 用来判别强弱队（elo 高低）；若不传则用保守的常数 0.2/-0.2。
    """
    path = DATA_RAW / "referee.json"
    if not path.exists():
        return {}
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception:
        return {}

    # 计算 elo 中位数用于强弱判定
    elo_median = 1800.0
    if teams_data:
        elos = [d.get("elo", 0) for d in teams_data.values()
                if isinstance(d, dict) and d.get("group") != "_" and d.get("elo")]
        if elos:
            elo_median = sorted(elos)[len(elos) // 2]

    out: Dict[str, float] = {}
    for key, entry in data.items():
        if key.startswith("_"):
            continue
        if not isinstance(entry, dict):
            continue
        if entry.get("confidence") == "低":
            continue
        severity = _referee_severity(entry)
        if severity == 0:
            continue
        team_a = entry.get("team_a")
        team_b = entry.get("team_b")
        if not team_a or not team_b:
            continue
        # 严判：低 elo 队 +0.2pp，高 elo 队 -0.2pp
        elo_a = teams_data.get(team_a, {}).get("elo", elo_median) if teams_data else elo_median
        elo_b = teams_data.get(team_b, {}).get("elo", elo_median) if teams_data else elo_median
        for team, elo in [(team_a, elo_a), (team_b, elo_b)]:
            sign = -1 if elo > elo_median else +1  # 强队挨严判减分
            pp = severity * sign * 0.2
            out[team] = out.get(team, 0.0) + pp

    for t in out:
        out[t] = max(-REFEREE_MAX_PP, min(REFEREE_MAX_PP, out[t]))
    return out


# ---------------------------------------------------------------------------
# Weather 极端天气
# ---------------------------------------------------------------------------
#
# wbgt_estimate_c（湿球黑球温度估算）是 FIFA 用的高温作业指标：
#   < 22 安全
#   22-26 medium
#   26-28 high
#   > 28 extreme
# 高温利防守反击/低位防守，不利持球高位的强队。
# 用 risk_level 字段（fetch_weather 已分级）+ Elo 中位数判强弱。

WEATHER_MAX_PP = 0.4


def get_weather_adjustments(teams_data: Dict[str, dict] | None = None) -> Dict[str, float]:
    """{team: context_pp}：极端天气下强 Elo 队微减，弱队微加"""
    path = DATA_RAW / "weather.json"
    if not path.exists():
        return {}
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception:
        return {}

    elo_median = 1800.0
    if teams_data:
        elos = [d.get("elo", 0) for d in teams_data.values()
                if isinstance(d, dict) and d.get("group") != "_" and d.get("elo")]
        if elos:
            elo_median = sorted(elos)[len(elos) // 2]

    # risk_level → 风险系数
    risk_severity = {
        "extreme": 2,
        "high":    1,
        "medium":  0,    # 中性不调整，避免噪声污染整届
        "low":     0,
    }

    out: Dict[str, float] = {}
    for m in data.get("matches", []):
        we = m.get("weather", {})
        severity = risk_severity.get(we.get("risk_level", "low"), 0)
        if severity == 0:
            continue
        team_a = m.get("team_a")
        team_b = m.get("team_b")
        if not team_a or not team_b:
            continue
        elo_a = teams_data.get(team_a, {}).get("elo", elo_median) if teams_data else elo_median
        elo_b = teams_data.get(team_b, {}).get("elo", elo_median) if teams_data else elo_median
        # 强队挨极端天气微减
        for team, elo in [(team_a, elo_a), (team_b, elo_b)]:
            sign = -1 if elo > elo_median else +1
            pp = severity * sign * 0.15   # high=±0.15, extreme=±0.30
            out[team] = out.get(team, 0.0) + pp

    for t in out:
        out[t] = max(-WEATHER_MAX_PP, min(WEATHER_MAX_PP, out[t]))
    return out


# ---------------------------------------------------------------------------
# Lineups 缺主力
# ---------------------------------------------------------------------------
#
# lineups.json 没有结构化的 absences 字段，但有：
#   - notes / key_changes_vs_previous（LLM 抽的自由文本）
#   - is_captain（队长是否在首发）
#
# 信号设计：
#   - 队长缺阵（启发式：notes 含 "out", "miss", "injur", "absent", "ban", "suspended"）→ -0.3pp
#   - confidence == "低" → 不启用（数据本身不可信）
#   - confirmed=false 且没有 key_changes → 不调整（这是默认状态）

LINEUP_MAX_PP = 0.5

ABSENCE_KEYWORDS = (
    "out", "miss", "absent", "injur", "ban", "suspend",
    "缺阵", "缺席", "伤", "停赛",
)


def _team_has_concerning_changes(team_data: dict) -> tuple[bool, str]:
    """返回 (是否有担忧的变化, 命中的关键字)"""
    if not isinstance(team_data, dict):
        return False, ""
    text_bits = []
    notes = team_data.get("notes") or ""
    if isinstance(notes, str):
        text_bits.append(notes)
    changes = team_data.get("key_changes_vs_previous") or []
    if isinstance(changes, list):
        for c in changes:
            if isinstance(c, str):
                text_bits.append(c)
            elif isinstance(c, dict):
                text_bits.append(str(c.get("description", "")))
    text = " ".join(text_bits).lower()
    for kw in ABSENCE_KEYWORDS:
        if kw in text:
            return True, kw
    return False, ""


def get_lineup_adjustments() -> Dict[str, float]:
    """{team: health_pp}：缺主力（队长伤/停赛）→ 该队微减"""
    path = DATA_RAW / "lineups.json"
    if not path.exists():
        return {}
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception:
        return {}

    out: Dict[str, float] = {}
    for key, entry in data.get("lineups", {}).items():
        if key.startswith("_"):
            continue
        if not isinstance(entry, dict):
            continue
        for side_key in ("team_a", "team_b"):
            side = entry.get(side_key)
            if not isinstance(side, dict):
                continue
            if side.get("confidence") == "低":
                continue
            team_name = side.get("team")
            if not team_name:
                continue
            hit, _kw = _team_has_concerning_changes(side)
            if not hit:
                continue
            out[team_name] = out.get(team_name, 0.0) - 0.3

    for t in out:
        out[t] = max(-LINEUP_MAX_PP, min(LINEUP_MAX_PP, out[t]))
    return out


# ---------------------------------------------------------------------------
# CLI 调试入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        from utils.io import load_teams
        teams = load_teams()
    except Exception:
        teams = None

    print("=== H2H adjustments ===")
    for t, pp in sorted(get_h2h_adjustments().items(), key=lambda x: -abs(x[1])):
        print(f"  {t:<20s} {pp:+.2f} pp")

    print("\n=== Referee adjustments ===")
    for t, pp in sorted(get_referee_adjustments(teams).items(), key=lambda x: -abs(x[1])):
        print(f"  {t:<20s} {pp:+.2f} pp")

    print("\n=== Weather adjustments ===")
    for t, pp in sorted(get_weather_adjustments(teams).items(), key=lambda x: -abs(x[1])):
        print(f"  {t:<20s} {pp:+.2f} pp")

    print("\n=== Lineups adjustments ===")
    for t, pp in sorted(get_lineup_adjustments().items(), key=lambda x: -abs(x[1])):
        print(f"  {t:<20s} {pp:+.2f} pp")
