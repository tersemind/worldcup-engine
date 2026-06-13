"""
Bracket Difficulty 调整（synthesizer adj_bracket 项）
=====================================================

诊断起源：phase3 验收发现 Germany 残差 -3.13pp，根因是 synth 的 8 项 adj
完全不知道 bracket draw 结构。统计验证：
  Upper 半区 12 强队（小组 1/2 名）avg elo=1915
  Lower 半区 12 强队 avg elo=1932（仅差 17）
  但 pure-MC 给两半区冠军份额 60.01% vs 66.04%（差 6pp）
  原因：Upper 半区 R32 有 Brazil-Japan(85 elo 差) + Germany-Norway(18 elo 差)
        两场强对强；Lower 半区 Spain/Argentina 都对弱队（Austria 327 / Uruguay 223）

设计：
1. 给每队估算"bracket 路径强度" path_strength = R32 对手 + R16/QF/SF 期望对手
2. avg path_strength = 全队中位数
3. adj_bracket = -K × (path_strength - median) / 100
   - 路径强 → 负 adj（synth 应下调 final）
   - 路径弱 → 正 adj
4. cap ±2.5pp，符合现有 8 项 adj 量级（最大 adj_squad ±3.5）

实现要点：
- **不重跑 MC**：仅用 elo + bracket slot + 小组排名推断对手期望强度
- **try/except 包裹**：bracket.json 缺失或 group_results 不可用时返回 {}
- **小组 1/2 名假设**：用当前 elo 推断小组出线名次（高 elo 默认 1 名）
- **3rd-place 简化**：路径上 3rd-place 槽位用 "best-third 池" 平均 elo 替代
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Optional

# 不破坏主流程：所有 import 失败都返回 {}
ROOT = Path(__file__).parent.parent.parent
DATA_RAW = ROOT / "data" / "raw"

# 调整参数
ADJ_BRACKET_K = 5.0          # path_strength 偏差 100 elo 时 → adj_bracket 5pp（再 cap）
ADJ_BRACKET_CAP = 2.5        # ±2.5pp cap，与其他 adj 同量级
# 等权 + 略偏后期（后期对手更强，对夺冠概率影响更大；不能 P(advance) 加权
# 否则 R32 弱队会主导路径分，违背 bracket 阻力的实际语义）
PATH_WEIGHTS = {
    "R32": 1.0,
    "R16": 1.0,
    "QF":  1.2,
    "SF":  1.4,
}


def _load_bracket() -> Optional[dict]:
    p = DATA_RAW / "bracket.json"
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def _build_slot_team(teams_data: dict) -> Dict[str, str]:
    """根据 elo 推断小组 1/2 名 → slot map (e.g. 'A1', 'A2')"""
    group_map: Dict[str, list] = {}
    for name, info in teams_data.items():
        if not isinstance(info, dict):
            continue
        g = info.get("group")
        if not g or g == "_":
            continue
        group_map.setdefault(g, []).append((name, info.get("elo", 0)))
    slot_team: Dict[str, str] = {}
    third_pool: list = []  # 每组第 3 名（fallback for 3rd-place 槽）
    for g, lst in group_map.items():
        lst.sort(key=lambda x: -x[1])
        if len(lst) >= 1:
            slot_team[f"{g}1"] = lst[0][0]
        if len(lst) >= 2:
            slot_team[f"{g}2"] = lst[1][0]
        if len(lst) >= 3:
            third_pool.append(lst[2])
    # 把 3rd_pool 平均 elo 存进 _third_avg_elo 供后续用
    if third_pool:
        avg = sum(e for _, e in third_pool) / len(third_pool)
        slot_team["_third_avg_elo"] = avg  # type: ignore
    return slot_team


def _team_to_r32_match(slot_team: dict, r32: list) -> Dict[str, dict]:
    """team → 它所在 R32 比赛 dict"""
    out = {}
    for r in r32:
        sa, sb = r.get("slot_a"), r.get("slot_b")
        ta = slot_team.get(sa) if sa and "3rd" not in sa else None
        tb = slot_team.get(sb) if sb and "3rd" not in sb else None
        if ta:
            out[ta] = r
        if tb:
            out[tb] = r
    return out


def _opp_elo(slot_team: dict, slot: str, teams_data: dict) -> float:
    """根据 slot 字符串返回期望对手 elo（3rd-place 用池平均）"""
    if slot is None:
        return 0.0
    if "3rd" in slot:
        return float(slot_team.get("_third_avg_elo", 1750.0))
    t = slot_team.get(slot)
    if not t:
        return 0.0
    info = teams_data.get(t, {})
    return float(info.get("elo", 0)) if isinstance(info, dict) else 0.0


def _r16_avg_opp_elo(match_id: int, bracket: dict, slot_team: dict, teams_data: dict) -> float:
    """估算 R16 阶段对手 elo = 对面那场 R32 的胜者期望 elo（强者更可能赢）"""
    r16_pairings = bracket.get("_round_of_16_pairings", [])
    r32 = bracket.get("round_of_32", [])
    pair = None
    for p in r16_pairings:
        if p.get("winner_of_match_a") == match_id:
            pair = p.get("winner_of_match_b"); break
        if p.get("winner_of_match_b") == match_id:
            pair = p.get("winner_of_match_a"); break
    if pair is None:
        return 0.0
    for r in r32:
        if r.get("match") == pair:
            return _r32_winner_expected_elo(r, slot_team, teams_data)
    return 0.0


def _r32_winner_expected_elo(match: dict, slot_team: dict, teams_data: dict) -> float:
    """
    R32 比赛胜者期望 elo = elo 加权（高 elo 队更可能赢，所以胜者期望 elo 偏高）
    用 max(ea, eb) + 0.3*(min) 简化，代表强者赢概率更高
    """
    ea = _opp_elo(slot_team, match.get("slot_a"), teams_data)
    eb = _opp_elo(slot_team, match.get("slot_b"), teams_data)
    if ea <= 0 and eb <= 0:
        return 0.0
    if ea <= 0:
        return eb
    if eb <= 0:
        return ea
    # logistic-style：强者占比 ~ 1 / (1 + 10^(-Δ/400))
    import math
    diff = ea - eb
    p_a = 1.0 / (1.0 + 10 ** (-diff / 400.0))
    return p_a * ea + (1 - p_a) * eb


def _qf_sf_avg_opp_elo(r32_match_id: int, bracket: dict, slot_team: dict,
                       teams_data: dict, level: str = "QF") -> float:
    """
    QF/SF 阶段对手 elo：对面 R32 比赛**胜者**的期望 elo（按 elo 加权，
    强者更可能晋级所以胜者 elo 偏高），再对多场取平均

    level="QF": 对手是同 QF 块内但跨 R16 的另一对 R16 配对的 2 场 R32 胜者
    level="SF": 对手是另一个 QF 半区的 8 场 R32 胜者
    """
    r32 = bracket.get("round_of_32", [])

    if level == "QF":
        if r32_match_id <= 76:
            qf_block = [73, 74, 75, 76]
        elif r32_match_id <= 80:
            qf_block = [77, 78, 79, 80]
        elif r32_match_id <= 84:
            qf_block = [81, 82, 83, 84]
        else:
            qf_block = [85, 86, 87, 88]
        r16_partner = {73: 74, 74: 73, 75: 76, 76: 75,
                       77: 78, 78: 77, 79: 80, 80: 79,
                       81: 82, 82: 81, 83: 84, 84: 83,
                       85: 86, 86: 85, 87: 88, 88: 87}
        skip = {r32_match_id, r16_partner.get(r32_match_id, -1)}
        qf_opps = [m for m in qf_block if m not in skip]
    else:  # SF
        if r32_match_id <= 80:
            qf_opps = list(range(81, 89))
        else:
            qf_opps = list(range(73, 81))

    # 每场 R32 胜者 expected elo（强者更可能赢）
    elos = []
    for m in qf_opps:
        for r in r32:
            if r.get("match") == m:
                we = _r32_winner_expected_elo(r, slot_team, teams_data)
                if we > 0:
                    elos.append(we)
                break
    return sum(elos) / len(elos) if elos else 0.0


def compute_bracket_strength(teams_data: Optional[dict] = None) -> Dict[str, float]:
    """
    每队 bracket 路径强度（加权平均期望对手 elo）

    返回 {team: path_strength_elo}
    """
    if teams_data is None:
        try:
            import sys as _sys
            _sys.path.insert(0, str(Path(__file__).parent.parent))
            from utils.io import load_teams
            teams_data = load_teams()
        except Exception:
            return {}
    bracket = _load_bracket()
    if not bracket:
        return {}
    r32 = bracket.get("round_of_32", [])
    if not r32:
        return {}

    slot_team = _build_slot_team(teams_data)
    team_match = _team_to_r32_match(slot_team, r32)

    out: Dict[str, float] = {}
    for team, match in team_match.items():
        sa, sb = match.get("slot_a"), match.get("slot_b")
        # R32 对手 = 另一 slot
        if slot_team.get(sa) == team:
            r32_opp = _opp_elo(slot_team, sb, teams_data)
        else:
            r32_opp = _opp_elo(slot_team, sa, teams_data)
        m_id = match.get("match")
        r16_opp = _r16_avg_opp_elo(m_id, bracket, slot_team, teams_data)
        qf_opp = _qf_sf_avg_opp_elo(m_id, bracket, slot_team, teams_data, "QF")
        sf_opp = _qf_sf_avg_opp_elo(m_id, bracket, slot_team, teams_data, "SF")

        # 加权平均
        w_total = sum(PATH_WEIGHTS.values())
        path = (PATH_WEIGHTS["R32"] * r32_opp +
                PATH_WEIGHTS["R16"] * r16_opp +
                PATH_WEIGHTS["QF"]  * qf_opp +
                PATH_WEIGHTS["SF"]  * sf_opp) / w_total
        out[team] = path
    return out


def get_bracket_adjustments(teams_data: Optional[dict] = None) -> Dict[str, float]:
    """
    返回每队 adj_bracket（pp 单位，加进 synthesizer.final_probability）

    路径强 → 负 adj；路径弱 → 正 adj。cap ±ADJ_BRACKET_CAP。
    失败时返回 {}（主流程兜底）。
    """
    try:
        strengths = compute_bracket_strength(teams_data)
        if not strengths:
            return {}
        # 用中位数作基准（更稳健，避免极端值拉偏）
        vals = sorted(strengths.values())
        median = vals[len(vals) // 2] if vals else 0.0
        out: Dict[str, float] = {}
        for team, s in strengths.items():
            delta = s - median  # 正 = 路径强；负 = 路径弱
            adj = -ADJ_BRACKET_K * delta / 100.0
            # cap
            adj = max(-ADJ_BRACKET_CAP, min(ADJ_BRACKET_CAP, adj))
            out[team] = round(adj, 2)
        return out
    except Exception:
        return {}


# ============== 自测 ==============
if __name__ == "__main__":
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.io import load_teams

    teams = load_teams()
    print("=== Bracket path strength（加权对手期望 elo）===")
    strengths = compute_bracket_strength(teams)
    sorted_s = sorted(strengths.items(), key=lambda x: -x[1])
    print(f"{'Team':<18}{'path_elo':>10}")
    for t, s in sorted_s[:8]:
        print(f"{t:<18}{s:>10.1f}  (HARD)")
    print("  ...")
    for t, s in sorted_s[-6:]:
        print(f"{t:<18}{s:>10.1f}  (easy)")

    print("\n=== adj_bracket（pp）===")
    adj = get_bracket_adjustments(teams)
    sorted_a = sorted(adj.items(), key=lambda x: x[1])
    print(f"{'Team':<18}{'adj_bracket':>14}")
    for t, a in sorted_a[:8]:
        print(f"{t:<18}{a:>+13.2f}pp")
    print("  ...")
    for t, a in sorted_a[-8:]:
        print(f"{t:<18}{a:>+13.2f}pp")
