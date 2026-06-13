"""
蒙特卡洛锦标赛模拟器
执行 N 次完整 2026 世界杯赛程模拟，统计夺冠/进决赛/进四强分布
"""
import numpy as np
import random
import sys
from pathlib import Path

# 引入同包模块
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.elo_engine import match_probabilities
from models.bracket_engine import (
    load_bracket, assign_third_places, resolve_slot, 
    build_real_round_of_32
)
from models.reference_calibrator import calibrate_all as calibrate_all_reference
from utils.io import load_teams, load_groups, save_output


# 参考报告 7.2 表 7.2：卫冕冠军魔咒（自 1962 巴西后 15 届无卫冕成功）
# 在淘汰赛每轮给卫冕冠军一个胜率折扣
DEFENDING_CHAMPION = "Argentina"   # 2022 卡塔尔冠军
DEFENDING_PENALTY = 0.95           # 每场淘汰赛胜率 ×0.95（约 -5% 单场）


# 关键节点 5-Agent 微调表（方案 A2：跨槽位生效）
# 加载 critical_node_adjustments.json → {frozenset({team_a, team_b}): adj_pp_for_lexico_first_team}
# 为避免 A vs B 和 B vs A 顺序歧义：约定 adj 总是加到"按字典序较小队名"的胜率上
_CRITICAL_LOOKUP_CACHE = None

def _load_critical_lookup() -> dict:
    """懒加载 critical_node_adjustments.json → 团队对查找表
    
    返回 {(team_lex_first, team_lex_second): adj_pp_for_first}
    例如 ('Brazil','France') → adj，表示当遇到 Brazil vs France 时给 Brazil 胜率 +adj pp
    """
    global _CRITICAL_LOOKUP_CACHE
    if _CRITICAL_LOOKUP_CACHE is not None:
        return _CRITICAL_LOOKUP_CACHE
    
    import json
    from pathlib import Path
    p = Path(__file__).parent.parent.parent / "data" / "outputs" / "critical_node_adjustments.json"
    if not p.exists():
        _CRITICAL_LOOKUP_CACHE = {}
        return _CRITICAL_LOOKUP_CACHE
    
    raw = json.load(open(p))
    lookup = {}
    for entry in raw.get("adjustments", {}).values():
        if not entry.get("applied"):
            continue
        a, b = entry["team_a"], entry["team_b"]
        adj = entry["final_adjustment_pp"] / 100.0  # pp → 小数
        # adj 是相对 team_a 的；归一到字典序较小队
        if a <= b:
            lookup[(a, b)] = adj
        else:
            lookup[(b, a)] = -adj
    _CRITICAL_LOOKUP_CACHE = lookup
    return lookup


def simulate_match(team_a_data: dict, team_b_data: dict, allow_draw: bool = True,
                    rng: np.random.Generator = None,
                    defending_champion: str = DEFENDING_CHAMPION,
                    defending_penalty: float = DEFENDING_PENALTY,
                    apply_critical_adj: bool = True,
                    elo_shifts: dict = None,
                    match_shifts: dict = None) -> str:
    """
    模拟一场比赛
    返回赢家球队名（如果不允许平局，平局会用 Elo 加权随机决出）
    
    输入：
        team_a_data / team_b_data: {"name": str, "elo": float, ...}
        allow_draw: 小组赛 True，淘汰赛 False
        defending_champion: 卫冕冠军球队名（在淘汰赛中胜率 ×penalty）
        defending_penalty: 卫冕冠军每场淘汰赛胜率乘数 (0.95 = -5%)
        apply_critical_adj: 是否在淘汰赛应用 critical_node_adjustments
                            （仅淘汰赛生效；小组赛不应用）
        elo_shifts: {team_name: delta_elo} 队级 AI 加权 Elo 加成（阶段 1/2 事件级）。
                    None 时行为不变（默认）；非 None 时把每队 elo 加上 ΔE 再算胜率。
        match_shifts: {(team_a_name, team_b_name): (delta_a, delta_b)} 比赛级 ΔElo（阶段 2）。
                    支持反向查找 (B,A) → (delta_b, delta_a)；None 时不影响。
                    用于按场注入 H2H/referee/weather/lineups 4 项调整。
    """
    if rng is None:
        rng = np.random.default_rng()

    elo_a = team_a_data["elo"]
    elo_b = team_b_data["elo"]
    a_name = team_a_data.get("name")
    b_name = team_b_data.get("name")
    # AI 加权 Elo 偏移（阶段 1/2）：队级
    # elo_shifts is None 时完全不影响 → 现系统行为保留
    if elo_shifts:
        elo_a = elo_a + elo_shifts.get(a_name, 0.0)
        elo_b = elo_b + elo_shifts.get(b_name, 0.0)
    # 比赛级（阶段 2）：每对 (A,B) 单独叠加。支持反向查找。
    if match_shifts and a_name and b_name:
        ms = match_shifts.get((a_name, b_name))
        if ms is None:
            ms_rev = match_shifts.get((b_name, a_name))
            if ms_rev is not None:
                ms = (ms_rev[1], ms_rev[0])
        if ms is not None:
            elo_a += ms[0]
            elo_b += ms[1]

    p = match_probabilities(elo_a, elo_b)
    r = rng.random()
    
    if allow_draw:
        if r < p["p_win_a"]:
            return team_a_data["name"]
        elif r < p["p_win_a"] + p["p_draw"]:
            return "DRAW"
        else:
            return team_b_data["name"]
    else:
        # 淘汰赛：平局用 Elo 加权打成胜负
        # 把平局概率按比例分配给双方
        p_a_adj = p["p_win_a"] + p["p_draw"] * (p["p_win_a"] / (p["p_win_a"] + p["p_win_b"] + 1e-9))

        # 卫冕魔咒：卫冕冠军在淘汰赛胜率乘以 penalty
        if defending_champion and defending_penalty < 1.0:
            if team_a_data.get("name") == defending_champion:
                p_a_adj *= defending_penalty
            elif team_b_data.get("name") == defending_champion:
                # 对手是卫冕冠军 → 卫冕冠军胜率(1-p_a_adj)×penalty → A 的胜率提升
                p_b_adj = (1 - p_a_adj) * defending_penalty
                p_a_adj = 1 - p_b_adj
        
        # 方案 A2：关键节点 5-Agent 微调（任何对阵出现该对都生效）
        if apply_critical_adj:
            lookup = _load_critical_lookup()
            if lookup:
                a_name = team_a_data.get("name")
                b_name = team_b_data.get("name")
                # 查表：按字典序较小者作 key
                if a_name and b_name:
                    if a_name <= b_name:
                        key = (a_name, b_name)
                        adj_for_a = lookup.get(key, 0.0)
                    else:
                        key = (b_name, a_name)
                        adj_for_a = -lookup.get(key, 0.0)
                    p_a_adj += adj_for_a
                    # clip 到 [0.01, 0.99]
                    p_a_adj = max(0.01, min(0.99, p_a_adj))

        if r < p_a_adj:
            return team_a_data["name"]
        else:
            return team_b_data["name"]


def simulate_group(group_teams: list, teams_data: dict, rng: np.random.Generator,
                    elo_shifts: dict = None,
                    match_shifts: dict = None) -> list:
    """
    模拟一个 4 队小组的全 6 场比赛，返回排名（含积分、净胜球估算）

    Args:
        elo_shifts: 透传给 simulate_match（AI 加权队级 ΔE）
        match_shifts: 透传给 simulate_match（AI 加权比赛级 ΔE，按 (A,B) 索引）
    """
    points = {t: 0 for t in group_teams}
    goals_for = {t: 0 for t in group_teams}
    goals_against = {t: 0 for t in group_teams}
    
    # 6 场比赛
    matches = [
        (group_teams[0], group_teams[1]),
        (group_teams[2], group_teams[3]),
        (group_teams[0], group_teams[2]),
        (group_teams[1], group_teams[3]),
        (group_teams[0], group_teams[3]),
        (group_teams[1], group_teams[2]),
    ]
    
    for team_a, team_b in matches:
        ta = {"name": team_a, **teams_data[team_a]}
        tb = {"name": team_b, **teams_data[team_b]}
        result = simulate_match(ta, tb, allow_draw=True, rng=rng,
                                 elo_shifts=elo_shifts, match_shifts=match_shifts)
        
        # 用 Elo 期望差近似估算比分 → 用于净胜球计算
        elo_diff = ta["elo"] - tb["elo"]
        # 期望进球差约 = elo_diff / 200（经验值）
        diff = elo_diff / 200.0
        
        if result == team_a:
            points[team_a] += 3
            ga = max(1, int(rng.normal(1.5 + diff/2, 0.7)))
            gb = max(0, int(rng.normal(0.8 - diff/2, 0.5)))
            if gb >= ga:
                gb = ga - 1
            goals_for[team_a] += ga
            goals_against[team_a] += gb
            goals_for[team_b] += gb
            goals_against[team_b] += ga
        elif result == team_b:
            points[team_b] += 3
            gb = max(1, int(rng.normal(1.5 - diff/2, 0.7)))
            ga = max(0, int(rng.normal(0.8 + diff/2, 0.5)))
            if ga >= gb:
                ga = gb - 1
            goals_for[team_a] += ga
            goals_against[team_a] += gb
            goals_for[team_b] += gb
            goals_against[team_b] += ga
        else:
            # 平局
            points[team_a] += 1
            points[team_b] += 1
            g = max(0, int(rng.normal(1.0, 0.6)))
            goals_for[team_a] += g
            goals_against[team_a] += g
            goals_for[team_b] += g
            goals_against[team_b] += g
    
    # 排名：积分 → 净胜球 → 进球数 → Elo
    standings = sorted(group_teams, key=lambda t: (
        -points[t],
        -(goals_for[t] - goals_against[t]),
        -goals_for[t],
        -teams_data[t]["elo"]
    ))
    
    return [(team, points[team], goals_for[team] - goals_against[team], goals_for[team]) for team in standings]


def select_best_thirds(third_place_teams: list, count: int = 8) -> list:
    """
    从 12 个小组第 3 名中选出 8 个最佳（积分 → 净胜球 → 进球 → Elo）
    third_place_teams 格式：[(team, points, goal_diff, goals_for, group), ...]
    """
    sorted_thirds = sorted(third_place_teams, key=lambda x: (
        -x[1], -x[2], -x[3]
    ))
    return [t[0] for t in sorted_thirds[:count]]


def build_round_of_32_matchups(group_winners: dict, group_runners_up: dict, best_thirds: list) -> list:
    """
    根据 FIFA 2026 规则构建 32 强对阵
    
    简化版：
    - 12 小组冠军 + 12 小组第 2 + 8 最佳第 3 = 32 队
    - 同组球队不在 32 强相遇
    - 上半区 (Pathway 1): E F H I 组冠军 + 部分对手
    - 下半区 (Pathway 2): C J K L 组冠军 + 部分对手
    - 这里采用近似对阵：组 N 头名 vs 组 (N+6) 次名 / 第 3
    """
    # 简化映射：12 个小组冠军 + 12 个小组次名 + 8 最佳第 3
    # 12 个 W (winners) + 8 W vs 第 3 + 4 W vs 次名（来自远端组）
    # 这里采用 FIFA 公布的固定 bracket 简化对应
    
    # 12 组 winners 排序
    winners = list(group_winners.values())
    runners = list(group_runners_up.values())
    thirds = best_thirds[:8]
    
    # 32 强 = 16 场对阵
    # 简化映射（接近 FIFA 实际签表）：
    # Match 1:  A1 vs (best 3rd from C/D/E/F)
    # Match 2:  C1 vs F2
    # Match 3:  D1 vs (best 3rd from B/E/F/I)
    # ... 等等
    # 此处采用配对算法保证：1) 同组不相遇 2) 高排名得弱对手
    
    all_qualifiers = winners + runners + thirds  # 32 队
    
    # 简化版：让 winners 对阵 runners + thirds，runners 对阵 thirds + winners
    # 使用 FIFA 2026 真实签表的简化对应
    matchups = [
        # Pathway 1 (Upper Half)
        (group_winners["H"], thirds[0] if len(thirds) > 0 else runners[0]),  # Spain vs ???
        (group_runners_up["E"], group_runners_up["I"]),  # E2 vs I2
        (group_winners["F"], thirds[1] if len(thirds) > 1 else runners[1]),  # Netherlands vs ???
        (group_runners_up["H"], group_runners_up["F"]),  # H2 vs F2 (same half)
        (group_winners["E"], thirds[2] if len(thirds) > 2 else runners[2]),  # Germany vs ???
        (group_runners_up["B"], group_runners_up["A"]),  # B2 vs A2
        (group_winners["I"], thirds[3] if len(thirds) > 3 else runners[3]),  # France vs ???
        (group_winners["B"], group_runners_up["G"]),  # B1 vs G2
        # Pathway 2 (Lower Half)
        (group_winners["A"], thirds[4] if len(thirds) > 4 else runners[4]),  # Mexico vs ???
        (group_winners["G"], group_runners_up["D"]),  # G1 vs D2
        (group_winners["D"], thirds[5] if len(thirds) > 5 else runners[5]),  # USA vs ???
        (group_winners["L"], group_runners_up["K"]),  # L1 vs K2
        (group_winners["C"], thirds[6] if len(thirds) > 6 else runners[6]),  # Brazil vs ???
        (group_winners["J"], group_runners_up["L"]),  # J1 vs L2 (Argentina vs Croatia?)
        (group_winners["K"], thirds[7] if len(thirds) > 7 else runners[7]),  # Portugal vs ???
        (group_runners_up["C"], group_runners_up["J"]),  # C2 vs J2
    ]
    
    return matchups


def simulate_tournament(teams_data: dict, groups: dict, rng: np.random.Generator,
                         bracket: dict = None,
                         return_bracket: bool = False,
                         elo_shifts: dict = None,
                         match_shifts: dict = None) -> dict:
    """
    模拟一届完整世界杯（使用真实 FIFA 2026 bracket）
    
    Args:
        return_bracket: True 时同时返回完整对阵明细 dict（用于"最可能剧本"分析）
    
    Returns:
        如果 return_bracket=False: {球队: 最终阶段}
        如果 return_bracket=True : (final_stages, bracket_record)
            bracket_record = {
                "r32": [(match_id, team_a, team_b, winner), ...],
                "r16": [(r16_id, team_a, team_b, winner), ...],
                "qf":  [(qf_id, ..., winner), ...],
                "sf":  [(sf_id, ..., winner), ...],
                "final": (team_a, team_b, winner),
            }
    """
    if bracket is None:
        bracket = load_bracket()
    
    # ============ 1. 小组赛 ============
    group_winners = {}      # {group_id: team}
    group_runners_up = {}   # {group_id: team}
    third_place = []        # [(team, pts, gd, gf, group_id), ...]
    group_standings_record = {}  # {group_id: [(team, rank, points, gd, gf), ...]}
    
    all_teams = set()
    for g_id, g_teams in groups.items():
        for t in g_teams:
            all_teams.add(t)
        standings = simulate_group(g_teams, teams_data, rng,
                                    elo_shifts=elo_shifts, match_shifts=match_shifts)
        group_winners[g_id] = standings[0][0]
        group_runners_up[g_id] = standings[1][0]
        third_place.append((standings[2][0], standings[2][1], standings[2][2], standings[2][3], g_id))
        # 记录完整排名（用于小组排名概率分布统计）
        group_standings_record[g_id] = [
            (team, rank, pts, gd, gf)
            for rank, (team, pts, gd, gf) in enumerate(standings, 1)
        ]
        # 第 4 名直接淘汰
    
    # ============ 2. 选 8 个最佳第三名 ============
    best_thirds_data = sorted(third_place, key=lambda x: (-x[1], -x[2], -x[3]))[:8]
    # third_places_dict: {group_id: team_name}
    third_places_dict = {item[4]: item[0] for item in best_thirds_data}
    
    # ============ 3. 真实 FIFA bracket: 第三名分配 + R32 对阵 ============
    third_assignments = assign_third_places(third_places_dict, bracket, rng)
    matchups_r32 = build_real_round_of_32(group_winners, group_runners_up, third_assignments, bracket)
    # matchups_r32: [(team_a, team_b, match_id), ...]
    
    # 记录 match_id → winner（用于树形推演）
    match_winners = {}  # {match_id: team_name}
    advancers = []      # R32 胜者
    r32_records = []    # [(match_id, team_a, team_b, winner), ...]
    
    for ta_name, tb_name, match_id in matchups_r32:
        ta = {"name": ta_name, **teams_data[ta_name]}
        tb = {"name": tb_name, **teams_data[tb_name]}
        winner = simulate_match(ta, tb, allow_draw=False, rng=rng,
                                 elo_shifts=elo_shifts, match_shifts=match_shifts)
        match_winners[match_id] = winner
        advancers.append(winner)
        r32_records.append((match_id, ta_name, tb_name, winner))
    
    # ============ 4. R16: 按 bracket._round_of_16_pairings 配对 ============
    r16_winners = []
    r16_match_winners = {}  # {r16_match_id: winner}
    r16_records = []
    for pairing in bracket["_round_of_16_pairings"]:
        r16_id = pairing["r16_match"]
        ma_id = pairing["winner_of_match_a"]
        mb_id = pairing["winner_of_match_b"]
        ta_name = match_winners.get(ma_id)
        tb_name = match_winners.get(mb_id)
        if not ta_name or not tb_name:
            continue
        ta = {"name": ta_name, **teams_data[ta_name]}
        tb = {"name": tb_name, **teams_data[tb_name]}
        winner = simulate_match(ta, tb, allow_draw=False, rng=rng,
                                 elo_shifts=elo_shifts, match_shifts=match_shifts)
        r16_match_winners[r16_id] = winner
        r16_winners.append(winner)
        r16_records.append((r16_id, ta_name, tb_name, winner))
    
    # ============ 5. 1/4 决赛: 按 bracket._quarterfinal_pairings 配对 ============
    qf_winners = []
    qf_match_winners = {}  # {qf_match_id: winner}
    qf_records = []
    for pairing in bracket["_quarterfinal_pairings"]:
        qf_id = pairing["qf_match"]
        ra_id = pairing["winner_of_r16_a"]
        rb_id = pairing["winner_of_r16_b"]
        ta_name = r16_match_winners.get(ra_id)
        tb_name = r16_match_winners.get(rb_id)
        if not ta_name or not tb_name:
            continue
        ta = {"name": ta_name, **teams_data[ta_name]}
        tb = {"name": tb_name, **teams_data[tb_name]}
        winner = simulate_match(ta, tb, allow_draw=False, rng=rng,
                                 elo_shifts=elo_shifts, match_shifts=match_shifts)
        qf_match_winners[qf_id] = winner
        qf_winners.append(winner)
        qf_records.append((qf_id, ta_name, tb_name, winner))
    
    # ============ 6. 半决赛: 按 bracket._semifinal_pairings 配对（强制半区分隔！）============
    sf_winners = []
    sf_match_winners = {}  # {sf_match_id: winner}
    sf_records = []
    for pairing in bracket["_semifinal_pairings"]:
        sf_id = pairing["sf_match"]
        qa_id = pairing["winner_of_qf_a"]
        qb_id = pairing["winner_of_qf_b"]
        ta_name = qf_match_winners.get(qa_id)
        tb_name = qf_match_winners.get(qb_id)
        if not ta_name or not tb_name:
            continue
        ta = {"name": ta_name, **teams_data[ta_name]}
        tb = {"name": tb_name, **teams_data[tb_name]}
        winner = simulate_match(ta, tb, allow_draw=False, rng=rng,
                                 elo_shifts=elo_shifts, match_shifts=match_shifts)
        sf_match_winners[sf_id] = winner
        sf_winners.append(winner)
        sf_records.append((sf_id, ta_name, tb_name, winner))
    
    # ============ 7. 决赛 ============
    if len(sf_winners) < 2:
        # 兜底（不应发生）
        if return_bracket:
            return {t: "GROUP" for t in all_teams}, None
        return {t: "GROUP" for t in all_teams}
    
    final_pair = bracket["_final_pairing"]
    finalists = [
        sf_match_winners.get(final_pair["winner_of_sf_a"]),
        sf_match_winners.get(final_pair["winner_of_sf_b"]),
    ]
    ta = {"name": finalists[0], **teams_data[finalists[0]]}
    tb = {"name": finalists[1], **teams_data[finalists[1]]}
    champion = simulate_match(ta, tb, allow_draw=False, rng=rng,
                                elo_shifts=elo_shifts, match_shifts=match_shifts)
    runner_up = finalists[0] if champion == finalists[1] else finalists[1]
    
    # ============ 8. 阶段标记（按最深进入轮次）============
    final_stages = {t: "GROUP" for t in all_teams}
    for t in advancers:
        final_stages[t] = "R32_WIN"     # 进入 R16
    for t in r16_winners:
        final_stages[t] = "R16_WIN"     # 进入 QF
    for t in qf_winners:
        final_stages[t] = "QF_WIN"      # 进入 SF
    for t in sf_winners:
        final_stages[t] = "SF_WIN"      # 进入决赛
    final_stages[runner_up] = "FINAL_LOSE"
    final_stages[champion] = "CHAMPION"
    
    if return_bracket:
        bracket_record = {
            "r32": r32_records,
            "r16": r16_records,
            "qf":  qf_records,
            "sf":  sf_records,
            "final": (finalists[0], finalists[1], champion),
            "group_standings": group_standings_record,   # {group: [(team, rank, pts, gd, gf), ...]}
        }
        return final_stages, bracket_record
    return final_stages


def run_monte_carlo(n_simulations: int = 100000, seed: int = 42, verbose: bool = True,
                     return_finals: bool = False,
                     elo_shifts: dict = None,
                     match_shifts: dict = None) -> dict:
    """
    执行 N 次蒙特卡洛模拟（使用真实 FIFA 2026 bracket）
    返回每队在各阶段的概率
    
    Args:
        return_finals: True 时额外返回所有决赛对阵的频次（用于热力图）
        elo_shifts: {team: delta_elo} 队级 AI 加权 Elo 偏移（阶段 1/2 事件级）。
                    None 时行为不变（默认）。
        match_shifts: {(team_a, team_b): (delta_a, delta_b)} 比赛级 AI 加权
                    Elo 偏移（阶段 2）。None 时不影响。
    """
    teams = load_teams()
    groups = load_groups()
    bracket = load_bracket()  # 加载一次，复用
    
    qualified_teams = {name: data for name, data in teams.items() if data.get("group") != "_"}
    
    rng = np.random.default_rng(seed)
    
    stage_counts = {t: {"GROUP": 0, "R32_WIN": 0, "R16_WIN": 0, "QF_WIN": 0, 
                        "SF_WIN": 0, "FINAL_LOSE": 0, "CHAMPION": 0} 
                    for t in qualified_teams}
    
    # 决赛对阵记录：{frozenset({A, B}): count}
    final_matchups = {}
    
    for sim_idx in range(n_simulations):
        if verbose and (sim_idx + 1) % max(1, n_simulations // 10) == 0:
            print(f"  Progress: {sim_idx + 1}/{n_simulations} ({(sim_idx+1)/n_simulations*100:.0f}%)")
        
        result = simulate_tournament(qualified_teams, groups, rng, bracket=bracket,
                                       elo_shifts=elo_shifts,
                                       match_shifts=match_shifts)
        for team, stage in result.items():
            if team in stage_counts:
                stage_counts[team][stage] += 1
        
        # 收集决赛双方
        if return_finals:
            finalists = [t for t, s in result.items() 
                          if s in ("CHAMPION", "FINAL_LOSE")]
            if len(finalists) == 2:
                key = frozenset(finalists)
                final_matchups[key] = final_matchups.get(key, 0) + 1
    
    # 计算概率
    probs = {}
    for team, counts in stage_counts.items():
        total = sum(counts.values())
        if total == 0:
            continue
        # 累积概率：进入某阶段 ≥ X
        # 阶段标记说明：
        #   R32_WIN = 赢了 R32，至少进入 R16
        #   R16_WIN = 赢了 R16，至少进入 QF（8 强）
        #   QF_WIN  = 赢了 QF，至少进入 SF（4 强）
        #   SF_WIN  = 赢了 SF，至少进入 Final
        #   FINAL_LOSE = 决赛输
        #   CHAMPION = 冠军
        # 注意：每个 team 在一届模拟中只有一个最终标记（最深进入的轮次）
        deeper = lambda *stages: sum(counts[s] for s in stages)
        
        p_champion   = counts["CHAMPION"] / total
        p_final      = deeper("FINAL_LOSE", "CHAMPION") / total                      # 进入决赛（亚军 + 冠军）
        p_advance_sf = deeper("QF_WIN", "SF_WIN", "FINAL_LOSE", "CHAMPION") / total  # 进入半决赛（4 强）
        p_advance_qf = deeper("R16_WIN", "QF_WIN", "SF_WIN", "FINAL_LOSE", "CHAMPION") / total  # 进入 8 强
        p_advance_r16 = deeper("R32_WIN", "R16_WIN", "QF_WIN", "SF_WIN", "FINAL_LOSE", "CHAMPION") / total  # 进入 16 强
        # 进入 R32 的概率 = 进 R16 概率 + R32 输的概率
        # 但当前阶段标记中 R32 输 = "GROUP" 之外的标记... 实际上 GROUP 包含小组出局 + R32 输
        # 简化：用 p_advance_r16 作为代理（进入 R16 即至少在 R32 赢了一场）
        # 准确版需要新增 "R32_LOSE" 标记，暂用 p_advance_r16 + 估算
        p_advance_r32 = p_advance_r16  # 至少打了 R32 = 进入 R16 的概率
        
        probs[team] = {
            "elo": qualified_teams[team]["elo"],
            "market_implied": qualified_teams[team]["market_implied"],
            "champion": round(p_champion, 4),
            "final": round(p_final, 4),
            "semifinal": round(p_advance_sf, 4),
            "quarterfinal": round(p_advance_qf, 4),
            "round_of_16": round(p_advance_r16, 4),
            "round_of_32": round(p_advance_r32, 4),
            "n_simulations": total
        }
    
    # ===== 参考规范 §9.6.2 表 9.13 六桶概率校准（Layer 5）=====
    # 仅校准淘汰赛长尾字段（champion/final/semifinal），
    # 并保留 raw_<field> 供审计；champion 总和重新归一化到 1。
    probs = calibrate_all_reference(probs)

    # 决赛对阵处理
    if return_finals:
        finals_list = []
        for matchup, count in final_matchups.items():
            teams_pair = sorted(list(matchup))
            finals_list.append({
                "team_a": teams_pair[0],
                "team_b": teams_pair[1],
                "probability": round(count / n_simulations, 5),
                "count": count
            })
        finals_list.sort(key=lambda x: -x["probability"])
        return probs, finals_list
    
    return probs


# ===== 自测 =====
if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
    print(f"=== 蒙特卡洛模拟 N={n} ===")
    probs = run_monte_carlo(n_simulations=n, verbose=True)
    
    # 按夺冠概率排序
    sorted_teams = sorted(probs.items(), key=lambda x: -x[1]["champion"])
    
    print(f"\n{'Rank':<5} {'Team':<18} {'Elo':<6} {'Champion':<10} {'Final':<8} {'SF':<7} {'QF':<7} {'Market':<8} {'Bias':<8}")
    print("-" * 95)
    for rank, (team, p) in enumerate(sorted_teams[:16], 1):
        bias = (p["champion"] - p["market_implied"]) * 100
        print(f"{rank:<5} {team:<18} {p['elo']:<6} {p['champion']*100:>6.2f}%   {p['final']*100:>5.1f}%  {p['semifinal']*100:>5.1f}%  {p['quarterfinal']*100:>5.1f}%  {p['market_implied']*100:>5.1f}%   {bias:>+6.1f}pp")
    
    save_output(f"mc_simulation_n{n}.json", probs)
    print(f"\n已保存到: data/outputs/mc_simulation_n{n}.json")
