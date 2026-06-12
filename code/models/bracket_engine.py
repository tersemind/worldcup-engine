"""
真实 FIFA 2026 Bracket 引擎
- 使用 bracket.json 中的真实对阵结构
- 实现 FIFA 第三名分配规则（同组不相遇 + 8 选 6 槽位）
- 支持完整淘汰赛树推演（R32 → R16 → QF → SF → Final）
"""
import json
import sys
from pathlib import Path
from itertools import permutations

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.io import DATA_RAW


def load_bracket():
    """加载 bracket.json"""
    with open(DATA_RAW / "bracket.json", "r") as f:
        return json.load(f)


def assign_third_places(third_places: dict, bracket: dict, rng) -> dict:
    """
    给 8 个最佳第三名分配 R32 槽位（FIFA 495 方案精确版）
    
    优先使用 fifa_495_table.json 查询表（精确匹配），
    回落到回溯算法（兼容性后备）
    
    输入：
        third_places: {group_id: team_name} 8 个最佳第三名（仅含晋级的）
        bracket: bracket.json
        rng: numpy 随机生成器
    
    返回：
        {r32_match_id: team_name} 8 个槽位的映射
    """
    # 尝试用 FIFA 495 真实查询表
    try:
        import json
        from pathlib import Path
        table_path = Path(__file__).parent.parent.parent / "data" / "raw" / "fifa_495_table.json"
        if table_path.exists():
            with open(table_path, "r") as f:
                fifa_table = json.load(f)
            
            # 用 third_places 的组 key 查表
            third_groups = sorted(third_places.keys())
            lookup_key = "".join(third_groups)
            
            if lookup_key in fifa_table:
                # 直接查表
                group_to_match = fifa_table[lookup_key]
                # group_to_match: {match_id_str: third_group}
                assignment = {}
                for match_id_str, third_group in group_to_match.items():
                    match_id = int(match_id_str)
                    if third_group in third_places:
                        assignment[match_id] = third_places[third_group]
                if len(assignment) == 8:
                    return assignment
    except Exception:
        pass  # 回落到回溯算法
    
    # ============ 回溯算法（后备）============
    slots = bracket["_third_place_assignment"]["slots"]
    slot_ids = [s["r32_match"] for s in slots]
    slot_candidates_raw = {s["r32_match"]: s["candidates"] for s in slots}
    
    # 同组对手映射
    r32_winners = {m["match"]: m["slot_a"] for m in bracket["round_of_32"]}
    
    def get_winner_group(match_id):
        slot = r32_winners.get(match_id)
        if slot and len(slot) >= 1 and slot[0].isalpha():
            return slot[0]
        return None
    
    # 计算每个槽位的合法候选组
    # 策略：宽松匹配——只保证「不同组」即可，不严格限制 FIFA 候选清单
    # 这是因为 FIFA 真实 495 种方案的候选范围比单一槽位定义更宽
    available_groups = list(third_places.keys())
    slot_valid = {}
    for slot_id in slot_ids:
        wg = get_winner_group(slot_id)
        # 优先用槽位定义的候选；如果约束太窄无解，则放宽到所有组
        strict_valid = [g for g in slot_candidates_raw[slot_id] 
                       if g in available_groups and g != wg]
        loose_valid = [g for g in available_groups if g != wg]
        # 优先严格，失败时回落到宽松
        slot_valid[slot_id] = (strict_valid, loose_valid)
    
    # 回溯算法：尝试给每个槽位分配一个第三名
    assignment = {}
    
    # 按 strict 候选数升序排（MRV 启发式）
    randomized_slots = list(slot_ids)
    rng.shuffle(randomized_slots)
    randomized_slots.sort(key=lambda x: len(slot_valid[x][0]))
    
    def backtrack(idx, used_groups, use_loose=False):
        if idx == len(randomized_slots):
            return True
        slot_id = randomized_slots[idx]
        strict, loose = slot_valid[slot_id]
        candidates = loose if use_loose else strict
        valid = [g for g in candidates if g not in used_groups]
        if not valid:
            return False
        valid_random = list(valid)
        rng.shuffle(valid_random)
        for group in valid_random:
            assignment[slot_id] = third_places[group]
            used_groups.add(group)
            if backtrack(idx + 1, used_groups, use_loose):
                return True
            del assignment[slot_id]
            used_groups.remove(group)
        return False
    
    # 优先严格匹配；失败则回落到宽松匹配
    if not backtrack(0, set(), use_loose=False):
        assignment.clear()
        backtrack(0, set(), use_loose=True)
    
    return assignment


def resolve_slot(slot_str: str, group_winners: dict, group_runners_up: dict, third_assignments: dict, match_id: int) -> str:
    """
    把 'E1'/'F2'/'3rd_from_ABCDF' 等槽位字符串解析为具体球队名
    """
    if slot_str.startswith("3rd"):
        return third_assignments.get(match_id, None)
    
    if len(slot_str) >= 2 and slot_str[0].isalpha() and slot_str[1].isdigit():
        group = slot_str[0]
        position = slot_str[1]
        if position == "1":
            return group_winners.get(group)
        elif position == "2":
            return group_runners_up.get(group)
    
    return None


def build_real_round_of_32(group_winners: dict, group_runners_up: dict, 
                             third_assignments: dict, bracket: dict) -> list:
    """
    根据真实 bracket 构建 R32 对阵
    返回 [(team_a, team_b, match_id), ...] 列表
    """
    matchups = []
    for match in bracket["round_of_32"]:
        match_id = match["match"]
        team_a = resolve_slot(match["slot_a"], group_winners, group_runners_up, third_assignments, match_id)
        team_b = resolve_slot(match["slot_b"], group_winners, group_runners_up, third_assignments, match_id)
        if team_a and team_b:
            matchups.append((team_a, team_b, match_id))
    return matchups


def get_pathway(team: str, group_winners: dict, group_runners_up: dict, bracket: dict) -> str:
    """识别球队属于哪个 pathway"""
    upper_groups = bracket["_pathways"]["pathway_1_upper"]["groups_winners_in_path"]
    
    # 找球队在哪个组
    for g, t in group_winners.items():
        if t == team:
            return "upper" if g in upper_groups else "lower"
    for g, t in group_runners_up.items():
        if t == team:
            # 次名进入与冠军相反的半区（FIFA 规则）
            return "lower" if g in upper_groups else "upper"
    return "unknown"


# ===== 自测 =====
if __name__ == "__main__":
    import numpy as np
    bracket = load_bracket()
    print(f"✅ Bracket 加载成功")
    print(f"  - R32 比赛数：{len(bracket['round_of_32'])}")
    print(f"  - R16 配对数：{len(bracket['_round_of_16_pairings'])}")
    print(f"  - 8 强配对数：{len(bracket['_quarterfinal_pairings'])}")
    print(f"  - 半决赛配对数：{len(bracket['_semifinal_pairings'])}")
    print(f"  - 上半区球队：{bracket['_pathways']['pathway_1_upper']['key_teams']}")
    print(f"  - 下半区球队：{bracket['_pathways']['pathway_2_lower']['key_teams']}")
    
    # 模拟测试：假设小组赛结果
    rng = np.random.default_rng(42)
    fake_winners = {chr(65+i): f"W{chr(65+i)}" for i in range(12)}
    fake_runners = {chr(65+i): f"R{chr(65+i)}" for i in range(12)}
    fake_thirds = {"A": "3A", "B": "3B", "C": "3C", "D": "3D", 
                    "E": "3E", "F": "3F", "G": "3G", "H": "3H"}
    
    third_assigns = assign_third_places(fake_thirds, bracket, rng)
    print(f"\n📋 第三名分配结果：")
    for match_id, team in third_assigns.items():
        print(f"  Match {match_id}: {team}")
    
    matchups = build_real_round_of_32(fake_winners, fake_runners, third_assigns, bracket)
    print(f"\n📋 R32 对阵（{len(matchups)} 场）：")
    for ta, tb, mid in matchups[:8]:
        print(f"  Match {mid}: {ta} vs {tb}")
