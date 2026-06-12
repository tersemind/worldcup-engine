"""
Elo 评级引擎
公式：P(A wins) = 1 / (1 + 10^((Elo_B - Elo_A) / s))
"""
import math


def expected_score(elo_a: float, elo_b: float, s: float = 600.0) -> float:
    """
    Elo 期望胜率（不含平局）
    参考方法论 s=600（国家队赛事），俱乐部赛事常用 s=400
    """
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / s))


def match_probabilities(elo_a: float, elo_b: float, s: float = 600.0,
                         draw_factor: float = 0.27) -> dict:
    """
    返回三结果概率：A 胜 / 平 / B 胜
    
    参考 Dixon-Coles 实证：低实力差时平局概率约 27-32%
    随着实力差扩大，平局概率衰减
    """
    base_a = expected_score(elo_a, elo_b, s)
    base_b = 1 - base_a
    
    # 平局概率 = draw_factor * 4 * P_a * P_b（实力越接近平局越多）
    # 极端值：等实力时 draw=0.27；100% vs 0% 时 draw≈0
    p_draw = draw_factor * 4 * base_a * base_b
    
    # 重新归一化
    p_a = base_a * (1 - p_draw)
    p_b = base_b * (1 - p_draw)
    
    return {
        "p_win_a": round(p_a, 4),
        "p_draw": round(p_draw, 4),
        "p_win_b": round(p_b, 4)
    }


def update_elo(elo_a: float, elo_b: float, result: float,
               k: float = 60.0, s: float = 600.0) -> tuple:
    """
    比赛后更新 Elo
    result: A 视角的实际结果 (胜=1, 平=0.5, 负=0)
    k: 赛事重要性系数
        友谊赛=20, 预选赛=30, 大洲杯=40, 世界杯小组=50, 世界杯淘汰赛=60
    """
    expected = expected_score(elo_a, elo_b, s)
    new_elo_a = elo_a + k * (result - expected)
    new_elo_b = elo_b + k * ((1 - result) - (1 - expected))
    return new_elo_a, new_elo_b


def get_k_value(stage: str) -> float:
    """根据赛事阶段返回 K 值（参考方法论）"""
    k_table = {
        "friendly": 20,
        "qualifier": 30,
        "continental_cup": 40,
        "wc_group": 50,
        "wc_knockout": 60,
    }
    return k_table.get(stage, 30)


# ===== 自测 =====
if __name__ == "__main__":
    # Spain (2155) vs France (2105)
    p = match_probabilities(2155, 2105, s=600)
    print(f"Spain vs France: {p}")
    
    # Spain (2155) vs Curacao (1480)
    p = match_probabilities(2155, 1480, s=600)
    print(f"Spain vs Curacao: {p}")
    
    # Update test: Spain beats France
    new_a, new_b = update_elo(2155, 2105, result=1.0, k=60)
    print(f"Spain Elo: 2155 -> {new_a:.1f}")
    print(f"France Elo: 2105 -> {new_b:.1f}")
