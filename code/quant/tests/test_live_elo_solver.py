"""live_elo_solver v1 单元测试。

覆盖：
  - 各分项独立贡献正确性
  - 字段缺失/类型异常的 fail-soft
  - time_decay 边界
  - 终场 phase=FT 的特殊处理
  - team_elos 注入路径（含异常）
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from quant.live_elo_solver import (
    solve, time_decay,
    SCORE_DIFF_PP_PER_GOAL, RED_CARD_PP_EACH, ELO_PER_PP,
    YELLOW_CARD_THRESHOLD, YELLOW_CARD_PP_EACH, SUB_FATIGUE_PP_EACH,
    TIME_DECAY_BASE, TIME_DECAY_END,
)


def _approx(a, b, tol=0.01):
    return abs(a - b) <= tol


# ─────────── time_decay ───────────

def test_time_decay_start():
    assert _approx(time_decay(0), TIME_DECAY_BASE)


def test_time_decay_end():
    assert _approx(time_decay(90), TIME_DECAY_END)


def test_time_decay_mid():
    # 45min → 中间值
    assert _approx(time_decay(45), (TIME_DECAY_BASE + TIME_DECAY_END) / 2)


def test_time_decay_overtime():
    # OT/PK > 90min 按 1.0 处理
    assert _approx(time_decay(120), TIME_DECAY_END)


def test_time_decay_none():
    # 缺失 → 极弱衰减
    assert _approx(time_decay(None), TIME_DECAY_BASE)


# ─────────── 分项独立贡献 ───────────

def test_pure_score_diff_at_full_time():
    # 1-0 @ 90min, 无卡无换 → ΔPP = 1 × 12 × 1.0 = 12
    state = {
        "team_a": "A", "team_b": "B",
        "score_a": 1, "score_b": 0,
        "elapsed_min": 90, "phase": "regular",
    }
    r = solve(state)
    assert _approx(r["components"]["score_diff_pp"], 12.0)
    assert _approx(r["components"]["red_pp"], 0.0)
    assert _approx(r["components"]["yellow_pp"], 0.0)
    assert _approx(r["components"]["sub_fatigue_pp"], 0.0)
    assert _approx(r["delta_total_pp"], 12.0)
    assert _approx(r["delta_elo_a"], 12.0 * ELO_PER_PP)
    assert _approx(r["delta_elo_b"], -12.0 * ELO_PER_PP)


def test_score_diff_clipped_at_3():
    # 5-0 → 应被 cap 到 3
    state = {"team_a": "A", "team_b": "B",
             "score_a": 5, "score_b": 0, "elapsed_min": 90, "phase": "regular"}
    r = solve(state)
    assert _approx(r["components"]["score_diff_pp"], 3 * SCORE_DIFF_PP_PER_GOAL * 1.0)


def test_red_card_only():
    # 0-0 + B 1红 → red_pp = +6（A 优势）
    state = {"team_a": "A", "team_b": "B",
             "score_a": 0, "score_b": 0, "elapsed_min": 60,
             "red_cards_a": 0, "red_cards_b": 1}
    r = solve(state)
    assert _approx(r["components"]["red_pp"], RED_CARD_PP_EACH)
    assert _approx(r["components"]["score_diff_pp"], 0.0)


def test_red_card_symmetry():
    # A 1红 → red_pp = -6
    state = {"team_a": "A", "team_b": "B",
             "score_a": 0, "score_b": 0, "elapsed_min": 60,
             "red_cards_a": 1, "red_cards_b": 0}
    r = solve(state)
    assert _approx(r["components"]["red_pp"], -RED_CARD_PP_EACH)


def test_yellow_below_threshold():
    # A 2 黄, B 2 黄 → 都不超阈值 → 0 贡献
    state = {"team_a": "A", "team_b": "B",
             "score_a": 0, "score_b": 0, "elapsed_min": 60,
             "yellow_cards_a": 2, "yellow_cards_b": 2}
    r = solve(state)
    assert _approx(r["components"]["yellow_pp"], 0.0)


def test_yellow_above_threshold():
    # A 4 黄(超1), B 5 黄(超2) → yellow_pp = (2-1) × 1 = +1（A 占优）
    state = {"team_a": "A", "team_b": "B",
             "score_a": 0, "score_b": 0, "elapsed_min": 60,
             "yellow_cards_a": 4, "yellow_cards_b": 5}
    r = solve(state)
    expected = ((5 - YELLOW_CARD_THRESHOLD) - (4 - YELLOW_CARD_THRESHOLD)) * YELLOW_CARD_PP_EACH
    assert _approx(r["components"]["yellow_pp"], expected)


def test_sub_fatigue():
    # A 用 1 换, B 用 5 换 → sub_fatigue = (5-1) × 1.5 = +6
    state = {"team_a": "A", "team_b": "B",
             "score_a": 0, "score_b": 0, "elapsed_min": 60,
             "subs_used_a": 1, "subs_used_b": 5}
    r = solve(state)
    assert _approx(r["components"]["sub_fatigue_pp"], 4 * SUB_FATIGUE_PP_EACH)


# ─────────── fail-soft（字段缺失/类型异常）───────────

def test_missing_score():
    # 无 score 字段 → 默认为 0-0
    state = {"team_a": "A", "team_b": "B", "elapsed_min": 60, "phase": "regular"}
    r = solve(state)
    assert _approx(r["components"]["score_diff_pp"], 0.0)
    assert "warnings" in r


def test_missing_elapsed_uses_45_fallback():
    # elapsed_min 缺失 → 用 45 兜底，应有 warning
    state = {"team_a": "A", "team_b": "B", "score_a": 1, "score_b": 0}
    r = solve(state)
    decay_45 = (TIME_DECAY_BASE + TIME_DECAY_END) / 2
    assert _approx(r["components"]["time_decay_factor"], decay_45)
    assert any("elapsed_min" in w for w in r["warnings"])


def test_missing_team_warns():
    # 无 team 字段 → warning 提示
    state = {"score_a": 1, "score_b": 0, "elapsed_min": 60}
    r = solve(state)
    assert any("team_a" in w or "team_b" in w for w in r["warnings"])


def test_invalid_int_field_falls_back_to_zero():
    # red_cards_a 是字符串 → 解析失败兜底 0
    state = {"team_a": "A", "team_b": "B",
             "score_a": 0, "score_b": 0, "elapsed_min": 60,
             "red_cards_a": "abc", "red_cards_b": 1}
    r = solve(state)
    # red_a=0 (兜底), red_b=1 → red_pp = +6
    assert _approx(r["components"]["red_pp"], RED_CARD_PP_EACH)


def test_phase_ft_uses_full_decay():
    # 1-0 @ 50min, phase=FT → 应按满衰减算（决定胜负已发生）
    state = {"team_a": "A", "team_b": "B",
             "score_a": 1, "score_b": 0, "elapsed_min": 50, "phase": "FT"}
    r = solve(state)
    # FT 时 score_diff_pp = 1 × 12 × 1.0 = 12
    assert _approx(r["components"]["score_diff_pp"], SCORE_DIFF_PP_PER_GOAL)


# ─────────── team_elos 注入路径 ───────────

def test_team_elos_injection():
    state = {"team_a": "Spain", "team_b": "Morocco",
             "score_a": 1, "score_b": 0, "elapsed_min": 90,
             "red_cards_a": 0, "red_cards_b": 1, "phase": "regular"}
    r = solve(state, team_elos={"Spain": 2050, "Morocco": 1820})
    assert "p_win_a" in r and "p_draw" in r and "p_win_b" in r
    # 概率合法
    assert 0 <= r["p_win_a"] <= 1
    assert 0 <= r["p_draw"] <= 1
    assert 0 <= r["p_win_b"] <= 1
    # 三概率和 ≈ 1
    assert _approx(r["p_win_a"] + r["p_draw"] + r["p_win_b"], 1.0, tol=0.01)
    # Spain 已经领先 + 红牌优势 → P(Spain) 应高于 50%
    assert r["p_win_a"] > 0.5
    # ΔE 注入正确
    assert r["elo_a_live"] == round(2050 + r["delta_elo_a"], 1)
    assert r["elo_b_live"] == round(1820 + r["delta_elo_b"], 1)


def test_team_elos_missing_team_warns():
    # 提供 team_elos 但缺其中一支
    state = {"team_a": "X", "team_b": "Y",
             "score_a": 1, "score_b": 0, "elapsed_min": 60}
    r = solve(state, team_elos={"X": 2000})  # Y 缺
    assert "p_win_a" not in r
    assert any("baseline elo" in w for w in r["warnings"])


# ─────────── 综合场景 ───────────

def test_realistic_spain_morocco():
    # Spain 1-0 Morocco @ 67min, Morocco 1红牌
    state = {"team_a": "Spain", "team_b": "Morocco",
             "score_a": 1, "score_b": 0, "elapsed_min": 67,
             "red_cards_a": 0, "red_cards_b": 1,
             "yellow_cards_a": 2, "yellow_cards_b": 3,
             "subs_used_a": 2, "subs_used_b": 3,
             "phase": "regular"}
    r = solve(state, team_elos={"Spain": 2050, "Morocco": 1820})
    # 各分项正向
    assert r["components"]["score_diff_pp"] > 0
    assert r["components"]["red_pp"] > 0
    assert r["components"]["sub_fatigue_pp"] > 0
    # 最终 P(Spain 胜) 接近 80%
    assert r["p_win_a"] > 0.7


if __name__ == "__main__":
    # 简易 runner（不依赖 pytest）
    import inspect
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print(f"  ✓ {name}")
        except AssertionError as e:
            failed += 1
            print(f"  ✗ {name}: {e}")
        except Exception as e:
            failed += 1
            print(f"  ✗ {name}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{passed+failed} passed")
    sys.exit(0 if failed == 0 else 1)
