"""v1 Live-Elo 反求（硬事实路径）。

输入比赛实时状态（比分/分钟/红黄牌/换人数），反求出双方的 ΔE（Elo 偏移），
配合 elo_engine.match_probabilities 即可得到当下 P(H/D/A)。

设计原则：
  - 保底路径：v2/v3' 全部失败时也要能跑出 P(H/D/A)
  - 字段缺失容忍：任何子项 None/缺失 → 该项贡献 0，不抛异常
  - 与 phase3 的 ELO_PER_PP 系数对齐（默认 10），便于"赛前 ΔE"与"赛中 ΔE"在同一坐标系比较
  - 公式经验值在文件顶部以常量集中暴露，未来可根据 jsonl 回测调参

使用：
  from quant.live_elo_solver import solve
  result = solve({
      "team_a": "Spain", "team_b": "Morocco",
      "score_a": 1, "score_b": 0, "elapsed_min": 67,
      "red_cards_a": 0, "red_cards_b": 1,
      "yellow_cards_a": 2, "yellow_cards_b": 3,
      "subs_used_a": 2, "subs_used_b": 3,
      "phase": "regular",
  })
  # → {"delta_elo_a": +84.0, "delta_elo_b": -120.0,
  #    "p_win_a": 0.71, "p_draw": 0.18, "p_win_b": 0.11,
  #    "components": {...}, "source": "v1_hard_facts"}
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.elo_engine import match_probabilities

# ─────────── 公式常量（v1 经验值，待回测校准）───────────

# pp ↔ Elo 转换系数（与 phase3 ai_weighted_baseline 对齐）
# 1pp 信号 ≈ 10 Elo 偏移
ELO_PER_PP = 10.0

# 比分项
SCORE_DIFF_CAP = 3                # 比分差最大算到 ±3（再大边际效用低）
SCORE_DIFF_PP_PER_GOAL = 12.0     # 单球差 × 时间衰减后的 pp

# 时间衰减：第 N 分钟的"相信度"
# 早期一球领先信号弱（容易反超），90min 时一球领先即决定胜负
TIME_DECAY_BASE = 0.4             # 第 0min 的衰减因子
TIME_DECAY_END = 1.0              # 第 90min 的衰减因子
REGULAR_MATCH_MIN = 90            # 常规时间长度

# 红牌：每张净红牌的影响
RED_CARD_PP_EACH = 6.0            # 经验：少一人 ≈ 1pp/min × 时间衰减，简化为常量

# 黄牌：弱信号（仅"累计 3 张以上"才显著影响踢法）
YELLOW_CARD_PP_EACH = 1.0
YELLOW_CARD_THRESHOLD = 3         # 单队累黄超阈值才计

# 换人余额（疲劳代理）：换得多 = 体能/伤病压力大
SUB_FATIGUE_PP_EACH = 1.5

# ─────────── 内部工具 ───────────

def _safe_int(v, default: int = 0) -> int:
    """容忍 None/str/float，转 int 失败返回默认值。"""
    if v is None:
        return default
    try:
        return int(v)
    except (ValueError, TypeError):
        return default


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _sign(v: float) -> int:
    return 1 if v > 0 else (-1 if v < 0 else 0)


def time_decay(elapsed_min: float) -> float:
    """elapsed_min 0-90 → decay 0.4-1.0 线性。

    超过 90min（OT/PK）按 1.0 处理（信号已最强）。
    """
    if elapsed_min is None:
        return TIME_DECAY_BASE  # 完全未知 → 极弱衰减
    e = max(0.0, min(REGULAR_MATCH_MIN, float(elapsed_min)))
    return TIME_DECAY_BASE + (TIME_DECAY_END - TIME_DECAY_BASE) * (e / REGULAR_MATCH_MIN)


# ─────────── 主函数 ───────────

def solve(match_state: dict, team_elos: Optional[dict] = None) -> dict:
    """v1 反求 ΔE 并算 P(H/D/A)。

    Args:
      match_state: 实时比赛状态 dict，至少需含 team_a/team_b/score_a/score_b/elapsed_min。
                   其余字段（red_cards_*, yellow_cards_*, subs_used_*, phase）缺失时
                   该项贡献 0。
      team_elos:  可选，{team_name: elo_baseline}。若提供则一并返回 P(H/D/A)；
                   不提供则只返回 ΔE 偏移（调用方自行算 P）。

    Returns:
      {
        "delta_elo_a": float, "delta_elo_b": float,
        "delta_total_pp": float,                 # ΔE_a - ΔE_b 等价的 pp 量
        "components": {                          # 审计：各分项 pp 贡献（已乘 time_decay）
          "score_diff_pp": float,
          "red_pp": float,
          "yellow_pp": float,
          "sub_fatigue_pp": float,
          "time_decay_factor": float,
        },
        "p_win_a": float, "p_draw": float, "p_win_b": float,   # team_elos 提供时
        "source": "v1_hard_facts",
        "warnings": [str, ...]                   # 字段缺失/异常告警
      }
    """
    warnings: list = []

    # 必需字段（缺一即 fail-soft，给保守默认）
    team_a = match_state.get("team_a")
    team_b = match_state.get("team_b")
    if not team_a or not team_b:
        warnings.append("missing team_a/team_b")
    score_a = _safe_int(match_state.get("score_a"))
    score_b = _safe_int(match_state.get("score_b"))
    elapsed_min = match_state.get("elapsed_min")
    if elapsed_min is None:
        warnings.append("missing elapsed_min, use 45 as fallback")
        elapsed_min = 45  # 中场作为保守估计
    phase = match_state.get("phase", "regular")

    # ① 比分项（带时间衰减）
    decay = time_decay(elapsed_min)
    score_diff_clipped = _clip(score_a - score_b, -SCORE_DIFF_CAP, SCORE_DIFF_CAP)
    score_diff_pp = score_diff_clipped * SCORE_DIFF_PP_PER_GOAL * decay

    # ② 红牌项（净差 × 经验系数；不做时间衰减——少一人的影响相对持续）
    red_a = _safe_int(match_state.get("red_cards_a"))
    red_b = _safe_int(match_state.get("red_cards_b"))
    red_pp = (red_b - red_a) * RED_CARD_PP_EACH

    # ③ 黄牌项（仅超阈值的部分计入）
    yellow_a = _safe_int(match_state.get("yellow_cards_a"))
    yellow_b = _safe_int(match_state.get("yellow_cards_b"))
    yellow_a_eff = max(0, yellow_a - YELLOW_CARD_THRESHOLD)
    yellow_b_eff = max(0, yellow_b - YELLOW_CARD_THRESHOLD)
    yellow_pp = (yellow_b_eff - yellow_a_eff) * YELLOW_CARD_PP_EACH

    # ④ 换人余额（疲劳代理）
    sub_a = _safe_int(match_state.get("subs_used_a"))
    sub_b = _safe_int(match_state.get("subs_used_b"))
    sub_fatigue_pp = (sub_b - sub_a) * SUB_FATIGUE_PP_EACH

    # 终场（FT）：信号封顶——比分差 ×1 衰减；其他维度仍计入
    # 注：FT 后预测应被 frozen_predictions 接管，这里仅作 fallback
    if phase == "FT":
        # 强行把 decay 推到 1.0（已经定局）
        score_diff_pp = score_diff_clipped * SCORE_DIFF_PP_PER_GOAL * 1.0

    total_pp = score_diff_pp + red_pp + yellow_pp + sub_fatigue_pp
    delta_elo_a = +total_pp * ELO_PER_PP
    delta_elo_b = -total_pp * ELO_PER_PP

    result = {
        "delta_elo_a": round(delta_elo_a, 2),
        "delta_elo_b": round(delta_elo_b, 2),
        "delta_total_pp": round(total_pp, 2),
        "components": {
            "score_diff_pp": round(score_diff_pp, 2),
            "red_pp": round(red_pp, 2),
            "yellow_pp": round(yellow_pp, 2),
            "sub_fatigue_pp": round(sub_fatigue_pp, 2),
            "time_decay_factor": round(decay, 3),
        },
        "source": "v1_hard_facts",
        "warnings": warnings,
    }

    # 可选：直接计算 P(H/D/A)
    if team_elos and team_a and team_b:
        elo_a_base = team_elos.get(team_a)
        elo_b_base = team_elos.get(team_b)
        if elo_a_base is not None and elo_b_base is not None:
            try:
                p = match_probabilities(elo_a_base + delta_elo_a,
                                        elo_b_base + delta_elo_b)
                result["p_win_a"] = round(p["p_win_a"], 4)
                result["p_draw"] = round(p["p_draw"], 4)
                result["p_win_b"] = round(p["p_win_b"], 4)
                result["elo_a_live"] = round(elo_a_base + delta_elo_a, 1)
                result["elo_b_live"] = round(elo_b_base + delta_elo_b, 1)
            except Exception as e:
                result["warnings"].append(f"match_probabilities failed: {e}")
        else:
            result["warnings"].append(f"missing baseline elo for {team_a}/{team_b}")

    return result


# ─────────── CLI 调试入口 ───────────

if __name__ == "__main__":
    # 演示：1-0 + 67min + 红牌优势
    demo_state = {
        "team_a": "Spain", "team_b": "Morocco",
        "score_a": 1, "score_b": 0,
        "elapsed_min": 67,
        "red_cards_a": 0, "red_cards_b": 1,
        "yellow_cards_a": 2, "yellow_cards_b": 3,
        "subs_used_a": 2, "subs_used_b": 3,
        "phase": "regular",
    }
    import json
    # 不带 team_elos 的纯 ΔE
    print("=== 仅 ΔE ===")
    print(json.dumps(solve(demo_state), ensure_ascii=False, indent=2))

    # 带 baseline elo 的完整 P(H/D/A)
    print("\n=== 注入 baseline elo ===")
    print(json.dumps(solve(demo_state, team_elos={"Spain": 2050, "Morocco": 1820}),
                     ensure_ascii=False, indent=2))
