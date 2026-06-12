"""
贝叶斯动态更新引擎（参考报告 7.2 Layer 5）
================================================

实现 MRCA（Match Result Calibration Agent）核心更新公式：

  posterior_p ∝ prior_p × likelihood(event)

其中 likelihood 由三层增强：
  1. 上下文感知（时间衰减）
  2. 对手强度调整（Elo 标准化）
  3. 情境因子叠加（高温/海拔/旅行）

参考报告表 7.2 权重矩阵：
                     基础权重    时间衰减    累计上限
  group_win        0.12      0.85^t     ±15pp
  group_draw       0.06      0.85^t     ±8pp
  group_loss       0.15      0.85^t     ±18pp
  knockout_win     0.18      0.85^t     ±22pp
  knockout_loss    0.20      0.85^t     ±25pp
  key_injury       0.25      no_decay   ±30pp
  red_card         0.10      0.85^t     ±12pp
  coach_change     0.08      half_full  ±10pp
"""
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict


# ============ 参考表 7.2 权重矩阵 ============
EVENT_WEIGHTS = {
    "group_win":      {"base": 0.12, "decay": True,  "cap_pp": 15.0},
    "group_draw":     {"base": 0.06, "decay": True,  "cap_pp": 8.0},
    "group_loss":     {"base": 0.15, "decay": True,  "cap_pp": 18.0},
    "knockout_win":   {"base": 0.18, "decay": True,  "cap_pp": 22.0},
    "knockout_loss":  {"base": 0.20, "decay": True,  "cap_pp": 25.0},
    "key_injury":     {"base": 0.25, "decay": False, "cap_pp": 30.0},
    "red_card":       {"base": 0.10, "decay": True,  "cap_pp": 12.0},
    "coach_change":   {"base": 0.08, "decay": True,  "cap_pp": 10.0},
}


# ============ 数据结构 ============
@dataclass
class MatchEvent:
    """一次赛中事件，喂给 update_prior()"""
    team: str
    event_type: str              # group_win/draw/loss, knockout_win/loss, key_injury, red_card, coach_change
    timestamp: str = ""          # ISO 时间，越近权重越高
    opponent: Optional[str] = None
    opponent_elo: Optional[float] = None
    own_elo: Optional[float] = None
    context: Dict[str, Any] = field(default_factory=dict)  # 海拔/温度/旅行
    blowout: bool = False        # 大比分（净胜≥3）
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UpdateLog:
    """单次更新的审计记录"""
    team: str
    event: str
    prior_pp: float
    posterior_pp: float
    delta_pp: float
    weight_used: float
    components: Dict[str, float]
    rationale: str = ""


# ============ 核心算子 ============
def time_decay_factor(event_ts: str, ref_ts: Optional[str] = None,
                       half_life_days: float = 30.0) -> float:
    """
    时间衰减：以 30 天为半衰期。事件发生越早衰减越多。
    参考报告 7.2.1：第 90 分钟进球权重约为第 45 分钟的 0.85^45min ≈ 较低
    我们简化为按"距今天数"做指数衰减。
    """
    if not event_ts:
        return 1.0
    try:
        ev = datetime.fromisoformat(event_ts.replace("Z", "+00:00")).replace(tzinfo=None)
        ref = (datetime.fromisoformat(ref_ts.replace("Z", "+00:00")).replace(tzinfo=None)
               if ref_ts else datetime.now())
        days = max(0.0, (ref - ev).days)
        return 0.5 ** (days / half_life_days)
    except Exception:
        return 1.0


def opponent_strength_factor(event: MatchEvent, default: float = 1.0) -> float:
    """
    对手强度归一化：击败 Elo 2150 的西班牙 vs 击败 Elo 1500 的弱旅，权重大不一样。
    
    系数 = (对手_elo - 平均_elo) / std_elo，bounded 到 [0.5, 2.0]
    """
    if event.opponent_elo is None:
        return default
    elo = event.opponent_elo
    # 大致定标：48 队 Elo 跨度 1400-2200，均值约 1750
    z = (elo - 1750) / 200.0
    factor = 1.0 + max(-0.5, min(1.0, z * 0.5))
    return round(factor, 3)


def blowout_modifier(event: MatchEvent) -> float:
    """大比分胜利 +50% 权重，被逆转失利 +30% 权重"""
    if event.blowout and "win" in event.event_type:
        return 1.5
    if event.blowout and "loss" in event.event_type:
        return 1.3
    return 1.0


def context_modifier(event: MatchEvent) -> float:
    """情境因子：高海拔/高温会归因到环境，弱化 likelihood 信号"""
    ctx = event.context or {}
    altitude = ctx.get("altitude_m", 0)
    wbgt = ctx.get("wbgt_c", 25)
    discount = 1.0
    if altitude > 1500:
        discount *= 0.85   # 极端环境，归因部分到环境
    if wbgt > 32:
        discount *= 0.9
    return discount


def update_prior(prior_pp: float, event: MatchEvent,
                  current_cumulative_pp: float = 0.0) -> UpdateLog:
    """
    单次贝叶斯更新（pp 空间，简化的 logit 近似）
    
    Args:
        prior_pp: 先验夺冠概率（百分点）
        event: 事件
        current_cumulative_pp: 该队迄今累积调整量（用于 cap 约束）
    
    Returns:
        UpdateLog（含 posterior_pp）
    """
    spec = EVENT_WEIGHTS.get(event.event_type)
    if spec is None:
        return UpdateLog(
            team=event.team, event=event.event_type,
            prior_pp=prior_pp, posterior_pp=prior_pp, delta_pp=0,
            weight_used=0, components={},
            rationale=f"未知事件类型: {event.event_type}",
        )

    # 1. 基础权重
    base_w = spec["base"]

    # 2. 时间衰减
    decay = time_decay_factor(event.timestamp) if spec["decay"] else 1.0

    # 3. 对手强度
    opp = opponent_strength_factor(event)

    # 4. 大比分修饰
    blow = blowout_modifier(event)

    # 5. 情境
    ctx = context_modifier(event)

    # 6. 综合权重
    weight = base_w * decay * opp * blow * ctx

    # 7. delta_pp 方向：win → 正，loss → 负
    sign = 1.0
    if "loss" in event.event_type:
        sign = -1.0
    elif event.event_type == "key_injury":
        sign = -1.0
    elif event.event_type == "red_card":
        sign = -1.0
    elif "draw" in event.event_type and event.opponent_elo and event.own_elo:
        # 弱队拿到平局是利好；强队被弱队逼平是利空
        sign = 1.0 if event.opponent_elo > event.own_elo else -0.5

    raw_delta = sign * weight * prior_pp  # 影响幅度与先验成比例

    # 8. cap 约束（既限制单次也限制累积）
    cap = spec["cap_pp"]
    new_cumulative = current_cumulative_pp + raw_delta
    if abs(new_cumulative) > cap:
        # 截断到 cap
        excess = abs(new_cumulative) - cap
        if raw_delta > 0:
            raw_delta -= excess
        else:
            raw_delta += excess

    posterior_pp = max(0.1, min(50.0, prior_pp + raw_delta))

    return UpdateLog(
        team=event.team,
        event=event.event_type,
        prior_pp=round(prior_pp, 3),
        posterior_pp=round(posterior_pp, 3),
        delta_pp=round(raw_delta, 3),
        weight_used=round(weight, 4),
        components={
            "base": base_w,
            "time_decay": round(decay, 3),
            "opponent_strength": opp,
            "blowout": blow,
            "context": round(ctx, 3),
            "sign": sign,
        },
        rationale=(
            f"{event.event_type} vs {event.opponent or '?'} "
            f"(opp_elo={event.opponent_elo}): "
            f"weight={weight:.4f}, delta={raw_delta:+.2f}pp"
        ),
    )


def batch_update(team_priors: Dict[str, float],
                  events: List[MatchEvent]) -> Dict[str, Any]:
    """
    对一批事件做顺序贝叶斯更新。
    
    Returns:
        {
            "posteriors": {team: pp},
            "logs": [UpdateLog, ...],
            "cumulative_deltas": {team: pp},
        }
    """
    posteriors = dict(team_priors)
    cumulative = {team: 0.0 for team in team_priors}
    logs = []

    for event in events:
        prior = posteriors.get(event.team)
        if prior is None:
            continue
        log = update_prior(prior, event, cumulative.get(event.team, 0.0))
        posteriors[event.team] = log.posterior_pp
        cumulative[event.team] = cumulative.get(event.team, 0.0) + log.delta_pp
        logs.append(log)

    return {
        "posteriors": posteriors,
        "cumulative_deltas": cumulative,
        "logs": [asdict(l) for l in logs],
    }


# ============ I/O：读历史事件流 ============
DATA_RAW = Path(__file__).parent.parent.parent / "data" / "raw"
DATA_OUTPUTS = Path(__file__).parent.parent.parent / "data" / "outputs"


def load_match_events(path: Optional[Path] = None) -> List[MatchEvent]:
    """从 data/raw/match_events.json 读事件流"""
    path = path or (DATA_RAW / "match_events.json")
    if not path.exists():
        return []
    raw = json.load(open(path))
    events = raw.get("events", []) if isinstance(raw, dict) else raw
    return [MatchEvent(**e) for e in events]


def save_run_summary(result: Dict[str, Any], path: Optional[Path] = None) -> Path:
    path = path or (DATA_OUTPUTS / "bayesian_update_summary.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return path
