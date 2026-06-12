"""
赛中三 Agent（参考报告 7.2）
================================

1. MatchResultCalibrator (MRCA) — 7.2.1
   接收赛果事件，调用 bayesian_updater 更新先验。

2. InjuryTrackerAgent (ITA) — 7.2.2
   监控核心球员状态，量化"依赖度 × 可替代性折扣"。

3. SentimentBiasAgent (SOA) — 7.2.3
   监测三指标：MMDI（模型-市场分歧）/ NMI（叙事动量）/ AVI（资金异常）

这三 Agent **不进入** Queen Swarm 主流程（不与 14 个赛前 Agent 一起跑），
而是作为独立的"赛中迭代层"被 cron_runner 在每场比赛结束后触发。
"""
from __future__ import annotations
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import asdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.bayesian_updater import (
    MatchEvent, batch_update, save_run_summary,
    EVENT_WEIGHTS, opponent_strength_factor,
)

DATA_RAW = Path(__file__).parent.parent.parent / "data" / "raw"
DATA_OUTPUTS = Path(__file__).parent.parent.parent / "data" / "outputs"


# ============================================================
# live_events.json (ESPN) → MatchEvent 转换
# ============================================================
# ESPN type 文本 → bayesian_updater 已知 event_type
_LIVE_EVENT_MAP = {
    "Red Card":      "red_card",
    "RedCard":       "red_card",
    "Yellow Red":    "red_card",       # 第二张黄=红
}

# 状态：post=刚结束（可以把 Goal 视为已敲定的 group_win/loss 早期信号）
# 我们仍不把 in-state 的 Goal 映射成 group_win（防止重复计分），
# 但允许 post 状态（比赛已结束、group_results 还没跑）短暂使用。
# group_results 跑完后会写 match_events.json，下一轮 in_match 会用那个权威源。
def _post_match_signals(match: dict, meta_ts: str) -> List["MatchEvent"]:
    """post 状态比赛：从 score 推导 group_win/group_loss/group_draw 临时信号。

    一旦 group_results 任务把权威结果写到 match_events.json，
    这个临时信号会被那边覆盖（time_decay 同一时间戳，bayes cap 也兜底）。
    """
    if match.get("state") != "post":
        return []
    sa = match.get("score_a")
    sb = match.get("score_b")
    team_a = match.get("team_a")
    team_b = match.get("team_b")
    if sa is None or sb is None or not team_a or not team_b:
        return []
    try:
        sa, sb = int(sa), int(sb)
    except (TypeError, ValueError):
        return []
    blowout = abs(sa - sb) >= 3
    if sa > sb:
        return [
            MatchEvent(team=team_a, event_type="group_win",
                        opponent=team_b, timestamp=meta_ts, blowout=blowout,
                        extra={"source": "live_events", "score": f"{sa}-{sb}"}),
            MatchEvent(team=team_b, event_type="group_loss",
                        opponent=team_a, timestamp=meta_ts, blowout=blowout,
                        extra={"source": "live_events", "score": f"{sa}-{sb}"}),
        ]
    if sb > sa:
        return [
            MatchEvent(team=team_b, event_type="group_win",
                        opponent=team_a, timestamp=meta_ts, blowout=blowout,
                        extra={"source": "live_events", "score": f"{sa}-{sb}"}),
            MatchEvent(team=team_a, event_type="group_loss",
                        opponent=team_b, timestamp=meta_ts, blowout=blowout,
                        extra={"source": "live_events", "score": f"{sa}-{sb}"}),
        ]
    return [
        MatchEvent(team=team_a, event_type="group_draw",
                    opponent=team_b, timestamp=meta_ts,
                    extra={"source": "live_events", "score": f"{sa}-{sb}"}),
        MatchEvent(team=team_b, event_type="group_draw",
                    opponent=team_a, timestamp=meta_ts,
                    extra={"source": "live_events", "score": f"{sa}-{sb}"}),
    ]


def _collect_match_context(live_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """从 live_events.json + weather.json + lineups.json 拼接每场进行中/刚结束比赛的上下文。

    输出供 web 直读，**不修改 priors**（避免 per-match 信号污染 per-tournament 概率）。
    每条 entry 形如：
      {
        "match_key": "Germany vs Curacao @ 2026-06-13",
        "state": "post",  # in/post
        "score": "5-0",
        "team_a": "Germany", "team_b": "Curacao",
        "weather": {"risk_level": "low", "temp_c": 32.3, "wbgt_c": 26.6},
        "lineups": {"team_a_confirmed": true, "team_b_confirmed": false, "concerns": []}
      }
    任一文件缺失 → 该字段为 None，不影响其他字段。
    """
    p = live_path or (DATA_RAW / "live_events.json")
    if not p.exists():
        return []
    try:
        raw = json.load(open(p))
    except Exception:
        return []

    # 一次性加载 weather / lineups（容错）
    weather_index: Dict[str, dict] = {}
    try:
        wdata = json.load(open(DATA_RAW / "weather.json"))
        for m in wdata.get("matches", []):
            key = f"{m.get('team_a')} vs {m.get('team_b')}"
            weather_index[key] = m.get("weather", {})
    except Exception:
        pass

    lineups_index: Dict[str, dict] = {}
    try:
        ldata = json.load(open(DATA_RAW / "lineups.json"))
        for k, v in ldata.get("lineups", {}).items():
            if k.startswith("_"):
                continue
            # 用 "team_a vs team_b" 做 key（去掉日期后缀）
            base = k.split(" @ ")[0]
            lineups_index[base] = v
    except Exception:
        pass

    out: List[Dict[str, Any]] = []
    for match in raw.get("live_matches", []):
        team_a = match.get("team_a")
        team_b = match.get("team_b")
        if not team_a or not team_b:
            continue
        key = f"{team_a} vs {team_b}"

        ctx: Dict[str, Any] = {
            "match_key": f"{key} (state={match.get('state')})",
            "state": match.get("state"),
            "score": match.get("score"),
            "team_a": team_a,
            "team_b": team_b,
            "weather": None,
            "lineups": None,
        }

        we = weather_index.get(key)
        if we:
            ctx["weather"] = {
                "risk_level": we.get("risk_level"),
                "temp_c": we.get("temp_c"),
                "wbgt_c": we.get("wbgt_estimate_c"),
                "precip_prob_pct": we.get("precip_prob_pct"),
            }

        lu = lineups_index.get(key)
        if lu:
            from data.match_context_adjustments import _team_has_concerning_changes
            concerns = []
            for side_key, label in [("team_a", team_a), ("team_b", team_b)]:
                side = lu.get(side_key) or {}
                hit, kw = _team_has_concerning_changes(side)
                if hit:
                    concerns.append({"team": label, "keyword": kw,
                                      "notes": side.get("notes", "")[:200]})
            ctx["lineups"] = {
                "team_a_confirmed": (lu.get("team_a") or {}).get("confirmed", False),
                "team_b_confirmed": (lu.get("team_b") or {}).get("confirmed", False),
                "concerns": concerns,
            }

        out.append(ctx)
    return out


def _load_live_events_as_match_events(path: Optional[Path] = None,
                                       max_age_minutes: float = 180.0) -> List["MatchEvent"]:
    """读 live_events.json 并把关键事件转成 MatchEvent。

    映射策略：
      - **进行中 (state=in)**：只取 Red Card / Yellow Red（决定性事件，cap 12pp）。
        不取 Goal——避免与赛末 group_results 重复计分。
      - **刚结束 (state=post)**：从 score 推导 group_win/loss/draw（含 blowout 标记）。
        这是 group_results 任务（60min 周期）跑完前的 fast-path 信号；
        一旦权威 match_events.json 写入，下次 in_match tick 用那个为准。

    失败/文件缺失 → 返回空 list（不破坏主流程）。
    """
    p = path or (DATA_RAW / "live_events.json")
    if not p.exists():
        return []
    try:
        raw = json.load(open(p))
    except Exception:
        return []

    out: List[MatchEvent] = []
    meta_ts = (raw.get("_metadata") or {}).get("updated_at", "")
    for match in raw.get("live_matches", []):
        team_a = match.get("team_a")
        team_b = match.get("team_b")
        if not team_a or not team_b:
            continue

        # 1. 进行中：红牌/二黄
        for ev in match.get("events", []):
            etype_raw = ev.get("type", "")
            etype = _LIVE_EVENT_MAP.get(etype_raw)
            if not etype:
                continue
            ev_team = ev.get("team")
            if ev_team not in (team_a, team_b):
                continue
            out.append(MatchEvent(
                team=ev_team,
                event_type=etype,
                timestamp=meta_ts,
                opponent=team_b if ev_team == team_a else team_a,
                extra={"source": "live_events", "minute": ev.get("minute", "")},
            ))

        # 2. 刚结束：临时 group_win/loss/draw（fast-path，等 group_results 覆盖）
        out.extend(_post_match_signals(match, meta_ts))
    return out


# ============================================================
# 1. MRCA — Match Result Calibrator
# ============================================================
class MatchResultCalibrator:
    """
    赛果校准 Agent（赛果校准规范 7.2.1）
    
    核心：接收 [比赛事件 + 当前先验]，输出 [更新后的后验 + cap 约束的 cumulative 调整]
    """
    name = "MRCA"

    def update(self, current_priors: Dict[str, float],
               events: List[MatchEvent]) -> Dict[str, Any]:
        """
        Args:
            current_priors: 当前各队夺冠概率（pp）
            events: 赛中事件列表
        
        Returns:
            {
                "posteriors": {team: pp},
                "deltas": {team: pp},
                "logs": [...],
                "summary": {n_events, n_teams_changed, max_swing_team, max_swing_pp}
            }
        """
        result = batch_update(current_priors, events)
        # 找最大波动
        deltas = result["cumulative_deltas"]
        if deltas:
            max_team = max(deltas, key=lambda t: abs(deltas[t]))
            max_swing = deltas[max_team]
        else:
            max_team, max_swing = None, 0.0

        result["summary"] = {
            "n_events": len(events),
            "n_teams_changed": sum(1 for d in deltas.values() if abs(d) > 0.01),
            "max_swing_team": max_team,
            "max_swing_pp": round(max_swing, 2),
        }
        return result


# ============================================================
# 2. ITA — Injury Tracking Agent
# ============================================================
class InjuryTrackerAgent:
    """
    伤病追踪 Agent（伤病追踪规范 7.2.2）
    
    核心公式：
      调整量 = base_weight × dependency × replaceability_discount × position_coef
    
    位置系数（经验）：
      forward / attacking_mid    1.0
      defensive_mid / wingback   0.7
      central_defender / GK      0.5
    """
    name = "ITA"

    POSITION_COEF = {
        "forward": 1.0,
        "winger": 1.0,
        "attacking_mid": 1.0,
        "central_mid": 0.85,
        "defensive_mid": 0.7,
        "wingback": 0.7,
        "fullback": 0.65,
        "central_defender": 0.5,
        "goalkeeper": 0.5,
    }

    REPLACEABILITY = {
        "starter": 1.0,       # 主力（无替代）
        "rotation": 0.6,
        "fringe": 0.3,
    }

    def evaluate_injury(self, team: str, player: str,
                         dependency: float,
                         position: str = "central_mid",
                         status: str = "starter",
                         injury_severity: float = 1.0,
                         expected_absent_matches: int = 1) -> Dict[str, Any]:
        """
        单条伤病事件评估
        
        Args:
            dependency: 0-1，球队对该球员的依赖度
            position: 位置（影响 coef）
            status: starter/rotation/fringe
            injury_severity: 0-1，严重程度（短期擦伤 0.3，断腿 1.0）
            expected_absent_matches: 预计缺阵场数
        """
        pos_coef = self.POSITION_COEF.get(position, 0.7)
        repl = self.REPLACEABILITY.get(status, 0.6)
        
        # 单场胜率影响（伤病追踪规范 7.2.2：姆巴佩缺 1 场 → -18pp 单场胜率）
        per_match_impact_pct = dependency * repl * pos_coef * injury_severity * 65.0
        # 夺冠概率影响：缺 N 场 × 平均 0.4（淘汰赛链）
        cumulative_pp = round(per_match_impact_pct * min(expected_absent_matches, 4) * 0.10, 3)

        # cap 在 ±30pp（参考表 7.2 key_injury 上限）
        cumulative_pp = max(-30.0, min(0.0, -cumulative_pp))

        return {
            "team": team,
            "player": player,
            "dependency": dependency,
            "position": position,
            "status": status,
            "per_match_impact_pct": round(per_match_impact_pct, 2),
            "cumulative_delta_pp": cumulative_pp,
            "rationale": (
                f"{player} ({position}, {status}, dep={dependency}): "
                f"单场 -{per_match_impact_pct:.1f}%，"
                f"预计缺阵 {expected_absent_matches} 场 → "
                f"夺冠概率 {cumulative_pp:+.2f}pp"
            ),
        }

    def evaluate_batch(self, injuries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """批量处理多条伤病"""
        evaluations = [self.evaluate_injury(**inj) for inj in injuries]
        # 按队聚合
        team_aggregate: Dict[str, float] = {}
        for ev in evaluations:
            t = ev["team"]
            team_aggregate[t] = team_aggregate.get(t, 0) + ev["cumulative_delta_pp"]
        return {
            "evaluations": evaluations,
            "team_aggregate_pp": {k: round(v, 2) for k, v in team_aggregate.items()},
            "n_evaluated": len(evaluations),
        }


# ============================================================
# 3. SOA — Sentiment & Odds Bias Agent
# ============================================================
class SentimentBiasAgent:
    """
    舆情偏差 Agent（舆情偏差规范 7.2.3）
    
    三大监测指标：
      MMDI = (model_prob - market_prob) / market_prob，>2.0 模型超买，<-2.0 模型超卖
      NMI  = 媒体叙事情感动量（外部输入）
      AVI  = 博彩资金异常波动（外部输入）
    """
    name = "SOA"

    MMDI_THRESHOLD = 0.30   # |MMDI| > 0.3 触发警报
    NMI_THRESHOLD = 1.5
    AVI_THRESHOLD = 1.5

    def compute_mmdi(self, model_prob_pct: float,
                       market_prob_pct: float) -> float:
        """模型-市场分歧指数"""
        if market_prob_pct < 0.5:  # 市场太低不看
            return 0.0
        return round((model_prob_pct - market_prob_pct) / market_prob_pct, 3)

    def classify(self, mmdi: float,
                  nmi: float = 0.0,
                  avi: float = 0.0) -> Dict[str, Any]:
        """
        参考表 7.2 续 - 异常事件分类矩阵：
          模型超买 (MMDI>+0.3, NMI>0)         → 维持，标记低估
          模型超卖 (MMDI<-0.3, NMI<0)         → 复核参数
          叙事泡沫 (MMDI>+0.15, NMI>1.5, AVI>1.5) → 降置信度
          信息冲击 (任意 MMDI, NMI/AVI 剧烈)   → 暂停更新
        """
        if abs(mmdi) > 1.0 and (abs(nmi) > 2.5 or avi > 3.0):
            return {
                "category": "info_shock",
                "action": "pause_update",
                "confidence_adjust": -0.5,
                "note": "突发事件信号过强，暂停自动更新等待人工",
            }
        if mmdi > self.MMDI_THRESHOLD and nmi > self.NMI_THRESHOLD and avi > self.AVI_THRESHOLD:
            return {
                "category": "narrative_bubble",
                "action": "reduce_confidence",
                "confidence_adjust": -0.2,
                "note": "媒体叙事泡沫 + 资金涌入，降低 20% 置信度",
            }
        if mmdi > self.MMDI_THRESHOLD and nmi >= 0:
            return {
                "category": "model_overbuy",
                "action": "maintain_flag_undervalued",
                "confidence_adjust": 0.0,
                "note": "模型显著高于市场，维持判断并标记市场低估",
            }
        if mmdi < -self.MMDI_THRESHOLD and nmi <= 0:
            return {
                "category": "model_oversell",
                "action": "review_assumptions",
                "confidence_adjust": -0.1,
                "note": "模型显著低于市场，触发参数复核",
            }
        return {
            "category": "consensus_confirmed",
            "action": "normal",
            "confidence_adjust": 0.0,
            "note": "市场与模型一致，正常更新",
        }

    def evaluate_team(self, team: str,
                       model_prob_pct: float,
                       market_prob_pct: float,
                       nmi: float = 0.0,
                       avi: float = 0.0) -> Dict[str, Any]:
        mmdi = self.compute_mmdi(model_prob_pct, market_prob_pct)
        cls = self.classify(mmdi, nmi, avi)
        return {
            "team": team,
            "model_prob_pct": model_prob_pct,
            "market_prob_pct": market_prob_pct,
            "mmdi": mmdi,
            "nmi": nmi,
            "avi": avi,
            **cls,
        }

    def evaluate_batch(self, teams_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """对一批球队跑偏差检测"""
        results = [self.evaluate_team(**t) for t in teams_data]
        # 按异常程度排序
        results.sort(key=lambda x: -abs(x["mmdi"]))
        return {
            "evaluations": results,
            "n_anomalies": sum(1 for r in results if r["category"] != "consensus_confirmed"),
            "n_pause_actions": sum(1 for r in results if r["action"] == "pause_update"),
        }


# ============ 调度入口 ============
def run_in_match_pipeline(events_path: Optional[Path] = None,
                            injuries_path: Optional[Path] = None,
                            verbose: bool = False) -> Dict[str, Any]:
    """
    完整赛中迭代：MRCA → ITA → SOA
    
    输入：
      data/raw/match_events.json   赛果事件流
      data/raw/in_match_injuries.json  赛中新增伤病
      data/outputs/synthesizer_report.json  当前先验
      data/raw/teams.json   市场隐含概率
    
    输出：
      data/outputs/in_match_update.json
    """
    # 1. 加载先验
    synth_path = DATA_OUTPUTS / "synthesizer_report.json"
    if not synth_path.exists():
        return {"error": "synthesizer_report.json 不存在，先跑赛前完整流水线"}
    synth = json.load(open(synth_path))
    priors = {t: info.get("final_probability", 0) for t, info in synth.items()}

    # 2. MRCA：消化赛果事件
    events_path = events_path or (DATA_RAW / "match_events.json")
    events = []
    if events_path.exists():
        raw = json.load(open(events_path))
        events = [MatchEvent(**e) for e in (raw.get("events", []) if isinstance(raw, dict) else raw)]

    # 追加：live_events.json 中的进行中/刚结束事件（红牌/进球等）
    # 设计：失败必须吞掉，绝不影响主流程
    try:
        live_events = _load_live_events_as_match_events()
        if live_events:
            events.extend(live_events)
    except Exception:
        pass

    mrca = MatchResultCalibrator()
    mrca_result = mrca.update(priors, events)

    # 3. ITA：消化赛中伤病
    injuries_path = injuries_path or (DATA_RAW / "in_match_injuries.json")
    injuries = []
    if injuries_path.exists():
        raw = json.load(open(injuries_path))
        injuries = raw.get("injuries", []) if isinstance(raw, dict) else raw

    ita = InjuryTrackerAgent()
    ita_result = ita.evaluate_batch(injuries) if injuries else {"evaluations": [], "n_evaluated": 0}

    # 把 ITA 的结果叠加到 posteriors
    posteriors_after_ita = dict(mrca_result["posteriors"])
    for team, dpp in ita_result.get("team_aggregate_pp", {}).items():
        if team in posteriors_after_ita:
            posteriors_after_ita[team] = max(0.1,
                                              round(posteriors_after_ita[team] + dpp, 3))

    # 4. SOA：跑模型 vs 市场
    teams_data_path = DATA_RAW / "teams.json"
    soa_evaluations = []
    if teams_data_path.exists():
        teams_dict = json.load(open(teams_data_path)).get("teams", {})
        soa_inputs = []
        for team, model_pp in posteriors_after_ita.items():
            market_pp = teams_dict.get(team, {}).get("market_implied", 0) * 100
            if market_pp > 0.5:
                soa_inputs.append({
                    "team": team,
                    "model_prob_pct": model_pp,
                    "market_prob_pct": round(market_pp, 2),
                })
        soa = SentimentBiasAgent()
        soa_result = soa.evaluate_batch(soa_inputs)
    else:
        soa_result = {"evaluations": [], "n_anomalies": 0}

    # 5. 收集 per-match 上下文（weather + lineups）—— 仅展示，不改 priors
    match_context: List[Dict[str, Any]] = []
    try:
        match_context = _collect_match_context()
    except Exception:
        match_context = []

    # 6. 写出
    out = {
        "ts": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "n_events": len(events),
        "n_injuries": len(injuries),
        "mrca": {
            "posteriors": mrca_result["posteriors"],
            "deltas": mrca_result["cumulative_deltas"],
            "summary": mrca_result["summary"],
            "logs": mrca_result["logs"][:20],
        },
        "ita": ita_result,
        "soa": soa_result,
        "final_posteriors": posteriors_after_ita,
        "match_context": match_context,
    }
    save_path = DATA_OUTPUTS / "in_match_update.json"
    save_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))

    if verbose:
        print(f"\n📡 赛中迭代完成")
        print(f"   MRCA: {mrca_result['summary']['n_events']} 事件，"
              f"最大波动 {mrca_result['summary']['max_swing_team']} "
              f"{mrca_result['summary']['max_swing_pp']:+.2f}pp")
        print(f"   ITA:  {ita_result['n_evaluated']} 个伤病评估")
        print(f"   SOA:  {soa_result['n_anomalies']} 个市场异常")
        print(f"   📄 写入 {save_path.relative_to(DATA_RAW.parent.parent)}")

    return out


if __name__ == "__main__":
    run_in_match_pipeline(verbose=True)
