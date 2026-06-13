"""
赛事推演 API 模块
==================

提供 8 个端点的数据生成函数，供 server.py 调用。

API 列表：
  /api/groups        — 12 小组每组排名（按 MC 期望得分）
  /api/r32           — 32 强名单 + 各 R32 场次预测
  /api/r16           — 16 强对阵 + 每场胜率
  /api/qf            — 8 强对阵
  /api/sf            — 4 强对阵
  /api/final_match   — 决赛对阵 + Top 比分
  /api/bracket       — 完整对阵树（嵌套 JSON）
  /api/match         — 任意两队对战（带 LLM 分析，见 match_analyzer.py）
"""
from __future__ import annotations
import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_RAW = Path(__file__).parent.parent.parent / "data" / "raw"
DATA_OUTPUTS = Path(__file__).parent.parent.parent / "data" / "outputs"


# ============ 基础数据加载 ============
def _load_teams():
    return json.load(open(DATA_RAW / "teams.json"))["teams"]


def _load_groups():
    return json.load(open(DATA_RAW / "groups.json"))


def _load_bracket():
    return json.load(open(DATA_RAW / "bracket.json"))


def _resolve_channel(channel: Optional[str]) -> str:
    """
    标准化 channel 参数。base / ai_phase3 双通道支持。
    无效值或 None 都退回 base，保证兼容老调用。
    """
    if channel == "ai_phase3":
        return "ai_phase3"
    return "base"


def _load_mc(channel: Optional[str] = None):
    """加载 MC 模拟概率。channel="ai_phase3" 时读 phase3，否则 base。
    AI 通道文件不存在自动回退 base，保证不破坏主流程。"""
    ch = _resolve_channel(channel)
    if ch == "ai_phase3":
        p = DATA_OUTPUTS / "mc_simulation_n100000_ai_phase3.json"
        if p.exists():
            return json.load(open(p))
        # 回退 base
    p = DATA_OUTPUTS / "mc_simulation_n100000.json"
    return json.load(open(p)) if p.exists() else {}


def _load_synth(channel: Optional[str] = None):
    """加载 synth 报告。channel="ai_phase3" 时读 phase3 报告，否则 base。
    AI 通道文件不存在自动回退 base。"""
    ch = _resolve_channel(channel)
    if ch == "ai_phase3":
        p = DATA_OUTPUTS / "synthesizer_report_ai_phase3.json"
        if p.exists():
            return json.load(open(p))
    p = DATA_OUTPUTS / "synthesizer_report.json"
    return json.load(open(p)) if p.exists() else {}


def _load_group_rank_dist():
    """加载小组第 1 名概率分布（MC 采样）"""
    p = DATA_OUTPUTS / "group_rank_dist.json"
    return json.load(open(p)) if p.exists() else None


def _load_critical_adjustments():
    """加载关键节点 5-Agent 微调结果"""
    p = DATA_OUTPUTS / "critical_node_adjustments.json"
    return json.load(open(p)) if p.exists() else None


def _load_match_bias():
    """加载单场偏差检测结果（Kalshi vs 模型）"""
    p = DATA_OUTPUTS / "match_bias.json"
    return json.load(open(p)) if p.exists() else None


def _mb_lookup(team_a: str, team_b: str, date: str) -> Optional[Dict]:
    """按 (team_a, team_b, date) 查偏差记录"""
    mb = _load_match_bias()
    if not mb:
        return None
    for m in mb.get("matches", []):
        if m["date"] == date and m["team_a"] == team_a and m["team_b"] == team_b:
            return m
    return None


def _load_upset_llm():
    """加载 LLM 终审结果"""
    p = DATA_OUTPUTS / "upset_llm_analysis.json"
    return json.load(open(p)) if p.exists() else None


def _upset_lookup(team_a: str, team_b: str, date: str) -> Optional[Dict]:
    """按 (team_a, team_b, date) 查 LLM 归因"""
    data = _load_upset_llm()
    if not data:
        return None
    key = f"{team_a}|{team_b}|{date}"
    return data.get("results", {}).get(key)


def _load_injuries() -> Optional[Dict]:
    """加载伤病雷达结果（来自 injury_radar.py 写入）"""
    p = DATA_RAW / "injuries.json"
    return json.load(open(p)) if p.exists() else None


def _inj_lookup(team: str) -> Optional[Dict]:
    """按队名查伤病。返回精简后给前端展示的结构（无伤返回 None）"""
    data = _load_injuries()
    if not data:
        return None
    info = data.get("injuries", {}).get(team)
    if not info:
        return None
    absences = info.get("absences", [])
    if not absences:
        return None
    return {
        "team": team,
        "n_absences": len(absences),
        "total_pp_impact": info.get("total_pp_impact", 0),
        "confidence": info.get("confidence", "中"),
        "notes": info.get("notes", ""),
        "absences": [
            {
                "player": a.get("player", "?"),
                "position": a.get("position", ""),
                "status": a.get("status", ""),
                "importance": a.get("importance", ""),
                "pp_impact": a.get("pp_impact", 0),
                "rationale": a.get("rationale", ""),
            }
            for a in absences
        ],
        "checked_at": info.get("checked_at", ""),
    }


def _ca_lookup(round_name: str, match_id: int) -> Optional[Dict]:
    """从 critical_node_adjustments.json 查某场的修正
    
    返回结构（applied=True 时）:
      {applied, final_adjustment_pp, consensus_winner, majority_count,
       total_agents, agents, summary}
    """
    ca = _load_critical_adjustments()
    if not ca:
        return None
    key = f"{round_name}:{match_id}"
    return ca.get("adjustments", {}).get(key)


def _load_most_likely(channel: Optional[str] = None):
    """加载方案 B 的「MC 最可能剧本」数据。
    channel="ai_phase3" 时优先读 phase3 派生文件，缺失自动回退 base。"""
    ch = _resolve_channel(channel)
    if ch == "ai_phase3":
        p = DATA_OUTPUTS / "most_likely_bracket_ai_phase3.json"
        if p.exists():
            return json.load(open(p))
    p = DATA_OUTPUTS / "most_likely_bracket.json"
    return json.load(open(p)) if p.exists() else None


def _ml_lookup(round_name: str, match_id: int, channel: Optional[str] = None) -> Optional[Dict]:
    """从最可能剧本里查某场比赛的 (team_a, team_b, winner, prob_pct)"""
    ml = _load_most_likely(channel)
    if not ml:
        return None
    rounds = ml.get("rounds", {})
    if round_name == "final":
        return rounds.get("final")
    for entry in rounds.get(round_name, []):
        if entry.get("match_id") == match_id:
            return entry
    return None


# ============ 1. 小组赛 ============
def api_groups(channel: Optional[str] = None) -> Dict[str, Any]:
    """
    12 小组每组排名 + 预测出线
    排序优先级：
      1. 若 group_rank_dist.json 存在 → 按"小组第 1 名概率"排序（MC 100k 实测）
      2. 否则回退到 Elo 排序

    channel="ai_phase3" 时使用 AI 加权 MC 通道。
    """
    teams = _load_teams()
    groups_data = _load_groups()
    mc = _load_mc(channel)
    rank_dist = _load_group_rank_dist()

    # 按 group 字段分组
    by_group: Dict[str, List[Dict]] = {}
    for name, info in teams.items():
        g = info.get("group")
        if g and g != "_":
            mc_data = mc.get(name, {})
            # 小组第 1 名概率（来自 MC 采样）
            first_pct = None
            if rank_dist:
                g_dist = rank_dist.get("groups", {}).get(g, {})
                if name in g_dist:
                    first_pct = g_dist[name].get("first_pct")
            by_group.setdefault(g, []).append({
                "team": name,
                "elo": info.get("elo"),
                "fifa_rank": info.get("fifa"),
                "xg_for": info.get("xg_for"),
                "xg_against": info.get("xg_against"),
                "squad_value_m_eur": info.get("squad_value_m_eur"),
                "market_implied_pct": round(info.get("market_implied", 0) * 100, 2),
                "first_pct": first_pct,   # 小组第 1 名概率
                # MC 出线概率
                "qual_to_r32_pct": round(mc_data.get("round_of_32", 0) * 100, 2),
                "qual_to_r16_pct": round(mc_data.get("round_of_16", 0) * 100, 2),
                "qual_to_qf_pct": round(mc_data.get("quarterfinal", 0) * 100, 2),
            })

    # 按"第 1 名概率"排序；若无数据则回退 Elo
    out_groups = {}
    for g in sorted(by_group):
        if rank_dist:
            teams_in_g = sorted(
                by_group[g],
                key=lambda x: (-(x["first_pct"] or 0), -x["elo"]),
            )
        else:
            teams_in_g = sorted(by_group[g], key=lambda x: -x["elo"])
        for rank, t in enumerate(teams_in_g, 1):
            t["predicted_rank"] = rank
        out_groups[g] = {
            "group": g,
            "teams": teams_in_g,
            "predicted_winner": teams_in_g[0]["team"] if teams_in_g else None,
            "predicted_runner_up": teams_in_g[1]["team"] if len(teams_in_g) > 1 else None,
            "predicted_3rd": teams_in_g[2]["team"] if len(teams_in_g) > 2 else None,
        }

    return {
        "n_groups": len(out_groups),
        "groups": out_groups,
        "method": "MC 100k 小组第 1 名概率排序" if rank_dist else "按 Elo 排序",
    }


# ============ 2. 32 强名单 ============
def api_r32(channel: Optional[str] = None) -> Dict[str, Any]:
    """
    32 强 = 12 小组前 2 名 (24 队) + 8 个最佳第 3 名
    返回名单 + R32 16 场对阵

    channel="ai_phase3" 时使用 AI 加权 MC 通道。
    """
    teams = _load_teams()
    bracket = _load_bracket()
    mc = _load_mc(channel)

    # 各组预测出线
    groups_result = api_groups(channel)["groups"]

    # 各组预测的 winner/runner_up
    g_winners = {g: info["predicted_winner"] for g, info in groups_result.items()}
    g_runners = {g: info["predicted_runner_up"] for g, info in groups_result.items()}

    # 8 个最佳第 3 名（按 Elo 取前 8）
    third_candidates = sorted(
        [(info["predicted_3rd"], teams.get(info["predicted_3rd"], {}).get("elo", 0))
         for info in groups_result.values() if info["predicted_3rd"]],
        key=lambda x: -x[1],
    )[:8]
    third_places = [t for t, _ in third_candidates]

    # 拼装 32 强名单
    r32_teams = []
    for g, w in sorted(g_winners.items()):
        r32_teams.append({"team": w, "seed": f"{g}1", "via": "group_winner"})
    for g, ru in sorted(g_runners.items()):
        if ru:
            r32_teams.append({"team": ru, "seed": f"{g}2", "via": "group_runner_up"})
    for i, t in enumerate(third_places, 1):
        r32_teams.append({"team": t, "seed": f"3rd_{i}", "via": "best_third"})

    # 对阵：用 bracket.round_of_32 + 解析 slot
    # slot 形如 "B2" "C1" "3rd_from_ABCDF"
    def resolve_slot(slot: str) -> Optional[str]:
        if slot.startswith("3rd_from_"):
            # 取候选组的最强第 3
            allowed_groups = slot.replace("3rd_from_", "")
            candidates = [(info["predicted_3rd"],
                            teams.get(info["predicted_3rd"], {}).get("elo", 0))
                          for g, info in groups_result.items()
                          if g in allowed_groups and info["predicted_3rd"]]
            if not candidates:
                return None
            return max(candidates, key=lambda x: x[1])[0]
        # 普通 slot 如 "B2" "C1"
        if len(slot) >= 2 and slot[0].isalpha():
            g = slot[0]
            pos = slot[1]
            if pos == "1":
                return g_winners.get(g)
            elif pos == "2":
                return g_runners.get(g)
        return None

    r32_matches = []
    for m in bracket.get("round_of_32", []):
        mid = m["match"]
        # 方案 B 优先：用 MC 最可能对阵覆盖 slot 解析结果
        ml_entry = _ml_lookup("r32", mid, channel=channel)
        if ml_entry:
            ta, tb = ml_entry["team_a"], ml_entry["team_b"]
        else:
            ta = resolve_slot(m["slot_a"])
            tb = resolve_slot(m["slot_b"])
        match_info = {
            "match_id": mid,
            "venue": m.get("venue"),
            "date": m.get("date"),
            "slot_a": m["slot_a"], "slot_b": m["slot_b"],
            "team_a": ta, "team_b": tb,
        }
        if ml_entry:
            match_info["_ml_prob_pct"] = ml_entry["prob_pct"]
            match_info["_ml_winner"] = ml_entry["winner"]
            match_info["_ml_alt_top"] = ml_entry.get("alt_matchups", [])[:3]
        if ta and tb:
            match_info["preview"] = _quick_match_preview(ta, tb, channel=channel)
        # 关键节点 5-Agent 微调（若有）
        _apply_critical_adjustment(match_info, "r32", mid)
        r32_matches.append(match_info)

    return {
        "n_teams": len(r32_teams),
        "teams": r32_teams,
        "n_matches": len(r32_matches),
        "matches": r32_matches,
        "method": ("MC 最可能剧本（方案 B/" + _resolve_channel(channel) + "）"
                   if _load_most_likely(channel) else "小组赛预测出线 → slot 解析"),
    }


def _apply_critical_adjustment(match_info: Dict, round_name: str, mid: int) -> None:
    """
    把 critical_node_adjustments.json 的修正应用到 match_info["preview"]。
    
    修改 p_win_a、p_win_b（保持 p_draw 不变），并附 _critical_adj 字段供前端展示。
    若 applied=False（分歧或失败），仍附 _critical_adj 但不动 p 值。
    """
    ca = _ca_lookup(round_name, mid)
    if not ca:
        return
    
    preview = match_info.get("preview")
    if not preview:
        # 仅记录元数据
        match_info["_critical_adj"] = {
            "applied": ca.get("applied"),
            "summary": ca.get("summary") or ca.get("reason") or ca.get("error"),
        }
        return
    
    # 元数据（无论是否 applied 都附）
    meta = {
        "applied": ca.get("applied", False),
        "summary": ca.get("summary") or ca.get("reason") or ca.get("error"),
        "consensus_winner": ca.get("consensus_winner"),
        "final_adjustment_pp": ca.get("final_adjustment_pp"),
        "majority_count": ca.get("majority_count"),
        "total_agents": ca.get("total_agents"),
        "agents": [
            {"name": a.get("agent_name"), "winner": a.get("predicted_winner"),
             "pp": a.get("adjustment_pp"), "rationale": a.get("rationale")}
            for a in ca.get("agents", [])
        ],
    }
    match_info["_critical_adj"] = meta
    
    # 若未生效则不动概率
    if not ca.get("applied"):
        return
    
    # 应用修正：final_adjustment_pp 是加到 team_a 胜率上的（正数推 team_a，负数推 team_b）
    adj_frac = ca["final_adjustment_pp"] / 100.0
    p_a = preview.get("p_win_a", 0)
    p_b = preview.get("p_win_b", 0)
    
    # 限制不出 [0.01, 0.99]，保留 p_draw
    new_p_a = max(0.01, min(0.99, p_a + adj_frac))
    new_p_b = max(0.01, min(0.99, p_b - adj_frac))
    
    # 二次归一（防小数漂移），保持 p_draw 不变
    p_d = preview.get("p_draw", 0)
    total = new_p_a + p_d + new_p_b
    if abs(total - 1.0) > 0.005 and total > 0:
        scale = (1.0 - p_d) / (new_p_a + new_p_b)
        new_p_a *= scale
        new_p_b *= scale
    
    preview["_raw_p_win_a"] = p_a   # 保留原值供审计
    preview["_raw_p_win_b"] = p_b
    preview["p_win_a"] = round(new_p_a, 4)
    preview["p_win_b"] = round(new_p_b, 4)
    
    # 同步更新 predicted_winner / winner_confidence_pct
    if new_p_a >= new_p_b:
        preview["predicted_winner"] = match_info["team_a"]
        preview["winner_confidence_pct"] = round((new_p_a + p_d * 0.5) * 100, 1)
    else:
        preview["predicted_winner"] = match_info["team_b"]
        preview["winner_confidence_pct"] = round((new_p_b + p_d * 0.5) * 100, 1)


# ============ 3. 16/8/4/决赛 名单 ============
def _propagate_winners(matches: List[Dict]) -> Dict[int, str]:
    """从一组场次取每场预测赢方，返回 {match_id: winner_team_name}
    
    自动识别 match_id/r16_match/qf_match/sf_match 键
    """
    winners = {}
    for m in matches:
        if not (m.get("team_a") and m.get("team_b")):
            continue
        preview = m.get("preview", {})
        p_a = preview.get("p_win_a", 0)
        p_b = preview.get("p_win_b", 0)
        winner = m["team_a"] if p_a >= p_b else m["team_b"]
        # 自动找 match_id 字段
        mid = m.get("match_id") or m.get("r16_match") or m.get("qf_match") or m.get("sf_match")
        if mid is not None:
            winners[mid] = winner
    return winners


def _build_round_from_most_likely(round_name: str, bracket_pairings: List[Dict],
                                   id_field: str, prev_round: str,
                                   prev_a_key: str, prev_b_key: str,
                                   channel: Optional[str] = None) -> Optional[List[Dict]]:
    """
    通用：从 most_likely_bracket.json 构造某轮 matches 列表。
    
    返回 None 表示数据缺失，调用方应回退到贪心传播。

    channel：同时影响"读哪份 most_likely 剧本"和"preview 走哪个 synth"。
    """
    ml = _load_most_likely(channel)
    if not ml:
        return None
    
    matches = []
    for p in bracket_pairings:
        mid = p[id_field]
        ml_entry = _ml_lookup(round_name, mid, channel=channel)
        if not ml_entry:
            return None  # 数据不完整，整体回退
        ta = ml_entry["team_a"]
        tb = ml_entry["team_b"]
        m = {
            id_field: mid,
            f"from_{prev_round}_a": p[prev_a_key],
            f"from_{prev_round}_b": p[prev_b_key],
            "team_a": ta, "team_b": tb,
            # 方案 B 新增：MC 真实概率元数据
            "_ml_prob_pct": ml_entry["prob_pct"],
            "_ml_winner": ml_entry["winner"],
            "_ml_alt_top": ml_entry.get("alt_matchups", [])[:3],
        }
        # 其他可选字段
        if "position" in p:
            m["position"] = p["position"]
        if "_pathway" in p:
            m["pathway"] = p["_pathway"]
        if ta and tb:
            m["preview"] = _quick_match_preview(ta, tb, channel=channel)
        # 关键节点 5-Agent 微调（若有）
        _apply_critical_adjustment(m, round_name, mid)
        matches.append(m)
    return matches


def api_r16(channel: Optional[str] = None) -> Dict[str, Any]:
    bracket = _load_bracket()
    pairings = bracket.get("_round_of_16_pairings", [])
    
    # 优先用方案 B：MC 最可能剧本（注意：most_likely_bracket 暂仅 base 通道，
    # phase3 用同一份对阵剧本，仅 preview 概率会反映 phase3 调整）
    r16_matches = _build_round_from_most_likely(
        "r16", pairings, "r16_match", "r32",
        "winner_of_match_a", "winner_of_match_b", channel=channel)
    
    # 回退：贪心 Top-1 传播
    if r16_matches is None:
        r32 = api_r32(channel)
        r32_winners = _propagate_winners(r32["matches"])
        r16_matches = []
        for p in pairings:
            ta = r32_winners.get(p["winner_of_match_a"])
            tb = r32_winners.get(p["winner_of_match_b"])
            m = {
                "r16_match": p["r16_match"],
                "from_r32_a": p["winner_of_match_a"],
                "from_r32_b": p["winner_of_match_b"],
                "team_a": ta, "team_b": tb,
            }
            if ta and tb:
                m["preview"] = _quick_match_preview(ta, tb, channel=channel)
            r16_matches.append(m)

    teams_in_r16 = list({m["team_a"] for m in r16_matches if m.get("team_a")} |
                        {m["team_b"] for m in r16_matches if m.get("team_b")})
    return {
        "n_teams": len(teams_in_r16),
        "teams": sorted(teams_in_r16),
        "n_matches": len(r16_matches),
        "matches": r16_matches,
        "method": ("MC 最可能剧本（方案 B/" + _resolve_channel(channel) + "）"
                   if _load_most_likely(channel) else "贪心 Top-1 传播"),
    }


def api_qf(channel: Optional[str] = None) -> Dict[str, Any]:
    bracket = _load_bracket()
    pairings = bracket.get("_quarterfinal_pairings", [])
    
    qf_matches = _build_round_from_most_likely(
        "qf", pairings, "qf_match", "r16",
        "winner_of_r16_a", "winner_of_r16_b", channel=channel)
    
    if qf_matches is None:
        r16 = api_r16(channel)
        r16_winners = _propagate_winners(r16["matches"])
        qf_matches = []
        for p in pairings:
            ta = r16_winners.get(p["winner_of_r16_a"])
            tb = r16_winners.get(p["winner_of_r16_b"])
            m = {
                "qf_match": p["qf_match"],
                "position": p.get("_position"),
                "from_r16_a": p["winner_of_r16_a"],
                "from_r16_b": p["winner_of_r16_b"],
                "team_a": ta, "team_b": tb,
            }
            if ta and tb:
                m["preview"] = _quick_match_preview(ta, tb, channel=channel)
            qf_matches.append(m)

    teams_in = sorted({t for m in qf_matches
                       for t in (m.get("team_a"), m.get("team_b")) if t})
    return {"n_teams": len(teams_in), "teams": teams_in,
            "n_matches": len(qf_matches), "matches": qf_matches,
            "method": ("MC 最可能剧本（方案 B/" + _resolve_channel(channel) + "）"
                       if _load_most_likely(channel) else "贪心 Top-1 传播")}


def api_sf(channel: Optional[str] = None) -> Dict[str, Any]:
    bracket = _load_bracket()
    pairings = bracket.get("_semifinal_pairings", [])
    
    sf_matches = _build_round_from_most_likely(
        "sf", pairings, "sf_match", "qf",
        "winner_of_qf_a", "winner_of_qf_b", channel=channel)
    
    if sf_matches is None:
        qf = api_qf(channel)
        qf_winners = _propagate_winners(qf["matches"])
        sf_matches = []
        for p in pairings:
            ta = qf_winners.get(p["winner_of_qf_a"])
            tb = qf_winners.get(p["winner_of_qf_b"])
            m = {
                "sf_match": p["sf_match"],
                "pathway": p.get("_pathway"),
                "from_qf_a": p["winner_of_qf_a"],
                "from_qf_b": p["winner_of_qf_b"],
                "team_a": ta, "team_b": tb,
            }
            if ta and tb:
                m["preview"] = _quick_match_preview(ta, tb, channel=channel)
            sf_matches.append(m)

    teams_in = sorted({t for m in sf_matches
                       for t in (m.get("team_a"), m.get("team_b")) if t})
    return {"n_teams": len(teams_in), "teams": teams_in,
            "n_matches": len(sf_matches), "matches": sf_matches,
            "method": ("MC 最可能剧本（方案 B/" + _resolve_channel(channel) + "）"
                       if _load_most_likely(channel) else "贪心 Top-1 传播")}


def api_final_match(channel: Optional[str] = None) -> Dict[str, Any]:
    """决赛对阵"""
    bracket = _load_bracket()
    fp = bracket.get("_final_pairing", {})
    
    # 优先用方案 B：MC 最可能剧本
    ml_final = _ml_lookup("final", 1, channel=channel)
    if ml_final:
        ta = ml_final["team_a"]
        tb = ml_final["team_b"]
        final = {
            "match_id": fp.get("final_match", 1),
            "from_sf_a": fp.get("winner_of_sf_a"),
            "from_sf_b": fp.get("winner_of_sf_b"),
            "team_a": ta, "team_b": tb,
            "_ml_prob_pct": ml_final["prob_pct"],
            "_ml_winner": ml_final["winner"],
            "_ml_alt_top": ml_final.get("alt_matchups", [])[:3],
        }
    else:
        # 回退：贪心传播
        sf = api_sf(channel)
        sf_winners = _propagate_winners(sf["matches"])
        ta = sf_winners.get(fp.get("winner_of_sf_a"))
        tb = sf_winners.get(fp.get("winner_of_sf_b"))
        final = {
            "match_id": fp.get("final_match", 1),
            "from_sf_a": fp.get("winner_of_sf_a"),
            "from_sf_b": fp.get("winner_of_sf_b"),
            "team_a": ta, "team_b": tb,
        }
    
    if ta and tb:
        final["preview"] = _quick_match_preview(ta, tb, channel=channel)
        # 关键节点 5-Agent 微调（若有）
        _apply_critical_adjustment(final, "final", 1)
        # 预测冠军：方案 B 优先用 MC 频次最高胜方，否则用贪心
        if ml_final:
            final["predicted_champion"] = ml_final["winner"]
        else:
            p_a = final["preview"]["p_win_a"]
            final["predicted_champion"] = ta if p_a >= 0.5 else tb

    # 第 3 名：从 sf 半决赛输方推出
    sf_data = api_sf(channel)
    losers = []
    for m in sf_data["matches"]:
        if not (m.get("team_a") and m.get("team_b")):
            continue
        # 方案 B：优先用 MC 频次最高胜方；否则用单场预测
        winner = m.get("_ml_winner")
        if not winner and m.get("preview"):
            winner = m["team_a"] if m["preview"]["p_win_a"] >= m["preview"]["p_win_b"] else m["team_b"]
        if not winner:
            continue
        loser = m["team_b"] if winner == m["team_a"] else m["team_a"]
        losers.append(loser)
    return {
        "final": final,
        "third_place_match": {
            "team_a": losers[0] if len(losers) > 0 else None,
            "team_b": losers[1] if len(losers) > 1 else None,
        } if len(losers) >= 2 else None,
    }


# ============ 4. 完整对阵树 ============
def api_bracket() -> Dict[str, Any]:
    """嵌套整个对阵树供前端可视化"""
    return {
        "round_of_32": api_r32()["matches"],
        "round_of_16": api_r16()["matches"],
        "quarterfinals": api_qf()["matches"],
        "semifinals": api_sf()["matches"],
        "final": api_final_match(),
    }


# ============ 5. 任意两队预测（快速版，不含 LLM）============
def _confidence_label(p: float) -> str:
    """路径表体例 风格的置信度档次"""
    if p >= 0.80:
        return "高"
    if p >= 0.65:
        return "中高"
    if p >= 0.45:
        return "中"
    if p >= 0.30:
        return "中低"
    return "低"


def _win_range(p: float) -> str:
    """Reference 风格的胜率区间（±5pp）"""
    if p >= 0.90:
        return f">{int(p * 100) - 5}%"
    if p <= 0.10:
        return f"<{int(p * 100) + 5}%"
    low = max(0, int(p * 100) - 5)
    high = min(100, int(p * 100) + 5)
    return f"{low}-{high}%"


def _goal_diff_range(lam_a: float, lam_b: float) -> str:
    """
    净胜球区间（主队视角）：
      正数 = 主队净胜，负数 = 主队净负，0 = 平局
    
    基于真实 Poisson 联合分布的 60% 中央置信区间（去掉两端各 20%）。
    覆盖大比分屠杀的尾部分布——对 Spain vs Cape Verde 这种强弱碰
    会给出例如 +1~+4，而不是早先卡死的 +1~+3。
    """
    from scipy.stats import poisson
    import numpy as np
    
    # 计算 9×9 联合分布的净胜球边际分布
    max_g = 9
    diff_probs = {}
    pa = poisson.pmf(np.arange(max_g + 1), lam_a)
    pb = poisson.pmf(np.arange(max_g + 1), lam_b)
    for i in range(max_g + 1):
        for j in range(max_g + 1):
            d = i - j
            diff_probs[d] = diff_probs.get(d, 0) + pa[i] * pb[j]
    
    # 找 [P20, P80] 中央 60% 区间（防止单点主导）
    diffs_sorted = sorted(diff_probs.keys())
    cum = 0.0
    p20_d = p80_d = None
    for d in diffs_sorted:
        cum += diff_probs[d]
        if p20_d is None and cum >= 0.20:
            p20_d = d
        if cum >= 0.80:
            p80_d = d
            break
    if p20_d is None: p20_d = diffs_sorted[0]
    if p80_d is None: p80_d = diffs_sorted[-1]
    
    def fmt(d):
        if d > 0: return f"+{d}"
        if d == 0: return "0"
        return str(d)  # d<0 自带负号
    
    if p20_d == p80_d:
        return fmt(p20_d)
    return f"{fmt(p20_d)}~{fmt(p80_d)}"


def _quick_match_preview(team_a: str, team_b: str,
                          neutral: bool = True,
                          venue_city: Optional[str] = None,
                          channel: Optional[str] = None) -> Dict[str, Any]:
    """
    单场预测：Elo + Poisson 集成 + Synth 球队级调整注入 + 主场加成
    
    流程：
      1. 基础 Elo 来自 teams.json
      2. **B 方案**：叠加 synth 调整（adj_health/context/psych/squad_value）
         1pp 夺冠调整 ≈ 6 Elo 单场调整
      3. **B+ 方案**：主场加成
         若 venue_city 在三国境内 + 主队是该国 → +30 Elo（≈ +5pp 单场胜率）
         未提供 venue_city 时尝试自动查 group_schedule.json
      4. Elo + Poisson 集成（DC + Bivariate + ZIGP）
    """
    from models.elo_engine import match_probabilities
    from models.poisson_model import predict_match_ensemble as pm

    teams = _load_teams()
    ta = teams.get(team_a)
    tb = teams.get(team_b)
    if not ta or not tb:
        return {"error": f"unknown team: {team_a if not ta else team_b}"}
    
    # ===== B 方案：注入 synth 球队级调整 =====
    PP_TO_ELO = 6.0
    
    synth = _load_synth(channel)
    synth_a = synth.get(team_a, {})
    synth_b = synth.get(team_b, {})
    
    def total_adj_pp(s: Dict) -> float:
        return (s.get("adj_health", 0) + s.get("adj_context", 0)
                + s.get("adj_psych", 0) + s.get("adj_squad_value", 0))
    
    adj_a_pp = total_adj_pp(synth_a)
    adj_b_pp = total_adj_pp(synth_b)
    
    ta_adj = dict(ta)
    tb_adj = dict(tb)
    ta_adj["elo"] = ta["elo"] + adj_a_pp * PP_TO_ELO
    tb_adj["elo"] = tb["elo"] + adj_b_pp * PP_TO_ELO
    
    sv_a = synth_a.get("adj_squad_value", 0)
    sv_b = synth_b.get("adj_squad_value", 0)
    ta_adj["xg_for"] = ta["xg_for"] * (1 + sv_a / 100.0)
    tb_adj["xg_for"] = tb["xg_for"] * (1 + sv_b / 100.0)
    
    # ===== B+ 方案：主场加成 =====
    # 60 Elo ≈ +8-10pp 单场胜率（参考 LLM 分析"USA 主场优势通常值 10-15%"下限）
    HOME_ELO_BOOST = 60
    HOST_NATION_VENUES = {
        "USA": {"Atlanta", "Boston", "Dallas", "Houston", "Kansas City",
                "Los Angeles", "Miami", "New York/New Jersey", "Philadelphia",
                "San Francisco Bay Area", "Seattle"},
        "Canada": {"Toronto", "Vancouver"},
        "Mexico": {"Mexico City", "Guadalajara", "Monterrey"},
    }
    
    # 未提供 venue_city 时尝试自动查 group_schedule
    if venue_city is None:
        try:
            sched_path = DATA_RAW / "group_schedule.json"
            if sched_path.exists():
                sched = json.load(open(sched_path))
                for m in sched.get("matches", []):
                    if (m["team_a"] == team_a and m["team_b"] == team_b):
                        venue_city = m.get("venue_city")
                        break
                    if (m["team_a"] == team_b and m["team_b"] == team_a):
                        # 主客对调：venue_city 含义不变（场地不变），但主队会反过来
                        # 这种情况实际不会发生在 group_schedule（赛程定主客），但兜底
                        venue_city = m.get("venue_city")
                        break
        except Exception:
            pass
    
    home_boost_a = 0
    home_boost_b = 0
    home_info = None
    if venue_city:
        for host, cities in HOST_NATION_VENUES.items():
            if venue_city in cities:
                if team_a == host:
                    home_boost_a = HOME_ELO_BOOST
                    home_info = f"{host} 主场（{venue_city}）"
                elif team_b == host:
                    home_boost_b = HOME_ELO_BOOST
                    home_info = f"{host} 主场（{venue_city}）"
                break
    
    ta_adj["elo"] += home_boost_a
    tb_adj["elo"] += home_boost_b

    elo_p = match_probabilities(ta_adj["elo"], tb_adj["elo"])
    poi = pm(ta_adj, tb_adj)
    p_a = (elo_p["p_win_a"] + poi["outcome"]["p_win_a"]) / 2
    p_d = (elo_p["p_draw"] + poi["outcome"]["p_draw"]) / 2
    p_b = (elo_p["p_win_b"] + poi["outcome"]["p_win_b"]) / 2

    # 全局 Top 3 比分（包括所有结果）
    top_scores_all = [
        {"score": f"{a}-{b}", "prob_pct": round(p * 100, 1), "kind": ("home" if a > b else ("away" if b > a else "draw"))}
        for (a, b), p in poi["top_scorelines"][:5]
    ]

    # 预测胜方：始终选 argmax（淘汰赛传播链需要球队名），但额外标记胶着场
    p_main = max(p_a, p_b)
    p_diff = abs(p_a - p_b)
    is_toss_up = bool((p_diff < 0.05) and (p_d > p_main - 0.05))
    
    if p_a >= p_b:
        winner = team_a
        win_conf = p_a + p_d * 0.5
        winner_kind = "home"
    else:
        winner = team_b
        win_conf = p_b + p_d * 0.5
        winner_kind = "away"
    
    # ---- 条件化"预测比分"（取符合胜方类别的 Top 2）----
    # 若势均力敌 → 用全局 Top 2（可能含平局）
    # 若有预测胜方 → 从该胜方类别（home/away）里取 Top 2
    # 这样既保证"叙事一致"，又给用户两个候选避免单点误导
    if is_toss_up:
        likely_pair = top_scores_all[:2]
    else:
        matching = [s for s in top_scores_all if s["kind"] == winner_kind]
        likely_pair = matching[:2] if matching else top_scores_all[:2]
    
    if not likely_pair:
        most_likely_score = "-"
    elif len(likely_pair) == 1:
        s = likely_pair[0]
        most_likely_score = f"{s['score']} ({s['prob_pct']:.0f}%)"
    else:
        s1, s2 = likely_pair[0], likely_pair[1]
        most_likely_score = f"{s1['score']} ({s1['prob_pct']:.0f}%) / {s2['score']} ({s2['prob_pct']:.0f}%)"
    
    # 给前端的 top_scorelines 返回 Top 3（不限类别，供单场预测页详情用）
    top_scores = [{"score": s["score"], "prob_pct": s["prob_pct"]} for s in top_scores_all[:3]]
    
    # 净胜球区间：统一**主队视角**（正数 = 主胜，负数 = 客胜）
    goal_diff_str = _goal_diff_range(poi["lambda_a"], poi["lambda_b"])

    return {
        "team_a": team_a, "team_b": team_b,
        "p_win_a": round(p_a, 4),
        "p_draw": round(p_d, 4),
        "p_win_b": round(p_b, 4),
        "lambda_a": poi["lambda_a"],
        "lambda_b": poi["lambda_b"],
        "predicted_winner": winner,
        "is_toss_up": is_toss_up,
        "winner_confidence_pct": round(win_conf * 100, 1),
        "top_scorelines": top_scores,
        "score_model": poi.get("model"),
        # B 方案：synth 球队级调整审计字段
        "synth_adjustment": {
            f"{team_a}_total_pp": round(adj_a_pp, 2),
            f"{team_a}_elo_boost": round(adj_a_pp * PP_TO_ELO, 1),
            f"{team_b}_total_pp": round(adj_b_pp, 2),
            f"{team_b}_elo_boost": round(adj_b_pp * PP_TO_ELO, 1),
            "PP_TO_ELO": PP_TO_ELO,
            f"{team_a}_breakdown": {
                "health": synth_a.get("adj_health", 0),
                "context": synth_a.get("adj_context", 0),
                "psych": synth_a.get("adj_psych", 0),
                "squad_value": synth_a.get("adj_squad_value", 0),
            },
            f"{team_b}_breakdown": {
                "health": synth_b.get("adj_health", 0),
                "context": synth_b.get("adj_context", 0),
                "psych": synth_b.get("adj_psych", 0),
                "squad_value": synth_b.get("adj_squad_value", 0),
            },
        },
        # B+ 方案：主场加成审计
        "home_advantage": {
            "venue_city": venue_city,
            f"{team_a}_elo_boost": home_boost_a,
            f"{team_b}_elo_boost": home_boost_b,
            "info": home_info,
        },
        # 深度分析报告体例 风格扩展字段
        "forecast_format": {
            "胜率_区间": _win_range(win_conf),
            "净胜球_区间": goal_diff_str,
            "净胜球_说明": "主队视角（正数=主胜，负数=客胜，0~+1 含平局）",
            "置信度档次": _confidence_label(win_conf),
            "预测比分": most_likely_score,
        },
    }


def api_match_bias(refresh: bool = False) -> Dict[str, Any]:
    """
    单场市场偏差检测：72 场赛程级 Kalshi vs 模型对比
    
    若 refresh=True 会强制重抓 Kalshi 并重算偏差；否则读已有 match_bias.json。
    """
    sys.path.insert(0, str(Path(__file__).parent.parent))
    if refresh:
        from data.fetch_kalshi_match_odds import build as fetch_kalshi
        from models.match_bias_detector import detect_match_bias
        fetch_kalshi()
        return detect_match_bias(save=True)
    
    data = _load_match_bias()
    if not data:
        return {"error": "match_bias.json 不存在，请用 ?refresh=1 触发首次抓取"}
    return data


def api_market_bias(refresh: bool = False) -> Dict[str, Any]:
    """
    市场偏差检测：48 队模型 vs 市场夺冠概率对比
    
    Args:
        refresh: True 时强制重跑检测（默认读最新快照）
    """
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from models.market_bias_detector import detect_market_bias, load_latest_snapshot, list_snapshots
    
    if refresh:
        data = detect_market_bias(save_snapshot=True)
    else:
        data = load_latest_snapshot()
    
    # 附带最近 5 个快照的元数据（供时间序列展示）
    data["history"] = list_snapshots(limit=5)
    return data


def api_group_schedule() -> Dict[str, Any]:
    """
    72 场小组赛完整赛程 + 预测 / 实际结果
    
    每场返回：
      - 基本信息：group, match_id, date, time_local, venue, venue_city, team_a, team_b
      - 状态：status = "played" / "scheduled"
      - 已结束场次：含 actual_result {score, winner, source}
      - 未开始场次：含 prediction {p_win_a, p_draw, p_win_b, 预测比分, 净胜球区间, 置信度, 预测胜方}
    """
    sched_path = DATA_RAW / "group_schedule.json"
    if not sched_path.exists():
        return {"error": "group_schedule.json 不存在"}
    sched = json.load(open(sched_path))
    
    res_path = DATA_OUTPUTS / "group_results.json"
    results = {}
    if res_path.exists():
        results = json.load(open(res_path)).get("results", {})
    
    def make_key(m):
        return f"{m['team_a']} vs {m['team_b']} @ {m['date']}"
    
    out_matches = []
    for m in sched["matches"]:
        entry = dict(m)  # 保留所有原字段
        key = make_key(m)
        actual = results.get(key)
        
        if actual:
            entry["status"] = "played"
            entry["actual_result"] = actual
        else:
            entry["status"] = "scheduled"
        
        # 不论 played 还是 scheduled 都跑预测（已结束场次也展示，方便 vs 实际对比）
        try:
            pred = _quick_match_preview(m["team_a"], m["team_b"],
                                          venue_city=m.get("venue_city"))
            if "error" not in pred:
                entry["prediction"] = {
                    "p_win_a": pred["p_win_a"],
                    "p_draw": pred["p_draw"],
                    "p_win_b": pred["p_win_b"],
                    "predicted_winner": pred["predicted_winner"],
                    "winner_confidence_pct": pred["winner_confidence_pct"],
                    "lambda_a": pred["lambda_a"],
                    "lambda_b": pred["lambda_b"],
                    "forecast_format": pred.get("forecast_format", {}),
                }
                # 已结束场次：标注预测是否命中
                if actual:
                    pred_w = pred["predicted_winner"]
                    actual_w = actual.get("winner")
                    entry["prediction_hit"] = (pred_w == actual_w)
        except Exception as e:
            entry["prediction_error"] = str(e)
        
        # 单场市场偏差（Kalshi vs 模型）— 只在 Kalshi 有覆盖时附加
        bias = _mb_lookup(m["team_a"], m["team_b"], m["date"])
        if bias:
            entry["market_bias"] = {
                "kalshi": bias["kalshi"],
                "kalshi_event_ticker": bias.get("event_ticker"),  # 用于构造 Kalshi 详情页 URL
                "delta_pp": bias["delta_pp"],
                "level": bias["level"],
                "level_text": bias["level_text"],
                "upset_signal": bias.get("upset_signal"),
            }
        
        # 双方伤病（来自 injury_radar），有伤才附加
        inj_a = _inj_lookup(m["team_a"])
        inj_b = _inj_lookup(m["team_b"])
        if inj_a:
            entry["team_a_injuries"] = inj_a
        if inj_b:
            entry["team_b_injuries"] = inj_b
        
        # LLM 终审归因（仅对候选爆冷场跑过）
        upset = _upset_lookup(m["team_a"], m["team_b"], m["date"])
        if upset and "label" in upset:
            entry["llm_judgment"] = {
                "label": upset["label"],
                "label_text": upset.get("label_text"),
                "label_color": upset.get("label_color"),
                "confidence": upset.get("confidence"),
                "key_evidence": upset.get("key_evidence", []),
                "rationale": upset.get("rationale"),
                "suggested_action": upset.get("suggested_action"),
                "weak_team": upset.get("weak_team"),
            }
        
        out_matches.append(entry)
    
    # 统计
    n_played = sum(1 for m in out_matches if m["status"] == "played")
    
    return {
        "n_matches": len(out_matches),
        "n_played": n_played,
        "n_scheduled": len(out_matches) - n_played,
        "matches": out_matches,
    }


def api_critical_compare() -> Dict[str, Any]:
    """
    LLM 微调影响对比面板
    
    返回三个维度的"修正前 vs 修正后"对比：
      1. 全局夺冠概率变化（Top 12）
      2. 17 场胶着对阵修正一览（含 5 Agent 详细投票）
      3. 关键剧本变化（半决赛 / 决赛对阵）
    """
    teams = _load_teams()
    ca = _load_critical_adjustments()
    if not ca:
        return {"error": "critical_node_adjustments.json 不存在，请先跑 run_critical_nodes.py"}
    
    # 加载前后两版 MC 概率
    mc_after = _load_mc()  # 已修正版（默认就是修正后）
    p_pre = DATA_OUTPUTS / "mc_simulation_n100000_pre_critical.json"
    mc_before = json.load(open(p_pre)) if p_pre.exists() else {}
    
    # 加载前后两版最可能剧本
    ml_after = _load_most_likely() or {}
    p_ml_pre = DATA_OUTPUTS / "most_likely_bracket_pre_critical.json"
    ml_before = json.load(open(p_ml_pre)) if p_ml_pre.exists() else {}
    
    # ===== 维度 1：全局夺冠概率变化 =====
    champion_changes = []
    # 取修正后 Top 12
    top_teams = sorted(mc_after.items(),
                        key=lambda x: -x[1].get("champion", 0))[:12]
    for name, d_after in top_teams:
        p_after = d_after.get("champion", 0) * 100
        p_before = mc_before.get(name, {}).get("champion", 0) * 100
        delta = p_after - p_before
        champion_changes.append({
            "team": name,
            "elo": teams.get(name, {}).get("elo"),
            "champion_pct_before": round(p_before, 2),
            "champion_pct_after": round(p_after, 2),
            "delta_pp": round(delta, 2),
        })
    
    # ===== 维度 2：17 场胶着对阵详细列表 =====
    adjustments_detail = []
    for key, entry in ca.get("adjustments", {}).items():
        adjustments_detail.append({
            "round": entry["round"],
            "match_id": entry["match_id"],
            "team_a": entry["team_a"],
            "team_b": entry["team_b"],
            "mc_p_win_a_decisive": round(entry.get("mc_p_win_a_decisive", 0) * 100, 1),
            "applied": entry.get("applied", False),
            "final_adjustment_pp": entry.get("final_adjustment_pp"),
            "consensus_winner": entry.get("consensus_winner"),
            "majority_count": entry.get("majority_count"),
            "total_agents": entry.get("total_agents"),
            "summary": entry.get("summary"),
            "agents": [
                {
                    "name": a.get("agent_name") or a.get("agent"),
                    "winner": a.get("predicted_winner"),
                    "pp": a.get("adjustment_pp"),
                    "rationale": a.get("rationale"),
                }
                for a in entry.get("agents", [])
            ],
        })
    # 按 round + match_id 排序
    round_order = {"r32": 1, "r16": 2, "qf": 3, "sf": 4, "final": 5}
    adjustments_detail.sort(key=lambda x: (round_order.get(x["round"], 9), x["match_id"]))
    
    # ===== 维度 3：关键剧本变化 =====
    scenario_changes = []
    
    def fmt_match(entry):
        if not entry:
            return None
        return {
            "team_a": entry["team_a"],
            "team_b": entry["team_b"],
            "winner": entry.get("winner"),
            "prob_pct": entry.get("prob_pct"),
            "winner_conditional_pct": entry.get("winner_conditional_pct"),
        }
    
    # SF1 & SF2
    for r in ("sf",):
        sf_b = ml_before.get("rounds", {}).get(r, [])
        sf_a = ml_after.get("rounds", {}).get(r, [])
        for i in range(min(len(sf_b), len(sf_a))):
            scenario_changes.append({
                "stage": f"半决赛 SF{sf_a[i]['match_id']}",
                "before": fmt_match(sf_b[i]),
                "after": fmt_match(sf_a[i]),
            })
    # 决赛
    fb = ml_before.get("rounds", {}).get("final")
    fa = ml_after.get("rounds", {}).get("final")
    if fb and fa:
        scenario_changes.append({
            "stage": "决赛",
            "before": fmt_match(fb),
            "after": fmt_match(fa),
        })
    
    # 修正最大的 Top 5（用于摘要）
    applied_only = [a for a in adjustments_detail if a.get("applied")]
    applied_only.sort(key=lambda x: -abs(x.get("final_adjustment_pp", 0)))
    
    return {
        "n_adjustments": len(adjustments_detail),
        "n_applied": sum(1 for a in adjustments_detail if a["applied"]),
        "method": "方案 A2: 5-Agent Byzantine 多数票 + 跨槽位生效（修正 ±pp 应用到任何 MC 模拟里出现此对阵的淘汰赛）",
        "champion_changes": champion_changes,
        "adjustments": adjustments_detail,
        "scenario_changes": scenario_changes,
        "top5_largest_adjustments": applied_only[:5],
    }


def api_path_distribution(top_n: int = 10, channel: Optional[str] = None) -> Dict[str, Any]:
    """
    参考表 3.6.20 风格：每轮 Top N 候选 + MC 概率分布

    输出结构（对齐 参考报告体例）：
      - 每轮列出 Top N 进入该轮概率最高的球队
      - 每队带累积概率（进入该轮 = 进入上一轮的条件概率）
      - 这才是 参考报告 "潜在对手" 的真实数据基础

    channel="ai_phase3" 时使用 AI 加权 MC 通道。
    """
    mc = _load_mc(channel)
    teams = _load_teams()

    def top_by(field: str, n: int = top_n):
        return sorted(
            [(name, info, mc.get(name, {}).get(field, 0))
             for name, info in teams.items() if info.get("group") != "_"],
            key=lambda x: -x[2],
        )[:n]

    def fmt_round(stage_key: str, stage_label: str):
        items = top_by(stage_key)
        return {
            "stage": stage_label,
            "field": stage_key,
            "candidates": [
                {
                    "team": name,
                    "elo": info.get("elo"),
                    "group": info.get("group"),
                    "prob_pct": round(p * 100, 2),
                    "market_pct": round(info.get("market_implied", 0) * 100, 2),
                }
                for name, info, p in items
            ],
        }

    return {
        "method": "MC 100,000 次蒙特卡洛抽样统计 (vs 当前推演面板的贪心 Top-1)",
        "rounds": [
            fmt_round("round_of_32", "进入 32 强"),
            fmt_round("round_of_16", "进入 16 强"),
            fmt_round("quarterfinal", "进入 8 强"),
            fmt_round("semifinal", "进入半决赛"),
            fmt_round("final", "进入决赛"),
            fmt_round("champion", "夺冠"),
        ],
    }


def api_team_path(team_name: str, channel: Optional[str] = None) -> Dict[str, Any]:
    """
    单队夺冠路径分解 — 严格按 路径表体例 表格格式
    
    输出 6 行：
      小组第 1 出线 / 进入 16 强 / 进入 8 强 / 进入半决赛 / 进入决赛 / 夺冠
    每行: 概率区间 + 置信度档次 + 说明

    channel="ai_phase3" 时使用 AI 加权 MC 通道。
    """
    teams = _load_teams()
    mc = _load_mc(channel)
    synth = _load_synth(channel)

    if team_name not in teams:
        return {"error": f"unknown team: {team_name}"}

    t = teams[team_name]
    mc_d = mc.get(team_name, {})
    synth_d = synth.get(team_name, {})

    # 概率
    p_r32 = mc_d.get("round_of_32", 0)
    p_r16 = mc_d.get("round_of_16", 0)
    p_qf = mc_d.get("quarterfinal", 0)
    p_sf = mc_d.get("semifinal", 0)
    p_final = mc_d.get("final", 0)
    p_champ = synth_d.get("final_probability", mc_d.get("champion", 0) * 100) / 100

    # 小组分析
    group = t.get("group")
    group_info = api_groups()["groups"].get(group, {})
    is_predicted_first = group_info.get("predicted_winner") == team_name

    def fmt(p: float, pad: int = 5) -> str:
        return f"{p * 100:>{pad}.0f}-{p * 100 + 5:.0f}%"

    return {
        "team": team_name,
        "elo": t.get("elo"),
        "fifa_rank": t.get("fifa"),
        "group": group,
        "predicted_group_winner": is_predicted_first,
        "市场隐含概率": round(t.get("market_implied", 0) * 100, 1),
        "path_table": [
            {
                "阶段": "小组第 1 出线",
                "概率区间": _win_range(0.85 if is_predicted_first else 0.50),
                "置信度": _confidence_label(0.85 if is_predicted_first else 0.50),
                "说明": f"{group} 组实力差距" + ("明显" if is_predicted_first else "需争夺出线"),
            },
            {
                "阶段": "进入 32 强 (R32)",
                "概率区间": _win_range(p_r32),
                "置信度": _confidence_label(p_r32),
                "说明": "12 小组前 2 + 8 个最佳第 3 名",
            },
            {
                "阶段": "进入 16 强 (R16)",
                "概率区间": _win_range(p_r16),
                "置信度": _confidence_label(p_r16),
                "说明": "32 强轮次对手取决于签运",
            },
            {
                "阶段": "进入 8 强 (QF)",
                "概率区间": _win_range(p_qf),
                "置信度": _confidence_label(p_qf),
                "说明": "淘汰赛压力显著，方差扩大",
            },
            {
                "阶段": "进入半决赛 (SF)",
                "概率区间": _win_range(p_sf),
                "置信度": _confidence_label(p_sf),
                "说明": "可能遭遇同档强队",
            },
            {
                "阶段": "进入决赛",
                "概率区间": _win_range(p_final),
                "置信度": _confidence_label(p_final),
                "说明": "半区路径决定性",
            },
            {
                "阶段": "夺冠",
                "概率区间": _win_range(p_champ),
                "置信度": _confidence_label(p_champ),
                "说明": f"市场隐含约 {round(t.get('market_implied', 0) * 100, 1)}%",
            },
        ],
        "synth_adjustments": {
            "health": synth_d.get("adj_health"),
            "context": synth_d.get("adj_context"),
            "psych": synth_d.get("adj_psych"),
            "squad_value": synth_d.get("adj_squad_value"),
        },
        "final_probability_pct": synth_d.get("final_probability"),
        "market_implied_pct": synth_d.get("market_implied"),
        "bias_pp": synth_d.get("bias_pp"),
    }


def api_match(team_a: str, team_b: str, neutral: bool = True,
                with_llm: bool = False) -> Dict[str, Any]:
    """
    任意两队预测（公开 API）
    
    Args:
        team_a, team_b: 球队名
        neutral: 是否中立场地（默认 True）
        with_llm: 是否调用 LLM 生成分析理由（默认 False，避免延迟）
    
    Returns:
        含数值预测 + 可选的 LLM 文字分析
    """
    teams = _load_teams()
    if team_a not in teams:
        return {"error": f"unknown team: {team_a}",
                 "available_teams": sorted([t for t in teams if teams[t].get("group") != "_"])[:20]}
    if team_b not in teams:
        return {"error": f"unknown team: {team_b}"}

    quick = _quick_match_preview(team_a, team_b, neutral)

    # 队基本面
    ta_info = teams[team_a]
    tb_info = teams[team_b]

    result = {
        "team_a": {
            "name": team_a,
            "elo": ta_info.get("elo"),
            "fifa_rank": ta_info.get("fifa"),
            "group": ta_info.get("group"),
            "xg_for": ta_info.get("xg_for"),
            "xg_against": ta_info.get("xg_against"),
            "xt_per_90": ta_info.get("xt_per_90"),
            "squad_value_m_eur": ta_info.get("squad_value_m_eur"),
        },
        "team_b": {
            "name": team_b,
            "elo": tb_info.get("elo"),
            "fifa_rank": tb_info.get("fifa"),
            "group": tb_info.get("group"),
            "xg_for": tb_info.get("xg_for"),
            "xg_against": tb_info.get("xg_against"),
            "xt_per_90": tb_info.get("xt_per_90"),
            "squad_value_m_eur": tb_info.get("squad_value_m_eur"),
        },
        "prediction": quick,
        "neutral_venue": neutral,
    }

    # 可选 LLM 分析
    if with_llm:
        try:
            from web.match_analyzer import analyze_match_with_llm
            llm_analysis = analyze_match_with_llm(team_a, team_b, quick,
                                                    ta_info, tb_info)
            result["llm_analysis"] = llm_analysis
        except Exception as e:
            result["llm_analysis"] = {"error": str(e), "available": False}

    return result


# ============ CLI 自测 ============
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("endpoint", choices=["groups", "r32", "r16", "qf", "sf",
                                          "final", "bracket", "match", "team_path"])
    ap.add_argument("--a", help="team A for match / team for team_path")
    ap.add_argument("--b", help="team B for match")
    ap.add_argument("--with-llm", action="store_true")
    args = ap.parse_args()

    if args.endpoint == "groups":
        result = api_groups()
    elif args.endpoint == "r32":
        result = api_r32()
    elif args.endpoint == "r16":
        result = api_r16()
    elif args.endpoint == "qf":
        result = api_qf()
    elif args.endpoint == "sf":
        result = api_sf()
    elif args.endpoint == "final":
        result = api_final_match()
    elif args.endpoint == "bracket":
        result = api_bracket()
    elif args.endpoint == "match":
        if not (args.a and args.b):
            print("--a and --b required for match")
            sys.exit(1)
        result = api_match(args.a, args.b, with_llm=args.with_llm)
    elif args.endpoint == "team_path":
        if not args.a:
            print("--a required for team_path")
            sys.exit(1)
        result = api_team_path(args.a)

    print(json.dumps(result, ensure_ascii=False, indent=2)[:3000])