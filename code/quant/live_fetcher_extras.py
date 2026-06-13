"""
ESPN summary endpoint 扩展抓取器
=================================

为 quant.live_trading_loop 提供 v2 量化信号所需的扩展字段。

ESPN scoreboard endpoint 只能拿到 score / events（goals/cards/subs）。
这里调 summary endpoint 取 boxscore.teams[].statistics 的：

  - possessionPct      持球率 (0-100)
  - totalShots         射门数
  - shotsOnTarget      射正
  - penaltyKickShots   点球次数
  - penaltyKickGoals   点球破门
  - redCards / yellowCards   补充确认（与 events 流一致性校验）

ESPN 不直接给 xG，由 v2 的 live_stats_features 用 shot* 近似估算。

设计：
  - 全 fail-soft：网络失败 / json 异常 / 字段缺失 都不抛
  - 缓存 30s（避免 60s tick 多次重复 fetch 同一 espn_id）
  - 与 fetch_live_events 解耦，独立 import 不污染主链路
"""
from __future__ import annotations

import json
import time
import urllib.request
import urllib.error
from typing import Dict, Any, Optional, List

ESPN_SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/summary?event={eid}"

# 30s 缓存：避免同一 tick 内 _fetch_parallel 多次或下个 tick 60s 内重复
_CACHE: Dict[str, tuple] = {}  # espn_id -> (timestamp, payload)
_CACHE_TTL_SEC = 30.0

# v2 关心的字段 → 标准化键
STAT_FIELD_MAP = {
    "possessionPct":      "possession_pct",
    "totalShots":         "shots_total",
    "shotsOnTarget":      "shots_on_target",
    "penaltyKickShots":   "penalty_attempts",
    "penaltyKickGoals":   "penalty_goals",
    "redCards":           "red_cards_summary",
    "yellowCards":        "yellow_cards_summary",
    "wonCorners":         "corners",
    "foulsCommitted":     "fouls",
    "offsides":           "offsides",
}


def _safe_float(x, default: float = 0.0) -> float:
    try:
        if isinstance(x, str):
            x = x.replace("%", "").strip()
        return float(x)
    except (ValueError, TypeError):
        return default


def _safe_int(x, default: int = 0) -> int:
    try:
        return int(float(x))
    except (ValueError, TypeError):
        return default


def _parse_team_stats(team_stats_block: Dict[str, Any]) -> Dict[str, Any]:
    """从 boxscore.teams[i] 节点解析 v2 关心字段"""
    out: Dict[str, Any] = {}
    statistics = team_stats_block.get("statistics") or []
    for s in statistics:
        name = s.get("name", "")
        if name in STAT_FIELD_MAP:
            std_key = STAT_FIELD_MAP[name]
            disp = s.get("displayValue", "")
            if "Pct" in name or std_key == "possession_pct":
                out[std_key] = _safe_float(disp)
            else:
                out[std_key] = _safe_int(disp)
    # team meta
    team_obj = team_stats_block.get("team") or {}
    out["_team_id"] = team_obj.get("id")
    out["_team_name"] = team_obj.get("displayName") or team_obj.get("name") or ""
    out["_home_away"] = team_stats_block.get("homeAway", "")
    return out


def fetch_summary(espn_id: str, timeout: int = 8) -> Optional[Dict[str, Any]]:
    """
    抓 ESPN summary endpoint 并解析成 quant.v2 友好 dict。

    返回:
      {
        "espn_id": str,
        "home": {possession_pct, shots_total, shots_on_target, ...},
        "away": {...},
        "fetched_at": <epoch>,
        "ok": True,
      }
    失败时返 {"espn_id": ..., "ok": False, "error": ...} 或 None。
    全 fail-soft 永不抛。
    """
    if not espn_id:
        return None
    eid = str(espn_id)
    now = time.time()

    # cache hit
    if eid in _CACHE:
        ts, cached = _CACHE[eid]
        if now - ts < _CACHE_TTL_SEC:
            return cached

    url = ESPN_SUMMARY_URL.format(eid=eid)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "wc-quant/0.1"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
        data = json.loads(raw)
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, TimeoutError, OSError) as e:
        out = {"espn_id": eid, "ok": False, "error": f"{type(e).__name__}: {e}"}
        _CACHE[eid] = (now, out)
        return out
    except Exception as e:
        out = {"espn_id": eid, "ok": False, "error": f"unexpected: {type(e).__name__}: {e}"}
        _CACHE[eid] = (now, out)
        return out

    bs = (data or {}).get("boxscore") or {}
    teams = bs.get("teams") or []
    home_stats: Dict[str, Any] = {}
    away_stats: Dict[str, Any] = {}
    for t in teams:
        ha = t.get("homeAway", "")
        parsed = _parse_team_stats(t)
        if ha == "home":
            home_stats = parsed
        elif ha == "away":
            away_stats = parsed

    out = {
        "espn_id": eid,
        "ok": True,
        "fetched_at": now,
        "home": home_stats,
        "away": away_stats,
    }
    _CACHE[eid] = (now, out)
    return out


def fetch_summary_batch(espn_ids: List[str], timeout: int = 8) -> Dict[str, Dict[str, Any]]:
    """
    批量抓 summary（顺序执行，依赖上层 ThreadPoolExecutor 做并行）。
    返回 {espn_id: summary_dict}，失败的 id 也保留 {ok:False}。
    """
    out: Dict[str, Dict[str, Any]] = {}
    for eid in espn_ids:
        s = fetch_summary(eid, timeout=timeout)
        if s is not None:
            out[str(eid)] = s
    return out


def cache_size() -> int:
    return len(_CACHE)


def clear_cache():
    _CACHE.clear()


# =================== CLI demo ===================
if __name__ == "__main__":
    import sys
    eid = sys.argv[1] if len(sys.argv) > 1 else "740966"  # 默认拿英超已知有数据的事件
    print(f"[demo] fetch_summary({eid})")
    s = fetch_summary(eid)
    if s and s.get("ok"):
        print(f"  espn_id={s['espn_id']} fetched_at={s['fetched_at']:.0f}")
        for side in ("home", "away"):
            blk = s.get(side, {})
            print(f"  [{side}] {blk.get('_team_name')} ({blk.get('_home_away')})")
            for k in ("possession_pct", "shots_total", "shots_on_target",
                      "penalty_attempts", "penalty_goals",
                      "red_cards_summary", "yellow_cards_summary",
                      "corners", "fouls", "offsides"):
                if k in blk:
                    print(f"    {k} = {blk[k]}")
    else:
        print(f"  FAIL: {s}")
