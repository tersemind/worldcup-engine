"""
赛程感知动态调度
================

读 group_schedule.json 的当天比赛，给 scheduler 提供两类决策：

1. 赛后窗口（kickoff+110~150min，间隔 5min）：
     给 group_results / live_events 临时加密，
     保证赛后 10min 内拿到 ESPN/FIFA 终场比分。

2. 赛前窗口（开赛前 30/15/5 分钟）：
     给 cascade（端到端预测）+ lineups + referee + h2h 临时加密。
     具体：cascade 由 scheduler.maybe_pre_match_cascade() 在到达 mark
           时强制 mark_dirty 触发。

设计原则：
  - 完全可关闭（match_aware.enabled=false）
  - 任何异常直接退回旧 interval（绝不阻塞调度）
  - 时区统一以「场馆当地时区 → 系统本地时区」为准；
    简化处理：直接用 group_schedule.json 中 date+time_local，
    时区差异交给 schedule_refresh 同步。
  - 无依赖（只用 json + datetime）
"""
from __future__ import annotations
import json
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple

ROOT = Path(__file__).parent.parent.parent
SCHEDULE_PATH = ROOT / "data" / "raw" / "group_schedule.json"
CONFIG_PATH = ROOT / "data" / "raw" / "scheduler_config.json"

# 60s 缓存（避免每个 task.should_run 都读盘）
_cache: Dict = {"matches": None, "ts": 0.0, "ttl_sec": 60.0}
_lock = threading.Lock()

# 默认配置（scheduler_config.json 无 match_aware 节点时用）
DEFAULT_MATCH_AWARE = {
    "enabled": True,
    "post_match": {
        "start_min": 110,         # kickoff 后 N 分钟开始密集刷
        "end_min": 150,           # kickoff 后 N 分钟结束密集刷
        "interval_min": 5,        # 窗口期间隔
        "tasks": ["group_results", "live_events"],
    },
    "pre_match": {
        "trigger_min_before": [30, 15, 5],  # 开赛前各触发一次 cascade
        "tasks": ["cascade", "lineups", "referee", "h2h"],
        "tight_interval_min": 5,            # 非 cascade 任务在赛前 60min 内的间隔
        "tight_window_min_before": 60,
    },
}

# 时区粗略映射（FIFA 2026 主办地：EDT/EST/CDT/CST/MDT/MST/PDT/PST）
# 调度器跑在 macOS 本地（北京时间）；这里转 UTC，再让 datetime 在本地比较
_TZ_OFFSET_HOURS = {
    "EDT": -4, "EST": -5,
    "CDT": -5, "CST": -6,
    "MDT": -6, "MST": -7,
    "PDT": -7, "PST": -8,
    "ADT": -3, "AST": -4,
    "AKDT": -8, "AKST": -9,
    "HDT": -9, "HST": -10,
    "UTC": 0, "GMT": 0,
    "CEST": 2, "CET": 1,
    "BST": 1,
    "JST": 9, "KST": 9,
    "AEST": 10, "AEDT": 11,
    "CST_CN": 8,  # 中国北京时间（保留以防）
    "ART": -3,    # 阿根廷
    "BRT": -3, "BRST": -2,  # 巴西
}


def _load_config() -> dict:
    """加载 scheduler_config.json 中的 match_aware 节点（带默认）。"""
    try:
        if CONFIG_PATH.exists():
            cfg = json.load(open(CONFIG_PATH))
            ma = cfg.get("match_aware") or {}
            # 浅合并默认值
            merged = json.loads(json.dumps(DEFAULT_MATCH_AWARE))
            for k, v in ma.items():
                if isinstance(v, dict) and isinstance(merged.get(k), dict):
                    merged[k].update(v)
                else:
                    merged[k] = v
            return merged
    except Exception:
        pass
    return DEFAULT_MATCH_AWARE


def _parse_kickoff_to_local(date: str, time_local: str, tz: str) -> Optional[datetime]:
    """把 group_schedule 的 (date, time_local, tz) 转为「调度器本机时区」的 datetime。

    例：("2026-06-11", "15:00", "EDT") 表示 6/11 15:00 EDT = 6/11 19:00 UTC
        若调度器跑在北京（UTC+8）→ 6/12 03:00 北京时间。

    失败返回 None（调用方自动忽略该场次）。
    """
    try:
        offset = _TZ_OFFSET_HOURS.get((tz or "").upper())
        if offset is None:
            # 时区未识别 → 直接当作本地时间用，至少不报错
            return datetime.strptime(f"{date} {time_local}", "%Y-%m-%d %H:%M")
        # 比赛 UTC 时间 = 当地时间 - offset
        kickoff_naive = datetime.strptime(f"{date} {time_local}", "%Y-%m-%d %H:%M")
        kickoff_utc_ts = kickoff_naive.timestamp() - offset * 3600
        # macOS time.localtime 默认用系统时区（北京/UTC+8）
        return datetime.fromtimestamp(kickoff_utc_ts)
    except Exception:
        return None


def _load_matches_today(now: Optional[datetime] = None) -> List[dict]:
    """加载「今天前后 48h」的全部场次（含 kickoff_local: datetime）。

    带 60s 缓存。
    """
    now = now or datetime.now()
    with _lock:
        if _cache["matches"] is not None and (time.time() - _cache["ts"]) < _cache["ttl_sec"]:
            return _cache["matches"]
        matches: List[dict] = []
        try:
            if SCHEDULE_PATH.exists():
                raw = json.load(open(SCHEDULE_PATH))
                for m in raw.get("matches", []):
                    ko = _parse_kickoff_to_local(m.get("date", ""), m.get("time_local", ""), m.get("tz", ""))
                    if ko is None:
                        continue
                    # 仅保留 ±48h 窗口内的（节省遍历）
                    if abs((ko - now).total_seconds()) > 48 * 3600:
                        continue
                    matches.append({
                        "team_a": m.get("team_a"),
                        "team_b": m.get("team_b"),
                        "kickoff": ko,
                        "match_id": m.get("match_id"),
                        "group": m.get("group"),
                    })
        except Exception:
            pass
        _cache["matches"] = matches
        _cache["ts"] = time.time()
        return matches


# ─────────── 公共 API ───────────

def dynamic_interval(task_name: str, default_interval_min: int, now: Optional[datetime] = None) -> Optional[int]:
    """根据当前时刻 + 当天比赛日程，返回「赛程感知后」的临时 interval（分钟）。

    Args:
      task_name: 任务名（kalshi / polymarket / group_results / cascade / ...）
      default_interval_min: 该任务默认 interval（来自 stage_overrides 或 cfg.interval_min）
      now: 测试用，默认 datetime.now()

    Returns:
      int — 应使用的临时 interval（分钟）
      None — match_aware 关闭或不命中任何窗口 → 调用方应回落到 default
    """
    cfg = _load_config()
    if not cfg.get("enabled", True):
        return None

    now = now or datetime.now()
    matches = _load_matches_today(now)
    if not matches:
        return None  # 当天没比赛 → 走默认节奏

    # ── 赛后窗口（group_results / live_events）──
    pm = cfg.get("post_match", {})
    if task_name in pm.get("tasks", []):
        start_min = int(pm.get("start_min", 110))
        end_min = int(pm.get("end_min", 150))
        for m in matches:
            elapsed_min = (now - m["kickoff"]).total_seconds() / 60.0
            if start_min <= elapsed_min <= end_min:
                return int(pm.get("interval_min", 5))

    # ── 赛前紧凑窗口（lineups / referee / h2h）──
    pre = cfg.get("pre_match", {})
    if task_name in pre.get("tasks", []) and task_name != "cascade":
        tight_window = int(pre.get("tight_window_min_before", 60))
        for m in matches:
            min_before = (m["kickoff"] - now).total_seconds() / 60.0
            if 0 < min_before <= tight_window:
                return int(pre.get("tight_interval_min", 5))

    return None  # 未命中任何动态窗口


def pre_match_cascade_marks(now: Optional[datetime] = None, lookback_sec: int = 60) -> List[Tuple[int, dict]]:
    """返回当下需要触发 cascade 的「赛前 mark」列表。

    每 tick (60s) 调一次。若某场比赛的某个 mark（如开赛前 30min）落在
    [now-lookback, now] 区间，就返回该 mark + 比赛信息，
    交给调用方决定是否 mark_dirty 触发 cascade。

    Returns:
      [(min_before, match_info), ...]  空列表 = 当前 tick 无需触发
    """
    cfg = _load_config()
    if not cfg.get("enabled", True):
        return []
    pre = cfg.get("pre_match", {})
    if "cascade" not in pre.get("tasks", []):
        return []

    now = now or datetime.now()
    marks = pre.get("trigger_min_before", [30, 15, 5])
    out: List[Tuple[int, dict]] = []
    for m in _load_matches_today(now):
        ko = m["kickoff"]
        for mb in marks:
            target = ko - timedelta(minutes=int(mb))
            # 落在 [now - lookback, now] 内 → 该 tick 命中
            delta = (now - target).total_seconds()
            if 0 <= delta <= lookback_sec:
                out.append((int(mb), m))
    return out


def describe_today(now: Optional[datetime] = None) -> dict:
    """诊断用：返回当天比赛 + 各场距离 kickoff 的相对分钟数。"""
    now = now or datetime.now()
    matches = _load_matches_today(now)
    out = []
    for m in matches:
        ko = m["kickoff"]
        dt_min = (ko - now).total_seconds() / 60.0
        out.append({
            "team_a": m["team_a"],
            "team_b": m["team_b"],
            "kickoff_local": ko.strftime("%Y-%m-%d %H:%M"),
            "min_until_kickoff": round(dt_min, 1),
        })
    return {
        "now": now.strftime("%Y-%m-%d %H:%M:%S"),
        "n_matches_in_48h": len(matches),
        "matches": out,
    }


if __name__ == "__main__":
    # 自检
    print(json.dumps(describe_today(), ensure_ascii=False, indent=2))
    print()
    print("dynamic_interval samples:")
    for tn in ["group_results", "live_events", "lineups", "kalshi"]:
        v = dynamic_interval(tn, 60)
        print(f"  {tn}: {v}  ({'override' if v is not None else 'use default'})")
    print()
    print("pre_match_cascade_marks (当前 tick):")
    print(pre_match_cascade_marks())
