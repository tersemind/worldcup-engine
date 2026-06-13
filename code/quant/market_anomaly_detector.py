"""
v3' Kalshi 市场异动检测器
=========================

核心设计：
  - 不进 ΔE 公式：模型不抄市场答案，避免循环依赖
  - 仅作 dirty flag：触发即标记"立刻关注 + 下次 tick 强制重算"
  - 5min 滑动窗口；任意 leg (p_win_a / p_draw / p_win_b) 跳变 ≥3pp 即异动

数据流：
  外部 (live_trading_loop) 每 tick 调 record_snapshot(matches)
       ↓
  内部维护 history: {event_ticker: deque[(ts, p_a, p_d, p_b)]}（容量 ~10）
       ↓
  detect_anomalies() 与 5min 前最早入队点比较
       ↓
  返回 list[{event_ticker, leg, prev_pp, now_pp, delta_pp, ts}]

线程安全：worldcup scheduler 串行调用，无需锁。
持久化：内存为主；可选 dump_state/load_state 在 scheduler 重启时恢复。
"""
from __future__ import annotations

import json
import time
from collections import deque
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

WINDOW_SEC = 300.0          # 5min 滑窗
THRESHOLD_PP = 3.0          # 异动阈值
MAX_HISTORY = 12            # 每 ticker 最多 12 个点
STATE_FILE = Path(__file__).resolve().parents[2] / "data" / "outputs" / "quant_anomaly_state.json"

# 内部状态：{event_ticker: deque[(ts, p_a, p_d, p_b)]}
_HISTORY: Dict[str, deque] = {}


def _safe_pct(p: Any) -> Optional[float]:
    """统一成 0-100 的 pp 单位（输入 0-1 prob → ×100；已是 pp 则原样）"""
    try:
        v = float(p)
    except (ValueError, TypeError):
        return None
    if 0.0 <= v <= 1.0:
        return v * 100.0
    if 0.0 <= v <= 100.0:
        return v
    return None


def _ticker_of(m: Dict[str, Any]) -> Optional[str]:
    """容错取 ticker"""
    return m.get("event_ticker") or m.get("ticker") or m.get("market_id")


def record_snapshot(matches: List[Dict[str, Any]], ts: Optional[float] = None) -> int:
    """
    记录一次 Kalshi 抓取快照到滑窗历史。

    Args:
        matches: kalshi_match_odds.json 的 matches 列表
        ts:       时间戳（默认 time.time()）

    Returns:
        本次记录到的 ticker 数（输入合法的）
    """
    if not matches:
        return 0
    now = ts if ts is not None else time.time()
    n = 0
    for m in matches:
        if not isinstance(m, dict):
            continue
        tk = _ticker_of(m)
        if not tk:
            continue
        p_a = _safe_pct(m.get("kalshi_p_win_a"))
        p_d = _safe_pct(m.get("kalshi_p_draw"))
        p_b = _safe_pct(m.get("kalshi_p_win_b"))
        if p_a is None and p_d is None and p_b is None:
            continue
        if tk not in _HISTORY:
            _HISTORY[tk] = deque(maxlen=MAX_HISTORY)
        _HISTORY[tk].append((now, p_a, p_d, p_b))
        n += 1
    return n


def _earliest_in_window(dq: deque, now: float) -> Optional[Tuple[float, Optional[float], Optional[float], Optional[float]]]:
    """返回 ≤ window_sec 范围内最早一个点；若所有点都在窗口外则返 None"""
    earliest = None
    for entry in dq:
        ts, *_ = entry
        if now - ts <= WINDOW_SEC:
            earliest = entry
            break
    return earliest


def detect_anomalies(now: Optional[float] = None,
                     threshold_pp: float = THRESHOLD_PP) -> List[Dict[str, Any]]:
    """
    检测当前历史窗口内 |Δp| ≥ threshold_pp 的 ticker。

    Returns:
        [
          {event_ticker, leg, prev_pp, now_pp, delta_pp, window_sec, ts}
          ...
        ]
        无异动返 []
    """
    if now is None:
        now = time.time()
    out: List[Dict[str, Any]] = []
    for tk, dq in _HISTORY.items():
        if len(dq) < 2:
            continue
        earliest = _earliest_in_window(dq, now)
        latest = dq[-1]
        if earliest is None or earliest[0] == latest[0]:
            continue
        labels = ("win_a", "draw", "win_b")
        prev_vals = earliest[1:4]
        now_vals = latest[1:4]
        for leg, pv, nv in zip(labels, prev_vals, now_vals):
            if pv is None or nv is None:
                continue
            delta = nv - pv
            if abs(delta) >= threshold_pp:
                out.append({
                    "event_ticker": tk,
                    "leg": leg,
                    "prev_pp": round(pv, 3),
                    "now_pp": round(nv, 3),
                    "delta_pp": round(delta, 3),
                    "window_sec": round(now - earliest[0], 1),
                    "ts": now,
                })
    # 大异动优先
    out.sort(key=lambda r: abs(r["delta_pp"]), reverse=True)
    return out


def history_size() -> int:
    return len(_HISTORY)


def history_for(ticker: str) -> List[Tuple[float, Optional[float], Optional[float], Optional[float]]]:
    return list(_HISTORY.get(ticker, []))


def reset():
    _HISTORY.clear()


# ====== 可选持久化（重启恢复，避免冷启动丢窗口） ======

def dump_state(path: Path = STATE_FILE) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {tk: list(dq) for tk, dq in _HISTORY.items()}
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps({"saved_at": time.time(), "history": payload}, ensure_ascii=False))
        tmp.replace(path)
        return True
    except Exception:
        return False


def load_state(path: Path = STATE_FILE) -> int:
    """启动时调用恢复历史；返回恢复的 ticker 数"""
    try:
        if not path.exists():
            return 0
        data = json.loads(path.read_text())
        history = data.get("history", {})
        n = 0
        for tk, entries in history.items():
            if not entries:
                continue
            dq = deque(maxlen=MAX_HISTORY)
            for e in entries:
                if isinstance(e, list) and len(e) >= 4:
                    dq.append(tuple(e))
            if dq:
                _HISTORY[tk] = dq
                n += 1
        return n
    except Exception:
        return 0


# =================== CLI demo ===================
if __name__ == "__main__":
    import time as _t
    print("[demo] simulate 5 ticks of one match with a price jump")
    base = _t.time() - 240  # 4min ago
    matches_t1 = [{"event_ticker": "KXWCGAME-DEMO", "kalshi_p_win_a": 0.30,
                   "kalshi_p_draw": 0.30, "kalshi_p_win_b": 0.40}]
    matches_t2 = [{"event_ticker": "KXWCGAME-DEMO", "kalshi_p_win_a": 0.31,
                   "kalshi_p_draw": 0.29, "kalshi_p_win_b": 0.40}]
    matches_t3 = [{"event_ticker": "KXWCGAME-DEMO", "kalshi_p_win_a": 0.30,
                   "kalshi_p_draw": 0.30, "kalshi_p_win_b": 0.40}]
    matches_t4 = [{"event_ticker": "KXWCGAME-DEMO", "kalshi_p_win_a": 0.45,  # 跳 +15pp
                   "kalshi_p_draw": 0.25, "kalshi_p_win_b": 0.30}]

    record_snapshot(matches_t1, ts=base)
    record_snapshot(matches_t2, ts=base + 60)
    record_snapshot(matches_t3, ts=base + 120)
    record_snapshot(matches_t4, ts=base + 240)

    print(f"  history size = {history_size()}")
    anomalies = detect_anomalies(now=base + 240)
    print(f"  found {len(anomalies)} anomalies:")
    for a in anomalies:
        print(f"    {a['event_ticker']} leg={a['leg']} "
              f"{a['prev_pp']:.1f} -> {a['now_pp']:.1f} (Δ{a['delta_pp']:+.1f}pp) "
              f"window={a['window_sec']}s")

    # 测试 dump/load
    ok = dump_state()
    print(f"  dump_state: {ok}")
    reset()
    print(f"  after reset: history_size={history_size()}")
    n = load_state()
    print(f"  load_state restored {n} tickers, history_size={history_size()}")
