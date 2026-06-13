"""quant tick 落盘工具。

两个产物：
  - data/outputs/in_match_live.json   每 tick 覆盖（snapshot, web 直读）
  - data/outputs/in_match_ticks.jsonl append-only（回测/审计）

设计：
  - 写入失败不抛异常，仅 log warning
  - jsonl 单行紧凑（核心字段，便于 grep + 不膨胀）
  - 自动按周 rotate（可选，S2 不上）
"""
from __future__ import annotations
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List

ROOT = Path(__file__).parent.parent.parent
OUT_DIR = ROOT / "data" / "outputs"

LIVE_SNAPSHOT_PATH = OUT_DIR / "in_match_live.json"
TICKS_JSONL_PATH = OUT_DIR / "in_match_ticks.jsonl"

log = logging.getLogger("quant.tick_writer")


def write_snapshot(payload: dict) -> bool:
    """覆盖写 in_match_live.json。失败返回 False 不抛。"""
    try:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        # 原子写：先 .tmp 再 rename
        tmp = LIVE_SNAPSHOT_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        tmp.replace(LIVE_SNAPSHOT_PATH)
        return True
    except Exception as e:
        log.warning(f"write_snapshot 失败: {type(e).__name__}: {e}")
        return False


def append_tick(tick_row: dict) -> bool:
    """追加一行到 in_match_ticks.jsonl。失败返回 False 不抛。"""
    try:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(TICKS_JSONL_PATH, "a") as f:
            f.write(json.dumps(tick_row, ensure_ascii=False) + "\n")
        return True
    except Exception as e:
        log.warning(f"append_tick 失败: {type(e).__name__}: {e}")
        return False


def build_tick_row(snapshot: dict) -> dict:
    """从完整 snapshot 抽核心字段构造 jsonl 行（紧凑版）。

    snapshot 结构见 live_trading_loop.run_tick。
    """
    live = snapshot.get("live_predictions", []) or []
    arb = snapshot.get("arbitrage_signals", []) or []
    # 只挑 BUY/SELL（NEUTRAL 不进 jsonl，节省空间）
    n_buy = sum(1 for s in arb if s.get("type") == "BUY")
    n_sell = sum(1 for s in arb if s.get("type") == "SELL")
    top_buy = max((s.get("edge_pp", 0) for s in arb if s.get("type") == "BUY"), default=0.0)
    top_sell = min((s.get("edge_pp", 0) for s in arb if s.get("type") == "SELL"), default=0.0)

    live_compact = []
    for p in live:
        v2 = p.get("v2") or {}
        live_compact.append({
            "match": f"{p.get('team_a')} vs {p.get('team_b')}",
            "score": f"{p.get('score_a', '-')}-{p.get('score_b', '-')}",
            "min": p.get("elapsed_min"),
            "phase": p.get("phase"),
            "p_win_a": p.get("p_win_a"),
            "p_draw": p.get("p_draw"),
            "p_win_b": p.get("p_win_b"),
            "p_win_a_v1plus_v2": p.get("p_win_a_v1plus_v2"),
            "p_draw_v1plus_v2": p.get("p_draw_v1plus_v2"),
            "p_win_b_v1plus_v2": p.get("p_win_b_v1plus_v2"),
            "delta_elo_a": p.get("delta_elo_a"),
            "delta_total_pp": p.get("delta_total_pp"),
            "v2_delta_pp": v2.get("delta_pp_a", 0.0),
            "v2_ok": v2.get("ok", False),
            "v1_warnings": p.get("v1_warnings", []),
        })

    anomalies = snapshot.get("market_anomalies", []) or []
    # 异动 top-3 预览
    anom_preview = [{"ticker": a.get("event_ticker"), "leg": a.get("leg"),
                     "delta_pp": a.get("delta_pp")} for a in anomalies[:3]]

    return {
        "ts": snapshot.get("ts"),
        "elapsed_sec": snapshot.get("elapsed_sec"),
        "n_live": len(live),
        "live": live_compact,
        "n_buy": n_buy,
        "n_sell": n_sell,
        "top_buy_edge_pp": round(top_buy, 2),
        "top_sell_edge_pp": round(top_sell, 2),
        "n_anomalies": len(anomalies),
        "anomalies_preview": anom_preview,
    }


def write_all(snapshot: dict) -> dict:
    """一次性写 snapshot + 追加 jsonl。返回写入状态。"""
    snap_ok = write_snapshot(snapshot)
    tick_ok = append_tick(build_tick_row(snapshot))
    return {"snapshot_written": snap_ok, "tick_appended": tick_ok}
