"""quant 主调度入口：run_tick()。

每 60s 由 scheduler 触发一次。

S2 范围：
  - 仅 v1（live_elo_solver 硬事实反求）
  - 并行 fetch ESPN scoreboard + Kalshi
  - 调 arbitrage_kalshi.generate_kalshi_signals(min_edge=0)
  - 落盘 snapshot + jsonl tick
  - v2/v3' 留空接口位（None/[]）

S3-S5 会逐步填 v2/v3'。
"""
from __future__ import annotations
import json
import logging
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "outputs"

# 让 import 找到兄弟模块
sys.path.insert(0, str(ROOT / "code"))

from quant.live_elo_solver import solve as v1_solve
from quant.tick_writer import write_all
from quant.live_fetcher_extras import fetch_summary
from quant.live_stats_features import compute_v2_features
from quant import market_anomaly_detector as v3

log = logging.getLogger("quant.live_trading_loop")


# ─────────── 实用工具 ───────────

def _has_live_match(now: Optional[datetime] = None, window_min: int = 150) -> bool:
    """快速判定当前是否需要跑 live tick。

    判断条件（任一为真即返回 True）：
      1. live_events.json 有 live_matches
      2. match_aware 当前在 [kickoff-5min, kickoff+window_min] 窗口（兜底，
         即便 ESPN 还没把比赛标 live 也启动 tick）

    完全无依赖 / 不抛异常。
    """
    # 路径 1：live_events 已有 live_matches
    p = RAW_DIR / "live_events.json"
    if p.exists():
        try:
            d = json.load(open(p))
            if d.get("live_matches"):
                return True
        except Exception:
            pass

    # 路径 2：match_aware 窗口
    try:
        sys.path.insert(0, str(ROOT / "code" / "data"))
        from match_aware import _load_matches_today  # type: ignore
        from datetime import datetime as _dt
        now = now or _dt.now()
        for m in _load_matches_today(now):
            ko = m["kickoff"]
            elapsed_min = (now - ko).total_seconds() / 60.0
            if -5 <= elapsed_min <= window_min:
                return True
    except Exception:
        pass

    return False


def _run_subprocess(cmd: list, timeout: int) -> tuple:
    """运行 fetch 子进程，返回 (rc, elapsed_sec, error_str)。

    不抛异常。timeout 触发时返回 (-1, timeout, 'timeout')。
    """
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, round(time.time() - t0, 2), None
    except subprocess.TimeoutExpired:
        return -1, timeout, "timeout"
    except Exception as e:
        return -2, round(time.time() - t0, 2), f"{type(e).__name__}: {e}"


def _fetch_parallel(timeout: int = 25) -> dict:
    """并行 fetch ESPN scoreboard + Kalshi。

    Polymarket 故意不并发——其 5min 节奏由原 scheduler 任务管。
    """
    fetches = {
        "live_events": ["python3", "code/data/fetch_live_events.py"],
        "kalshi": ["python3", "code/data/fetch_kalshi_match_odds.py"],
    }
    results: dict = {}
    with ThreadPoolExecutor(max_workers=len(fetches)) as ex:
        futures = {ex.submit(_run_subprocess, cmd, timeout): name
                   for name, cmd in fetches.items()}
        for fut in as_completed(futures):
            name = futures[fut]
            rc, dt, err = fut.result()
            results[name] = {"rc": rc, "elapsed_sec": dt, "error": err}
    return results


def _load_baseline_elos() -> dict:
    """加载 teams.json 的 baseline elo 字典。失败返回空 dict。"""
    try:
        teams = json.load(open(RAW_DIR / "teams.json"))
        return {name: t["elo"] for name, t in teams.get("teams", {}).items()
                if isinstance(t, dict) and "elo" in t}
    except Exception as e:
        log.warning(f"_load_baseline_elos 失败: {e}")
        return {}


def _normalize_live_match(m: dict) -> dict:
    """把 ESPN live_matches 里的字段标准化到 live_elo_solver 接受的格式。

    fetch_live_events.py 当前输出字段：
      team_a, team_b, score_a, score_b, status_clock_minute (or elapsed_min),
      events: [{type, minute, team, ...}]

    我们从 events 数组聚合红黄牌 / 换人数（v1 必需）。
    """
    out = {
        "team_a": m.get("team_a"),
        "team_b": m.get("team_b"),
        "score_a": m.get("score_a"),
        "score_b": m.get("score_b"),
        "elapsed_min": m.get("elapsed_min") or m.get("status_clock_minute"),
        "phase": m.get("phase") or m.get("state") or "regular",
    }

    # 从 events 聚合卡牌/换人
    red_a = red_b = yel_a = yel_b = sub_a = sub_b = 0
    ta, tb = out["team_a"], out["team_b"]
    for ev in (m.get("events") or []):
        et = (ev.get("type") or "").lower()
        team = ev.get("team")
        if "red" in et:
            if team == ta: red_a += 1
            elif team == tb: red_b += 1
        elif "yellow" in et:
            if team == ta: yel_a += 1
            elif team == tb: yel_b += 1
        elif "sub" in et:
            if team == ta: sub_a += 1
            elif team == tb: sub_b += 1
    out.update({
        "red_cards_a": red_a, "red_cards_b": red_b,
        "yellow_cards_a": yel_a, "yellow_cards_b": yel_b,
        "subs_used_a": sub_a, "subs_used_b": sub_b,
    })
    return out


# ─────────── 主入口 ───────────

def run_tick(verbose: bool = False) -> dict:
    """执行一次 live tick。

    Returns:
      {
        "changed": bool,
        "summary": str,
        "elapsed_sec": float,
        ... (诊断字段)
      }
    """
    t0 = time.time()

    if not _has_live_match():
        return {"changed": False, "summary": "no live match", "elapsed_sec": round(time.time() - t0, 2)}

    # ── Step 1: 并行 fetch ──
    fetch_results = _fetch_parallel(timeout=25)
    fetch_failed = [k for k, v in fetch_results.items() if v["rc"] != 0]
    if "live_events" in fetch_failed:
        # ESPN fetch 都拿不到 → 这次 tick 没意义
        return {
            "changed": False,
            "summary": f"fetch_live_events failed: {fetch_results['live_events'].get('error')}",
            "elapsed_sec": round(time.time() - t0, 2),
            "fetch_results": fetch_results,
        }

    # ── Step 2: 读 live_events + baseline elos ──
    try:
        live_data = json.load(open(RAW_DIR / "live_events.json"))
    except Exception as e:
        return {"changed": False, "summary": f"读 live_events 失败: {e}",
                "elapsed_sec": round(time.time() - t0, 2)}
    live_matches_raw = live_data.get("live_matches") or []

    if not live_matches_raw:
        # fetch 成功但当前真没 live → 写空 snapshot 让前端知道"已检查过"
        snapshot = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "elapsed_sec": round(time.time() - t0, 2),
            "n_live_matches": 0,
            "live_predictions": [],
            "arbitrage_signals": [],
            "market_anomalies": [],
            "_freshness": {
                "live_events_ts": live_data.get("_metadata", {}).get("fetched_at"),
            },
            "_status": "no_live_matches_after_fetch",
        }
        write_all(snapshot)
        return {"changed": False, "summary": "no live after fetch",
                "elapsed_sec": snapshot["elapsed_sec"]}

    baseline_elos = _load_baseline_elos()

    # ── Step 3: v1 反求 P(H/D/A) per match + v2 质量调整叠加 ──
    live_predictions = []
    v2_failed_count = 0
    for raw in live_matches_raw:
        m = _normalize_live_match(raw)
        # v1
        try:
            r = v1_solve(m, team_elos=baseline_elos)
        except Exception as e:
            log.warning(f"v1_solve 失败 for {m.get('team_a')} vs {m.get('team_b')}: {e}")
            r = {"warnings": [f"v1_solve exception: {e}"], "source": "v1_failed",
                 "delta_elo_a": 0.0, "delta_elo_b": 0.0}

        # v2 — 仅在 v1 成功 + 有 espn_id 时尝试
        v2_block = {"ok": False, "delta_pp_a": 0.0, "components": {},
                    "warnings": ["v2_skipped"]}
        espn_id = raw.get("espn_id") or raw.get("event_id")
        if espn_id and r.get("source") != "v1_failed":
            try:
                summary = fetch_summary(str(espn_id), timeout=6)
                v2_block = compute_v2_features(
                    summary,
                    elapsed_min=int(m.get("elapsed_min") or 0),
                    phase=m.get("phase") or "",
                )
            except Exception as e:
                log.warning(f"v2 异常 for {espn_id}: {e}")
                v2_failed_count += 1
                v2_block = {"ok": False, "delta_pp_a": 0.0, "components": {},
                            "warnings": [f"v2_exception: {type(e).__name__}: {e}"]}
        else:
            v2_block["warnings"] = ["no_espn_id" if not espn_id else "v1_failed_skip_v2"]

        # 把 v2.delta_elo 叠加到 v1 上重算 P(H/D/A)
        v2_delta_elo = float(v2_block.get("delta_elo_a", 0.0))
        if v2_block.get("ok") and abs(v2_delta_elo) > 1e-6 and r.get("source") != "v1_failed":
            try:
                # 用 elo_engine 重算（与 v1 同口径）
                from models.elo_engine import match_probabilities  # type: ignore
                team_a, team_b = m["team_a"], m["team_b"]
                elo_a = baseline_elos.get(team_a, 1500.0) + r.get("delta_elo_a", 0.0) + v2_delta_elo
                elo_b = baseline_elos.get(team_b, 1500.0) + r.get("delta_elo_b", 0.0)
                probs = match_probabilities(elo_a, elo_b)
                r["p_win_a_v1plus_v2"] = float(probs["p_win_a"])
                r["p_draw_v1plus_v2"] = float(probs["p_draw"])
                r["p_win_b_v1plus_v2"] = float(probs["p_win_b"])
            except Exception as e:
                log.warning(f"v1+v2 P 重算失败: {e}")

        live_predictions.append({
            **m, **r,
            "v1_warnings": r.get("warnings", []),
            "v2": v2_block,
            "espn_id": espn_id,
        })

    # ── Step 4: arbitrage_kalshi 重算（min_edge=0 全量）──
    arb_signals = []
    arb_error = None
    try:
        from models.arbitrage_kalshi import generate_kalshi_signals
        arb_signals = generate_kalshi_signals(bankroll=10000,
                                              min_edge_pp=0.0,
                                              future_only=True)
    except Exception as e:
        arb_error = f"{type(e).__name__}: {e}"
        log.warning(f"arbitrage_kalshi 失败: {arb_error}")

    # ── Step 5: v3' Kalshi 异动检测 ──
    market_anomalies: list = []
    v3_error = None
    try:
        kalshi_path = OUT_DIR / "kalshi_match_odds.json"
        if kalshi_path.exists():
            kdata = json.load(open(kalshi_path))
            kmatches = kdata.get("matches") or []
            v3.record_snapshot(kmatches)
            market_anomalies = v3.detect_anomalies()
            # 异动持久化（重启恢复用）
            v3.dump_state()
    except Exception as e:
        v3_error = f"{type(e).__name__}: {e}"
        log.warning(f"v3' 异动检测失败: {v3_error}")

    v2_status = (f"v2_active (failed_count={v2_failed_count}/{len(live_matches_raw)})"
                 if v2_failed_count == 0 else
                 f"v2_partial (failed={v2_failed_count}/{len(live_matches_raw)})")

    # ── Step 6: 组 snapshot + 落盘 ──
    elapsed = round(time.time() - t0, 2)
    snapshot = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "elapsed_sec": elapsed,
        "n_live_matches": len(live_predictions),
        "live_predictions": live_predictions,
        "arbitrage_signals": arb_signals,
        "market_anomalies": market_anomalies,
        "_freshness": {
            "live_events_ts": live_data.get("_metadata", {}).get("fetched_at"),
            "kalshi_ts": _read_ts(OUT_DIR / "kalshi_match_odds.json"),
        },
        "_diagnostics": {
            "fetch_results": fetch_results,
            "v2_status": v2_status,
            "v3_error": v3_error,
            "arb_error": arb_error,
            "n_anomalies": len(market_anomalies),
        },
    }
    write_status = write_all(snapshot)

    n_buy = sum(1 for s in arb_signals if s.get("type") == "BUY")
    n_sell = sum(1 for s in arb_signals if s.get("type") == "SELL")
    n_anom = len(market_anomalies)
    summary = (f"live={len(live_predictions)} buys={n_buy} sells={n_sell} "
               f"anom={n_anom} v2fail={v2_failed_count} "
               f"snap={'ok' if write_status['snapshot_written'] else 'FAIL'} "
               f"tick={'ok' if write_status['tick_appended'] else 'FAIL'}")
    return {
        "changed": True,
        "summary": summary,
        "elapsed_sec": elapsed,
        "n_live_matches": len(live_predictions),
        "n_buy": n_buy,
        "n_sell": n_sell,
        "n_anomalies": n_anom,
        "fetch_results": fetch_results,
    }


def _read_ts(p: Path) -> Optional[str]:
    """读 JSON 文件的 _fetched_at 字段（只为诊断用）。"""
    try:
        d = json.load(open(p))
        return d.get("_fetched_at") or d.get("market", {}).get("fetched_at")
    except Exception:
        return None


# ─────────── CLI 调试入口 ───────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    print("=== quant.live_trading_loop run_tick demo ===")
    r = run_tick(verbose=True)
    print(json.dumps(r, ensure_ascii=False, indent=2))
