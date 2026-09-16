"""
48 队阵容市值跑批
====================
逻辑：
  1. 从 teams.json 取所有 48 强（key = 队名）
  2. 对每队并发调 squad_value_radar
  3. 写入 data/raw/squad_value.json

设计：
  - 默认每周跑 1 次（市值变化慢）
  - 已抓且 < 7 天的跳过
  - 失败不影响其他队（save_atomic 批量写）

用法：
  python3 code/data/fetch_squad_value.py                # 全部 48 队
  python3 code/data/fetch_squad_value.py --workers 4
  python3 code/data/fetch_squad_value.py --force        # 重抓所有
  python3 code/data/fetch_squad_value.py --team Brazil  # 单队
"""
import sys
import json
import time
import logging
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.squad_value_radar import analyze_squad_value

logging.basicConfig(level=logging.WARNING)

ROOT = Path(__file__).parent.parent.parent
OUT_PATH = ROOT / "data" / "raw" / "squad_value.json"
TEAMS_PATH = ROOT / "data" / "raw" / "teams.json"

_FILE_LOCK = threading.Lock()
_PRINT_LOCK = threading.Lock()


def load_teams() -> list:
    d = json.load(open(TEAMS_PATH))
    raw = d.get("teams", {})
    if isinstance(raw, dict):
        return sorted(raw.keys())
    if isinstance(raw, list):
        return sorted([t.get("name") for t in raw if t.get("name")])
    return []


def load_existing() -> dict:
    if OUT_PATH.exists():
        try:
            return json.load(open(OUT_PATH))
        except Exception:
            return {}
    return {}


def save_atomic(payload: dict):
    with _FILE_LOCK:
        existing = load_existing()
        teams_payload = existing.get("teams", {})
        teams_payload.update(payload)
        out = {
            "_meta": {
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "n_teams": len(teams_payload),
            },
            "teams": teams_payload,
        }
        tmp = OUT_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(out, ensure_ascii=False, indent=2))
        tmp.replace(OUT_PATH)


def is_recent_enough(entry: dict, max_age_hours: int = 168) -> bool:
    if not entry or entry.get("error"):
        return False
    ts = entry.get("checked_at")
    if not ts:
        return False
    try:
        dt = datetime.fromisoformat(ts)
    except Exception:
        return False
    return (datetime.now() - dt) <= timedelta(hours=max_age_hours)


def _safe_print(*a, **kw):
    with _PRINT_LOCK:
        print(*a, **kw, flush=True)


def run_one(team: str, force: bool, existing_teams: dict) -> dict:
    if not force and is_recent_enough(existing_teams.get(team)):
        _safe_print(f"  ⏭  {team}：已存在且新鲜，跳过")
        return {"team": team, "skipped": True, "ok": True}
    t0 = time.time()
    try:
        data = analyze_squad_value(team)
        v = data.get("total_value_eur")
        v_str = f"€{v/1e6:.0f}M" if v else "?"
        ok = not data.get("error") and v is not None
        _safe_print(f"  {'✓' if ok else '✗'} {team:<18s} {v_str:>10s} ({time.time()-t0:.1f}s)")
        return {"team": team, "data": data, "ok": ok}
    except Exception as e:
        _safe_print(f"  ✗ {team}: {e}")
        return {"team": team, "ok": False, "error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--team", help="只跑单队")
    ap.add_argument("--only-failed", action="store_true",
                    help="只重跑上次 total_value_eur=null 或 error 的队（隐式 --force）")
    args = ap.parse_args()

    if args.only_failed:
        existing = load_existing().get("teams", {})
        teams = [k for k, v in existing.items()
                 if v.get("total_value_eur") is None or v.get("error")]
        args.force = True  # 必须强制重抓
    elif args.team:
        teams = [args.team]
    else:
        teams = load_teams()
    if not teams:
        print("无队伍，退出。")
        return

    print(f"=== 阵容市值跑批：{len(teams)} 队（workers={args.workers}）===")
    existing = load_existing().get("teams", {})
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(run_one, t, args.force, existing) for t in teams]
        for fut in as_completed(futs):
            try:
                results.append(fut.result())
            except Exception as e:
                _safe_print(f"  ✗ future failed: {e}")

    n_ok = sum(1 for r in results if r.get("ok") and not r.get("skipped"))
    n_skip = sum(1 for r in results if r.get("skipped"))
    n_fail = sum(1 for r in results if not r.get("ok"))

    to_save = {r["team"]: r["data"] for r in results if r.get("data")}
    if to_save:
        save_atomic(to_save)

    print(f"\n成功: {n_ok} / 失败: {n_fail} / 跳过(新鲜): {n_skip}")
    print(f"写入: {OUT_PATH}")


if __name__ == "__main__":
    main()
