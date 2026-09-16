"""
裁判任命跑批
==============
逻辑：
  1. 从 group_schedule.json 找未来 N 小时内（默认 30h）开赛的场次
  2. 对每场调 referee_radar 抓主裁信息
  3. 写入 data/raw/referee.json，key = "A vs B @ date"

设计：
  - 大部分时段无场次 → 秒退
  - 已抓且未变（confidence != "低"）的跳过，避免反复 LLM 调用
  - 默认窗口 30h：FIFA 通常赛前 24-48h 公布裁判

用法：
  python3 code/data/fetch_referee.py
  python3 code/data/fetch_referee.py --hours 48
  python3 code/data/fetch_referee.py --force
  python3 code/data/fetch_referee.py --match "USA vs Paraguay"
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
from models.referee_radar import analyze_referee

logging.basicConfig(level=logging.WARNING)

ROOT = Path(__file__).parent.parent.parent
OUT_PATH = ROOT / "data" / "raw" / "referee.json"
SCHEDULE_PATH = ROOT / "data" / "raw" / "group_schedule.json"

_FILE_LOCK = threading.Lock()
_PRINT_LOCK = threading.Lock()


def find_upcoming(hours_ahead: int = 30, fetch_all: bool = False):
    sched = json.load(open(SCHEDULE_PATH))
    now = datetime.now()
    cutoff = now + timedelta(hours=hours_ahead)
    out = []
    for m in sched.get("matches", []):
        try:
            dt = datetime.strptime(f"{m['date']} {m.get('time_local','15:00')}",
                                    "%Y-%m-%d %H:%M")
        except Exception:
            continue
        if fetch_all:
            out.append({
                "match_id": m.get("match_id"),
                "team_a": m["team_a"], "team_b": m["team_b"],
                "date": m["date"], "time": m.get("time_local"),
                "match_dt": dt,
            })
        elif now - timedelta(hours=1) <= dt <= cutoff:
            out.append({
                "match_id": m.get("match_id"),
                "team_a": m["team_a"], "team_b": m["team_b"],
                "date": m["date"], "time": m.get("time_local"),
                "match_dt": dt,
            })
    out.sort(key=lambda x: x["match_dt"])
    return out


def find_by_query(query: str):
    sched = json.load(open(SCHEDULE_PATH))
    q = query.lower().replace(" vs ", "|").split("|")
    if len(q) != 2:
        return []
    ta, tb = q[0].strip(), q[1].strip()
    out = []
    for m in sched.get("matches", []):
        a = m["team_a"].lower(); b = m["team_b"].lower()
        if (ta in a and tb in b) or (ta in b and tb in a):
            out.append({"team_a": m["team_a"], "team_b": m["team_b"],
                         "date": m["date"], "time": m.get("time_local"),
                         "match_dt": datetime.now()})
    return out


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
        existing.update(payload)
        existing["_meta"] = {
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "n_matches": sum(1 for k in existing if not k.startswith("_")),
        }
        tmp = OUT_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(existing, ensure_ascii=False, indent=2))
        tmp.replace(OUT_PATH)


def is_recent_enough(entry: dict, max_age_hours: int = 24) -> bool:
    """裁判信息一旦公布就不会变，但有可能从"未公布(低)"变为"已公布(中/高)"，所以仅当 confidence != 低 时跳过"""
    if not entry or entry.get("error"):
        return False
    if entry.get("confidence") == "低":
        return False  # 还没真公布，应重试
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


def run_one(m: dict, force: bool, existing: dict) -> dict:
    key = f"{m['team_a']} vs {m['team_b']} @ {m['date']}"
    if not force and is_recent_enough(existing.get(key)):
        _safe_print(f"  ⏭  {key}：已存在且新鲜，跳过")
        return {"key": key, "skipped": True, "ok": True}
    t0 = time.time()
    try:
        data = analyze_referee(m["team_a"], m["team_b"], m["date"])
        name = data.get("name") or "?"
        ok = not data.get("error")
        _safe_print(f"  {'✓' if ok else '✗'} {key} → {name} ({time.time()-t0:.1f}s)")
        return {"key": key, "data": data, "ok": ok}
    except Exception as e:
        _safe_print(f"  ✗ {key}: {e}")
        return {"key": key, "ok": False, "error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=30)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--match", help="只跑单场 'A vs B'")
    ap.add_argument("--all", action="store_true", help="抓全部 72 场（注意：远期场次裁判未公布）")
    args = ap.parse_args()

    if args.match:
        matches = find_by_query(args.match)
    elif args.all:
        matches = find_upcoming(args.hours, fetch_all=True)
    else:
        matches = find_upcoming(args.hours)
    if not matches:
        print(f"无即将开赛场次（窗口 {args.hours}h），秒退。")
        return

    print(f"=== 裁判任命跑批：{len(matches)} 场（workers={args.workers}）===")
    existing = load_existing()
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(run_one, m, args.force, existing) for m in matches]
        for fut in as_completed(futs):
            try:
                results.append(fut.result())
            except Exception as e:
                _safe_print(f"  ✗ future failed: {e}")

    n_ok = sum(1 for r in results if r.get("ok") and not r.get("skipped"))
    n_skip = sum(1 for r in results if r.get("skipped"))
    n_fail = sum(1 for r in results if not r.get("ok"))

    to_save = {r["key"]: r["data"] for r in results if r.get("data")}
    if to_save:
        save_atomic(to_save)

    print(f"\n成功: {n_ok} / 失败: {n_fail} / 跳过(新鲜): {n_skip}")
    print(f"写入: {OUT_PATH}")


if __name__ == "__main__":
    main()
