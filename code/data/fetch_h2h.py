"""
近 5 场对阵（H2H）跑批
=========================
逻辑：
  1. 从 group_schedule.json 找 "未来 N 小时内" 即将开赛的场次（默认 24h）
  2. 对每场调 h2h_radar 抓双方近 5 次交手
  3. 写入 data/raw/h2h.json，key = "team_a vs team_b @ date"

设计：
  - H2H 数据更新极慢（5 次交手只在双方再次踢球后变化），所以已抓且 confirmed 高的跳过
  - 默认窗口 24h，比 lineups 的 3h 宽：H2H 信息可以提前更早抓
  - 与 lineups 并列：lineups 关心"今天上谁"，h2h 关心"历史交手"

用法：
  python3 code/data/fetch_h2h.py                  # 默认 24h 窗口
  python3 code/data/fetch_h2h.py --hours 48
  python3 code/data/fetch_h2h.py --workers 4
  python3 code/data/fetch_h2h.py --force          # 重抓已有的
  python3 code/data/fetch_h2h.py --match "USA vs Paraguay"
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
from models.h2h_radar import analyze_h2h

logging.basicConfig(level=logging.WARNING)

ROOT = Path(__file__).parent.parent.parent
H2H_PATH = ROOT / "data" / "raw" / "h2h.json"
SCHEDULE_PATH = ROOT / "data" / "raw" / "group_schedule.json"

_FILE_LOCK = threading.Lock()
_PRINT_LOCK = threading.Lock()


def find_upcoming_matches(hours_ahead: int = 24, fetch_all: bool = False):
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


def find_match_by_query(query: str):
    sched = json.load(open(SCHEDULE_PATH))
    q = query.lower().replace(" vs ", "|").split("|")
    if len(q) != 2:
        return []
    ta, tb = q[0].strip(), q[1].strip()
    out = []
    for m in sched.get("matches", []):
        a = m["team_a"].lower()
        b = m["team_b"].lower()
        if (ta in a and tb in b) or (ta in b and tb in a):
            try:
                dt = datetime.strptime(f"{m['date']} {m.get('time_local','15:00')}",
                                        "%Y-%m-%d %H:%M")
            except Exception:
                dt = datetime.now()
            out.append({
                "match_id": m.get("match_id"),
                "team_a": m["team_a"], "team_b": m["team_b"],
                "date": m["date"], "time": m.get("time_local"),
                "match_dt": dt,
            })
    return out


def load_existing() -> dict:
    if H2H_PATH.exists():
        try:
            return json.load(open(H2H_PATH))
        except Exception:
            return {}
    return {}


def save_atomic(payload: dict):
    """带 lock 的原子写"""
    with _FILE_LOCK:
        existing = load_existing()
        existing.update(payload)
        existing["_meta"] = {
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "n_matches": sum(1 for k in existing if not k.startswith("_")),
        }
        tmp = H2H_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(existing, ensure_ascii=False, indent=2))
        tmp.replace(H2H_PATH)


def is_recent_enough(entry: dict, max_age_hours: int = 168) -> bool:
    """已抓且 7 天内的就不重抓（H2H 变化慢）"""
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


def _safe_print(*args, **kwargs):
    with _PRINT_LOCK:
        print(*args, **kwargs, flush=True)


def run_one(match: dict, force: bool, existing: dict) -> dict:
    key = f"{match['team_a']} vs {match['team_b']} @ {match['date']}"
    if not force and is_recent_enough(existing.get(key)):
        _safe_print(f"  ⏭  {key}：已存在且新鲜，跳过")
        return {"key": key, "skipped": True, "ok": True}
    t0 = time.time()
    try:
        data = analyze_h2h(match["team_a"], match["team_b"])
        n = len(data.get("matches", []))
        ok = not data.get("error")
        _safe_print(f"  {'✓' if ok else '✗'} {key} ({time.time()-t0:.1f}s, {n} 场)")
        return {"key": key, "data": data, "ok": ok}
    except Exception as e:
        _safe_print(f"  ✗ {key}: {e}")
        return {"key": key, "ok": False, "error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=24)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--match", help="只跑单场 'A vs B'")
    ap.add_argument("--all", action="store_true", help="抓全部 72 场（无视窗口）")
    args = ap.parse_args()

    if args.match:
        matches = find_match_by_query(args.match)
    elif args.all:
        matches = find_upcoming_matches(args.hours, fetch_all=True)
    else:
        matches = find_upcoming_matches(args.hours)

    if not matches:
        print(f"无即将开赛场次（窗口 {args.hours}h），秒退。")
        return

    print(f"=== H2H 跑批：{len(matches)} 场（窗口 {args.hours}h，workers={args.workers}）===")
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

    # 批量写
    to_save = {r["key"]: r["data"] for r in results if r.get("data")}
    if to_save:
        save_atomic(to_save)

    print(f"\n成功: {n_ok} / 失败: {n_fail} / 跳过(已新鲜): {n_skip}")
    print(f"写入: {H2H_PATH}")


if __name__ == "__main__":
    main()
