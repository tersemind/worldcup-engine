"""
赛前首发阵容跑批（并发）
=========================
逻辑：
  1. 从 group_schedule.json 找 "未来 N 小时内" 即将开赛的场次（默认 3h）
  2. 对每场的双队并发调 lineup_radar（Serper + LLM）
  3. 写入 data/raw/lineups.json

设计理由：
  - 教练公布首发时间通常是赛前 1-2h（FIFA 规则要求赛前 75 分钟）
  - 我们提前 3h 抓，赶上 "预测 XI" 报道；准确度会随窗口缩小提升
  - 已确认（confirmed=true）的场次直接跳过（避免重复消耗 quota）

用法：
  python3 code/data/fetch_lineups.py                  # 默认 3h 窗口
  python3 code/data/fetch_lineups.py --hours 6        # 调窗口
  python3 code/data/fetch_lineups.py --workers 8      # 并发数
  python3 code/data/fetch_lineups.py --force          # 忽略 confirmed 已抓的，重新跑
  python3 code/data/fetch_lineups.py --match "USA vs Paraguay"   # 只跑单场
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
from models.lineup_radar import analyze_team_lineup

logging.basicConfig(level=logging.WARNING)

ROOT = Path(__file__).parent.parent.parent
LINEUPS_PATH = ROOT / "data" / "raw" / "lineups.json"
SCHEDULE_PATH = ROOT / "data" / "raw" / "group_schedule.json"

_FILE_LOCK = threading.Lock()
_PRINT_LOCK = threading.Lock()


def find_upcoming_matches(hours_ahead: int = 3):
    """找未来 N 小时内即将开赛的场次"""
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
        if now - timedelta(minutes=30) <= dt <= cutoff:
            out.append({
                "match_id": m.get("match_id"),
                "team_a":   m["team_a"],
                "team_b":   m["team_b"],
                "date":     m["date"],
                "time":     m.get("time_local"),
                "match_dt": dt,
            })
    out.sort(key=lambda x: x["match_dt"])
    return out


def find_match_by_query(query: str):
    """支持 --match 'USA vs Paraguay' 这样的查询"""
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
                continue
            out.append({
                "match_id": m.get("match_id"),
                "team_a":   m["team_a"],
                "team_b":   m["team_b"],
                "date":     m["date"],
                "time":     m.get("time_local"),
                "match_dt": dt,
            })
    return out


def _load_lineups():
    if LINEUPS_PATH.exists():
        try:
            return json.load(open(LINEUPS_PATH))
        except Exception:
            pass
    return {"_metadata": {}, "lineups": {}}


def _save_lineups(data):
    data["_metadata"] = {
        "last_updated": datetime.now().isoformat(timespec="seconds"),
        "source": "fetch_lineups.py (Serper + LLM)",
        "n_matches": len(data.get("lineups", {})),
    }
    with _FILE_LOCK:
        LINEUPS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def _match_key(m):
    return f"{m['team_a']} vs {m['team_b']} @ {m['date']}"


def _process_team(team: str, opponent: str, date_str: str, idx: int, total: int):
    # 1-3s 随机错峰避免并发时 Lingya 集中触发空响应
    import random
    time.sleep(random.uniform(0.5, 3.0))
    t0 = time.time()
    try:
        res = analyze_team_lineup(team, opponent, date_str)
    except Exception as e:
        res = {"team": team, "error": f"exception: {e}",
               "checked_at": datetime.now().isoformat(timespec="seconds")}
    elapsed = time.time() - t0
    with _PRINT_LOCK:
        if "error" in res:
            print(f"[{idx}/{total}] ✗ {team:22s} vs {opponent:18s} {res['error'][:60]} ({elapsed:.1f}s)", flush=True)
        else:
            n = len(res.get("starters", []))
            conf = "✓" if res.get("confirmed") else "?"
            print(f"[{idx}/{total}] ✓ {team:22s} vs {opponent:18s} {n}人 {res.get('formation','-')} {conf}confirm ({elapsed:.1f}s)", flush=True)
    return res


def main(hours: int = 3, workers: int = 8, force: bool = False,
         match_query: str = None):
    if match_query:
        matches = find_match_by_query(match_query)
        scope = f"指定场次 ({match_query})"
    else:
        matches = find_upcoming_matches(hours)
        scope = f"未来 {hours}h 窗口"
    
    print(f"=== 首发阵容跑批 ===")
    print(f"范围: {scope}")
    print(f"候选场次: {len(matches)}")
    
    if not matches:
        print("无场次，写入空 stub")
        _save_lineups(_load_lineups())
        return
    
    data = _load_lineups()
    if "lineups" not in data:
        data["lineups"] = {}
    
    # 跳过已 confirmed 的（除非 --force）
    todo = []
    for m in matches:
        k = _match_key(m)
        existing = data["lineups"].get(k, {})
        a_ok = existing.get("team_a", {}).get("confirmed") and not existing.get("team_a", {}).get("error")
        b_ok = existing.get("team_b", {}).get("confirmed") and not existing.get("team_b", {}).get("error")
        if not force and a_ok and b_ok:
            print(f"  跳过（已 confirmed）: {k}")
            continue
        todo.append(m)
    
    if not todo:
        print("所有场次都已 confirmed，无需重抓")
        return
    
    # 构造 (team, opp, date, label) 队列；每场两队各一个 work item
    work = []
    for m in todo:
        work.append((m["team_a"], m["team_b"], m["date"], _match_key(m), "team_a"))
        work.append((m["team_b"], m["team_a"], m["date"], _match_key(m), "team_b"))
    
    print(f"待跑: {len(todo)} 场 × 2 = {len(work)} 个查询，并发 {workers}\n")
    
    t0 = time.time()
    n_ok, n_fail = 0, 0
    
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for i, (team, opp, date_str, key, side) in enumerate(work, 1):
            fut = pool.submit(_process_team, team, opp, date_str, i, len(work))
            futures[fut] = (key, side)
        
        for fut in as_completed(futures):
            key, side = futures[fut]
            try:
                res = fut.result()
            except Exception as e:
                with _PRINT_LOCK:
                    print(f"[!] {key}/{side} 异常: {e}")
                n_fail += 1
                continue
            if "error" in res:
                n_fail += 1
            else:
                n_ok += 1
            # 写入对应 side
            with _FILE_LOCK:
                if key not in data["lineups"]:
                    data["lineups"][key] = {}
                data["lineups"][key][side] = res
    
    _save_lineups(data)
    elapsed = time.time() - t0
    print(f"\n=== 完成 ===")
    print(f"总耗时: {elapsed:.1f}s")
    print(f"成功: {n_ok} / 失败: {n_fail}")
    print(f"写入: {LINEUPS_PATH}")
    
    # 打印关键场次摘要
    if not match_query:
        print(f"\n=== 已抓场次摘要 ===")
        for m in todo:
            k = _match_key(m)
            info = data["lineups"].get(k, {})
            a = info.get("team_a", {})
            b = info.get("team_b", {})
            a_n = len(a.get("starters", []))
            b_n = len(b.get("starters", []))
            a_f = a.get("formation", "-") or "-"
            b_f = b.get("formation", "-") or "-"
            print(f"  {k}:  {a_n}人({a_f}) vs {b_n}人({b_f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=3,
                     help="赛前窗口（小时），默认 3h（赶上预测 XI 报道）")
    ap.add_argument("--workers", type=int, default=8,
                     help="并发线程数（默认 8）")
    ap.add_argument("--force", action="store_true",
                     help="忽略已 confirmed 的场次，强制重抓")
    ap.add_argument("--match", type=str, default=None,
                     help="只跑单场: 'USA vs Paraguay'")
    args = ap.parse_args()
    main(hours=args.hours, workers=args.workers,
         force=args.force, match_query=args.match)
