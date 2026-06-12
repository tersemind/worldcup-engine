"""
伤病雷达跑批
=============
对"近 24h 内有比赛"的球队批量调 injury_radar，把结果写入 injuries.json。

完成后可选触发下游重算：synthesizer → monte_carlo → match_bias_detector

用法：
  python3 code/models/run_injury_radar.py                  # 跑批
  python3 code/models/run_injury_radar.py --cascade        # 跑批 + 自动重算下游
  python3 code/models/run_injury_radar.py --hours 48       # 改时间窗口（默认 24h）
  python3 code/models/run_injury_radar.py --max 3          # 限制场数（测试用）
"""
import sys
import json
import time
import logging
import argparse
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.injury_radar import analyze_team_injuries
from utils.io import save_output

logging.basicConfig(level=logging.WARNING)

# 并发写 injuries.json 的互斥锁
_INJURIES_LOCK = threading.Lock()
# 控制台打印的互斥锁（避免多线程交错）
_PRINT_LOCK = threading.Lock()

ROOT = Path(__file__).parent.parent.parent
INJURIES_PATH = ROOT / "data" / "raw" / "injuries.json"
SCHEDULE_PATH = ROOT / "data" / "raw" / "group_schedule.json"


def find_upcoming_teams(hours_ahead: int = 24):
    """
    扫赛程，找未来 N 小时内有比赛的球队
    返回 [{team, opponent, date, time_local}, ...]
    """
    sched = json.load(open(SCHEDULE_PATH))
    now = datetime.now()
    cutoff = now + timedelta(hours=hours_ahead)
    
    teams_to_check = []
    seen = set()
    
    for m in sched.get("matches", []):
        # 比赛时间（用 date + time_local，时区按 EDT/纽约时区粗略处理）
        try:
            dt_str = f"{m['date']} {m.get('time_local','15:00')}"
            match_dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M")
        except Exception:
            continue
        
        if match_dt <= now - timedelta(hours=2):
            continue  # 已过去
        if match_dt > cutoff:
            continue  # 超出窗口
        
        for team, opponent in [(m["team_a"], m["team_b"]), (m["team_b"], m["team_a"])]:
            key = (team, m["date"])
            if key in seen:
                continue
            seen.add(key)
            teams_to_check.append({
                "team": team,
                "opponent": opponent,
                "date": m["date"],
                "time_local": m.get("time_local"),
                "match_dt": match_dt.isoformat(timespec="minutes"),
            })
    
    # 按时间排序
    teams_to_check.sort(key=lambda x: x["match_dt"])
    return teams_to_check


def find_all_teams():
    """
    扫赛程，全 72 场涉及的所有球队（去重）
    每队取其"最近的下一场比赛"作为搜索上下文
    返回 [{team, opponent, date}, ...]
    """
    sched = json.load(open(SCHEDULE_PATH))
    now = datetime.now()
    
    # 收集每队的所有未来比赛
    team_matches = {}  # team -> list of (match_dt, opponent, date)
    
    for m in sched.get("matches", []):
        try:
            dt_str = f"{m['date']} {m.get('time_local','15:00')}"
            match_dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M")
        except Exception:
            continue
        
        for team, opponent in [(m["team_a"], m["team_b"]), (m["team_b"], m["team_a"])]:
            team_matches.setdefault(team, []).append((match_dt, opponent, m["date"]))
    
    # 每队取最近一场（未来的，若全部已过则取最后一场）
    teams_to_check = []
    for team, matches in team_matches.items():
        matches.sort(key=lambda x: x[0])  # 按时间升序
        # 优先取未来场
        future = [m for m in matches if m[0] > now]
        chosen = future[0] if future else matches[-1]
        match_dt, opponent, date = chosen
        teams_to_check.append({
            "team": team,
            "opponent": opponent,
            "date": date,
            "match_dt": match_dt.isoformat(timespec="minutes"),
        })
    
    # 按 team 名字排序
    teams_to_check.sort(key=lambda x: x["team"])
    return teams_to_check


def merge_into_injuries_json(team: str, result: dict):
    """把单队结果合并到 injuries.json（线程安全：互斥锁包住 读-改-写）"""
    with _INJURIES_LOCK:
        # 加载已有
        if INJURIES_PATH.exists():
            data = json.load(open(INJURIES_PATH))
        else:
            data = {"_metadata": {}, "injuries": {}}
        
        if "injuries" not in data:
            data["injuries"] = {}
        
        # 写入该队
        if "error" in result:
            # 只记录失败信息（不覆盖已有数据）
            existing = data["injuries"].get(team, {})
            existing["_last_check_error"] = result["error"]
            existing["_last_check_at"] = result.get("checked_at", "")
            data["injuries"][team] = existing
        else:
            # 完整写入（覆盖之前的）
            data["injuries"][team] = {
                "absences": result.get("absences", []),
                "total_pp_impact": result.get("total_pp_impact", 0),
                "confidence": result.get("confidence", "中"),
                "notes": result.get("notes", ""),
                "checked_at": result.get("checked_at", ""),
                "added_by": "llm_radar",
            }
        
        # 更新元数据
        data["_metadata"] = {
            "last_updated": datetime.now().isoformat(timespec="seconds"),
            "source": "injury_radar.py (Serper + LLM, parallel)",
            "format_version": "v2",
        }
        
        INJURIES_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def _process_one_team(t: dict, idx: int, total: int) -> dict:
    """单队处理（在线程池中跑）：搜索 + LLM + 写盘"""
    # 0.5-3s 随机错峰，避免并发时 Lingya 集中触发空响应（参考 fetch_lineups.py）
    import random
    time.sleep(random.uniform(0.5, 3.0))
    match_t = time.time()
    try:
        result = analyze_team_injuries(t["team"], t["opponent"], t["date"])
    except Exception as e:
        result = {"error": f"exception: {e}",
                  "checked_at": datetime.now().isoformat(timespec="seconds")}
    elapsed = time.time() - match_t
    merge_into_injuries_json(t["team"], result)
    
    # 打印进度（加锁防止多线程交错）
    with _PRINT_LOCK:
        if "error" in result:
            print(f"[{idx}/{total}] ✗ {t['team']:22s} vs {t['opponent']:18s} "
                  f"{result['error'][:60]} ({elapsed:.1f}s)", flush=True)
        else:
            n_abs = len(result.get("absences", []))
            pp = result.get("total_pp_impact", 0)
            if n_abs > 0:
                names = ", ".join(a["player"] for a in result["absences"][:3])
                print(f"[{idx}/{total}] ✓ {t['team']:22s} vs {t['opponent']:18s} "
                      f"{n_abs}伤 {pp:+.1f}pp [{names}] ({elapsed:.1f}s)", flush=True)
            else:
                print(f"[{idx}/{total}] ✓ {t['team']:22s} vs {t['opponent']:18s} "
                      f"全员健康 ({elapsed:.1f}s)", flush=True)
    
    return {"team": t["team"], "opponent": t["opponent"], "date": t["date"],
            "result": result, "elapsed": elapsed}


def cascade_downstream():
    """触发下游重算: synthesizer → mc 100k → match_bias"""
    steps = [
        ("synthesizer", ["python3", "code/models/synthesizer.py"]),
        ("monte_carlo (100k)", ["python3", "code/models/monte_carlo.py", "100000"]),
        ("match_bias_detector", ["python3", "code/models/match_bias_detector.py"]),
    ]
    print("\n=== 触发下游级联重算 ===")
    for name, cmd in steps:
        t0 = time.time()
        print(f"  跑 {name}...", end="", flush=True)
        try:
            result = subprocess.run(cmd, cwd=str(ROOT),
                                       capture_output=True, text=True, timeout=300)
            elapsed = time.time() - t0
            if result.returncode == 0:
                print(f" ✓ ({elapsed:.1f}s)")
            else:
                print(f" ✗ ({elapsed:.1f}s)")
                print(f"    stderr: {result.stderr[:200]}")
        except subprocess.TimeoutExpired:
            print(" ✗ 超时（>5min）")
        except Exception as e:
            print(f" ✗ {e}")


def main(hours: int = 24, cascade: bool = False, max_n: int = None,
         cascade_only: bool = False, all_teams: bool = False,
         workers: int = 8, retry_failed: bool = True):
    # --cascade-only: 跳过 LLM 跑批，仅触发下游重算（用于已有 injuries.json 时）
    if cascade_only:
        print("=== 仅触发下游级联重算（不跑 LLM 雷达）===\n")
        cascade_downstream()
        return
    
    # --all: 全 48 队（去重，每队取最近一场）；否则 --hours 窗口
    if all_teams:
        teams = find_all_teams()
        scope_desc = f"全部 {len(teams)} 队（去重，每队 1 次查询）"
    else:
        teams = find_upcoming_teams(hours)
        scope_desc = f"未来 {hours} 小时窗口（{len(teams)} 队）"
    
    if max_n:
        teams = teams[:max_n]
    
    print(f"=== 伤病雷达跑批（并发）===", flush=True)
    print(f"范围: {scope_desc}", flush=True)
    print(f"待检查球队: {len(teams)}", flush=True)
    print(f"并发 workers: {workers}", flush=True)
    print("", flush=True)
    
    if not teams:
        print("无球队，跳过")
        if cascade:
            cascade_downstream()
        return
    
    t0 = time.time()
    found_injuries = []
    failed_teams = []
    n_updated = 0
    n_failed = 0
    total = len(teams)
    
    # ─── 第一轮：并发跑全部 ───
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process_one_team, t, i, total): t
                   for i, t in enumerate(teams, 1)}
        for fut in as_completed(futures):
            try:
                rec = fut.result()
            except Exception as e:
                t = futures[fut]
                with _PRINT_LOCK:
                    print(f"[!] {t['team']} 线程异常: {e}", flush=True)
                failed_teams.append(t)
                n_failed += 1
                continue
            
            res = rec["result"]
            if "error" in res:
                n_failed += 1
                failed_teams.append({"team": rec["team"], "opponent": rec["opponent"],
                                      "date": rec["date"]})
            else:
                n_updated += 1
                for a in res.get("absences", []):
                    found_injuries.append({"team": rec["team"],
                                            "match_date": rec["date"], **a})
    
    # ─── 第二轮：失败队自动重试一次（串行，避免再次触发限流）───
    if retry_failed and failed_teams:
        print(f"\n=== 自动重试失败的 {len(failed_teams)} 队（串行）===", flush=True)
        retry_round = list(failed_teams)
        failed_teams = []
        for i, t in enumerate(retry_round, 1):
            rec = _process_one_team(t, i, len(retry_round))
            res = rec["result"]
            if "error" in res:
                failed_teams.append(t)
            else:
                # 修正计数：把上轮的失败转成成功
                n_failed -= 1
                n_updated += 1
                for a in res.get("absences", []):
                    found_injuries.append({"team": rec["team"],
                                            "match_date": rec["date"], **a})
    
    total_elapsed = time.time() - t0
    print(f"\n=== 完成 ===", flush=True)
    print(f"总耗时: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
    print(f"更新: {n_updated} / 失败: {n_failed}")
    print(f"发现伤病: {len(found_injuries)} 条")
    
    if failed_teams:
        print("\n=== 失败列表（可单独重跑）===")
        for t in failed_teams:
            print(f"  python3 code/models/injury_radar.py --team \"{t['team']}\" "
                  f"--opponent \"{t['opponent']}\" --date {t['date']}")
    
    if found_injuries:
        print("\n=== 关键伤病 ===")
        found_injuries.sort(key=lambda x: (x.get("importance") != "high",
                                              x.get("pp_impact", 0)))
        for inj in found_injuries:
            print(f"  🩹 [{inj['team']:14s}] {inj['player']:20s} "
                  f"{inj['position']:20s} {inj['status']:10s} {inj.get('pp_impact', 0):+.1f}pp")
    
    if cascade and n_updated > 0:
        cascade_downstream()
        print("\n下游已重算，前端刷新即可看到新数据。")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=24, help="比赛时间窗口（小时）")
    ap.add_argument("--all", dest="all_teams", action="store_true",
                     help="全 48 队（去重）；优先级 > --hours")
    ap.add_argument("--cascade", action="store_true", help="跑完后触发下游重算")
    ap.add_argument("--cascade-only", action="store_true",
                     help="仅触发下游重算，不跑 LLM 雷达")
    ap.add_argument("--max", type=int, default=None, help="最多 N 场（调试）")
    ap.add_argument("--workers", type=int, default=8,
                     help="并发线程数（默认 8；保守 4-6，激进 10-12）")
    ap.add_argument("--no-retry", dest="retry_failed", action="store_false",
                     help="失败队不自动重试")
    args = ap.parse_args()
    main(hours=args.hours, cascade=args.cascade, max_n=args.max,
         cascade_only=args.cascade_only, all_teams=args.all_teams,
         workers=args.workers, retry_failed=args.retry_failed)
