#!/usr/bin/env python3
"""
WorldCup Predict 每日调度链路（v2）
==================================

8 步流水线：
  1. fetch_elo            — 联网抓 eloratings.net TSV
  2. fetch_injury_news    — RSS 拉伤病新闻 → 更新 injuries.json
  3. fetch_market         — Polymarket / Kalshi 隐含概率（如可用）
  4. mc_simulation        — 100k 蒙特卡洛
  5. swarm                — 14-Agent Swarm（LLM + fallback）
  6. synthesize           — Synthesizer 综合报告
  7. uncertainty          — Layer 5 三层不确定性分解
  8. detect_changes       — 与上轮比较，>3pp 写报警 + 通知

设计原则：
  - 每步独立 try/except，失败不阻塞后续（除非有依赖）
  - 步骤间通过磁盘文件（JSON）传递数据，便于调试
  - 全程写入 logs/cron_<date>.log 与 run_summary.json
  - 支持 CLI 选择性跳过某步（--skip swarm,uncertainty）
  - 支持 dry-run 不实际调用子进程

部署（macOS launchd 或 cron）参考 scripts/install_launchd.sh
"""
from __future__ import annotations
import json
import sys
import os
import time
import argparse
import subprocess
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Callable, List, Optional

sys.path.insert(0, str(Path(__file__).parent))
from utils.io import DATA_OUTPUTS, DATA_RAW, ROOT


LOGS = ROOT / "logs"
LOGS.mkdir(exist_ok=True)
CODE = ROOT / "code"


# ============ 日志 ============
_log_file_path: Optional[Path] = None


def _log_path() -> Path:
    global _log_file_path
    if _log_file_path is None:
        _log_file_path = LOGS / f"cron_{datetime.now().strftime('%Y%m%d')}.log"
    return _log_file_path


def log(msg: str = "") -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}" if msg else ""
    print(line)
    with open(_log_path(), "a") as f:
        f.write(line + "\n")


# ============ 步骤抽象 ============
class Step:
    """一个流水线步骤"""

    def __init__(self, name: str, fn: Callable[[], Dict[str, Any]],
                 timeout: int = 120, required: bool = False,
                 depends: Optional[List[str]] = None):
        self.name = name
        self.fn = fn
        self.timeout = timeout
        self.required = required          # 失败时是否中止整条流水线
        self.depends = depends or []      # 依赖的前置步骤名
        self.result: Dict[str, Any] = {"status": "pending"}

    def run(self, prior_results: Dict[str, Dict[str, Any]],
            dry_run: bool = False) -> None:
        # 依赖检查（dry_run / skipped 都视为通过）
        OK_STATES = {"ok", "skipped", "dry_run"}
        for dep in self.depends:
            dep_status = prior_results.get(dep, {}).get("status")
            if dep_status not in OK_STATES:
                self.result = {
                    "status": "skipped",
                    "reason": f"依赖 {dep} 状态={dep_status}",
                }
                log(f"   ⊘ {self.name} 跳过：依赖 {dep} 失败")
                return

        if dry_run:
            self.result = {"status": "dry_run"}
            log(f"   ⊙ {self.name} (dry-run)")
            return

        t0 = time.time()
        try:
            payload = self.fn()
            self.result = {
                "status": "ok",
                "duration_s": round(time.time() - t0, 2),
                **(payload or {}),
            }
            log(f"   ✓ {self.name}  ({self.result['duration_s']}s)")
        except Exception as e:
            self.result = {
                "status": "failed",
                "duration_s": round(time.time() - t0, 2),
                "error": f"{type(e).__name__}: {e}",
                "trace": traceback.format_exc()[-500:],
            }
            log(f"   ✗ {self.name} 失败: {e}")


# ============ 步骤实现 ============
def _run_subproc(args: List[str], timeout: int) -> Dict[str, Any]:
    """跑子进程并返回基础信息"""
    res = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    return {
        "returncode": res.returncode,
        "stdout_tail": (res.stdout or "")[-400:],
        "stderr_tail": (res.stderr or "")[-200:],
    }


def step_fetch_elo() -> Dict[str, Any]:
    return _run_subproc([sys.executable, str(CODE / "data" / "elo_fetcher.py")],
                        timeout=60)


def step_fetch_injury_news() -> Dict[str, Any]:
    """RSS 抓伤病新闻（已有脚本）"""
    fetcher = CODE / "data" / "injury_news_fetcher.py"
    if not fetcher.exists():
        return {"skipped": True, "note": "injury_news_fetcher 不存在"}
    return _run_subproc([sys.executable, str(fetcher)], timeout=90)


def step_fetch_market() -> Dict[str, Any]:
    """Polymarket / Kalshi 行情（如脚本可独立跑）"""
    # multi_source_market 需要外部喂数据（claude WebFetch），不能独立跑
    # 这里只做"如果存在 latest_market.json 就并入 teams.json"
    latest = DATA_RAW / "latest_market.json"
    if not latest.exists():
        return {"skipped": True, "note": "无 latest_market.json，跳过行情合并"}
    # 调用 multi_source_market 的 write_to_teams_json
    sys.path.insert(0, str(CODE))
    from data.multi_source_market import aggregate_market_odds, write_to_teams_json
    market_data = json.load(open(latest))
    agg = aggregate_market_odds(market_data)
    write_to_teams_json(agg)
    return {"n_teams_updated": len(agg)}


def step_mc_simulation() -> Dict[str, Any]:
    """100k 蒙特卡洛"""
    return _run_subproc(
        [sys.executable, str(CODE / "models" / "finals_analyzer.py"), "100000"],
        timeout=300,
    )


def step_swarm() -> Dict[str, Any]:
    """14-Agent Swarm"""
    sys.path.insert(0, str(CODE))
    from agents.swarm import QueenSwarm
    result = QueenSwarm().run()
    swarm_path = DATA_OUTPUTS / "swarm_consensus.json"
    if swarm_path.exists():
        d = json.load(open(swarm_path))
        top5 = sorted(d.get("swarm_predictions", {}).items(),
                      key=lambda x: -x[1])[:5]
        return {
            "n_agents": d.get("n_agents", 0),
            "top5": [(t, round(p, 2)) for t, p in top5],
        }
    return {"warning": "swarm_consensus.json 未生成"}


def step_synthesize() -> Dict[str, Any]:
    return _run_subproc(
        [sys.executable, str(CODE / "models" / "synthesizer.py")],
        timeout=60,
    )


def step_uncertainty() -> Dict[str, Any]:
    """Layer 5 三层不确定性分解（Top 8，调度场景用快配参数）"""
    sys.path.insert(0, str(CODE))
    from models.uncertainty import decompose_uncertainty
    teams = json.load(open(DATA_OUTPUTS / "synthesizer_report.json"))
    top8 = sorted(teams.items(),
                  key=lambda x: -x[1].get("final_probability", 0))[:8]

    decomp = {}
    for name, _ in top8:
        try:
            # 调度场景：每队 ~3 秒（30 bootstrap × 1500 sim）；8 队约 25s
            decomp[name] = decompose_uncertainty(
                name, n_bootstrap=30, n_sim_per_bootstrap=1500
            )
        except Exception as e:
            decomp[name] = {"error": str(e)}

    out = DATA_OUTPUTS / "uncertainty_decomposition.json"
    out.write_text(json.dumps(decomp, ensure_ascii=False, indent=2))
    return {"n_teams": len(decomp), "output": str(out.relative_to(ROOT))}


def step_detect_changes() -> Dict[str, Any]:
    """对比上一轮，找 ≥3pp 突变"""
    curr_path = DATA_OUTPUTS / "synthesizer_report.json"
    if not curr_path.exists():
        return {"warning": "synthesizer_report.json 不存在"}

    curr = json.load(open(curr_path))
    snapshot = LOGS / "previous_synth.json"
    prev = json.load(open(snapshot)) if snapshot.exists() else {}

    alerts = []
    if prev:
        for team in curr:
            if team not in prev:
                continue
            p_prev = prev[team].get("final_probability", 0)
            p_curr = curr[team].get("final_probability", 0)
            delta = p_curr - p_prev
            if abs(delta) >= 3.0:
                alerts.append({
                    "team": team,
                    "previous": round(p_prev, 2),
                    "current": round(p_curr, 2),
                    "delta_pp": round(delta, 2),
                })

    snapshot.write_text(json.dumps(curr, ensure_ascii=False, indent=2))

    if alerts:
        log(f"   🚨 检测到 {len(alerts)} 个 ≥3pp 突变：")
        for a in alerts:
            arrow = "📈" if a["delta_pp"] > 0 else "📉"
            log(f"     {arrow} {a['team']}: {a['previous']:.1f}% → "
                f"{a['current']:.1f}% ({a['delta_pp']:+.2f}pp)")
        # 写报警文件给外部消费
        (LOGS / "alerts.json").write_text(json.dumps({
            "ts": datetime.now().isoformat(timespec="seconds"),
            "alerts": alerts,
        }, ensure_ascii=False, indent=2))
    elif prev:
        log("   ✓ 概率稳定（无 ≥3pp 突变）")
    else:
        log("   📌 首次运行，建立基线")

    return {"n_alerts": len(alerts), "alerts": alerts}


# ============ 流水线编排 ============
def build_pipeline() -> List[Step]:
    return [
        Step("fetch_elo",           step_fetch_elo,           timeout=60),
        Step("fetch_injury_news",   step_fetch_injury_news,   timeout=90),
        Step("fetch_market",        step_fetch_market,        timeout=30),
        Step("mc_simulation",       step_mc_simulation,       timeout=300,
             required=True),
        Step("swarm",               step_swarm,               timeout=180,
             depends=["mc_simulation"]),
        Step("synthesize",          step_synthesize,          timeout=60,
             depends=["mc_simulation"]),
        Step("uncertainty",         step_uncertainty,         timeout=120,
             depends=["synthesize", "swarm"]),
        Step("detect_changes",      step_detect_changes,      timeout=10,
             depends=["synthesize"]),
    ]


def run_pipeline(skip: Optional[List[str]] = None,
                 only: Optional[List[str]] = None,
                 dry_run: bool = False) -> Dict[str, Any]:
    skip = set(skip or [])
    only = set(only or [])

    log("=" * 60)
    log(f"🚀 WorldCup Predict 调度链 启动 ({'DRY-RUN' if dry_run else 'LIVE'})")
    log("=" * 60)

    pipeline = build_pipeline()
    results: Dict[str, Dict[str, Any]] = {}
    t_start = time.time()

    for i, step in enumerate(pipeline, 1):
        if step.name in skip or (only and step.name not in only):
            step.result = {"status": "skipped", "reason": "用户跳过"}
            results[step.name] = step.result
            log(f"\n[{i}/{len(pipeline)}] {step.name} (skipped)")
            continue

        log(f"\n[{i}/{len(pipeline)}] {step.name}")
        step.run(results, dry_run=dry_run)
        results[step.name] = step.result

        # required 步骤失败 → 中止
        if step.required and step.result.get("status") == "failed":
            log(f"\n❌ 必需步骤 {step.name} 失败，流水线中止")
            break

    duration = round(time.time() - t_start, 2)
    log("\n" + "=" * 60)

    # 总结
    n_ok = sum(1 for r in results.values() if r.get("status") == "ok")
    n_failed = sum(1 for r in results.values() if r.get("status") == "failed")
    n_skipped = sum(1 for r in results.values() if r.get("status") == "skipped")
    log(f"📊 共 {len(pipeline)} 步：成功 {n_ok} | 失败 {n_failed} | 跳过 {n_skipped} | "
        f"耗时 {duration}s")

    summary = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "duration_s": duration,
        "n_ok": n_ok,
        "n_failed": n_failed,
        "n_skipped": n_skipped,
        "steps": results,
    }
    summary_path = DATA_OUTPUTS / "run_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    log(f"📄 摘要写入: {summary_path.relative_to(ROOT)}")

    # 打印 Top 5
    synth = DATA_OUTPUTS / "synthesizer_report.json"
    if synth.exists():
        d = json.load(open(synth))
        top5 = sorted(d.items(), key=lambda x: -x[1].get("final_probability", 0))[:5]
        log("\n🏆 当前 Top 5：")
        for t, info in top5:
            ci_lo = info.get("ci_lower", 0)
            ci_hi = info.get("ci_upper", 0)
            log(f"     {t:<14} {info['final_probability']:>5.1f}%  "
                f"[CI: {ci_lo:.1f}-{ci_hi:.1f}]")

    log("\n✅ 流水线结束\n")
    return summary


# ============ launchd / cron 提示 ============
def install_hint():
    py = sys.executable
    here = Path(__file__).resolve()
    cmd = f"0 9 * * * {py} {here} >> {LOGS}/cron.log 2>&1"
    print()
    print("# ===== 选项 A：crontab（兼容性好）=====")
    print(f"crontab -e")
    print(f"# 然后粘贴：")
    print(f"{cmd}")
    print()
    print("# ===== 选项 B：launchd（macOS 推荐）=====")
    print(f"bash {ROOT}/scripts/install_launchd.sh")
    print()


# ============ CLI ============
def main():
    p = argparse.ArgumentParser(description="WorldCup Predict 每日调度")
    p.add_argument("--skip", default="",
                   help="逗号分隔的跳过步骤，如 --skip fetch_market,uncertainty")
    p.add_argument("--only", default="",
                   help="只跑指定步骤，逗号分隔")
    p.add_argument("--dry-run", action="store_true", help="不实际执行")
    p.add_argument("--install", action="store_true", help="打印部署提示")
    p.add_argument("--list", action="store_true", help="列出所有步骤")
    args = p.parse_args()

    if args.install:
        install_hint()
        return
    if args.list:
        print("步骤列表：")
        for i, s in enumerate(build_pipeline(), 1):
            req = " [REQUIRED]" if s.required else ""
            dep = f" depends={s.depends}" if s.depends else ""
            print(f"  {i}. {s.name}  (timeout={s.timeout}s{req}){dep}")
        return

    skip = [s.strip() for s in args.skip.split(",") if s.strip()]
    only = [s.strip() for s in args.only.split(",") if s.strip()]
    summary = run_pipeline(skip=skip, only=only, dry_run=args.dry_run)
    sys.exit(0 if summary["n_failed"] == 0 else 1)


if __name__ == "__main__":
    main()
