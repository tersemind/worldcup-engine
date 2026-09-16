"""
Scheduler 守护进程启动器
========================

设计目标：
  - 启动 scheduler.py 进程，并彻底脱离当前 shell 会话/进程组
  - 关掉 codebuddy/zsh/iTerm 也不会被 SIGHUP 杀掉
  - PID/日志写到固定路径，方便后续 stop / status
  - 防重复启动：检测到已有 daemon 在跑就拒绝

用法：
    python3 code/data/scheduler_daemon.py start [--bootstrap=default|full|none]
    python3 code/data/scheduler_daemon.py stop
    python3 code/data/scheduler_daemon.py status
    python3 code/data/scheduler_daemon.py restart [--bootstrap=...]

原理（macOS/Linux 通用）：
  - 用 subprocess.Popen(start_new_session=True) → 子进程 setsid() 自成 session leader
  - 子进程不再属于父 shell 的进程组，不再收 SIGHUP
  - stdin 重定向到 /dev/null，stdout/stderr 重定向到日志文件
  - 子进程完全独立后，父启动器立即退出
"""
import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent.parent
PID_FILE = ROOT / "data" / "scheduler.pid"
LOG_FILE = ROOT / "data" / "scheduler.daemon.log"
SCHEDULER_SCRIPT = ROOT / "code" / "data" / "scheduler.py"


def _is_running(pid: int) -> bool:
    """检查 PID 是否还活着（os.kill(pid, 0) 不发信号只探活）"""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _read_pid() -> Optional[int]:
    if not PID_FILE.exists():
        return None
    try:
        pid = int(PID_FILE.read_text().strip())
        return pid if _is_running(pid) else None
    except (ValueError, OSError):
        return None


def cmd_status():
    pid = _read_pid()
    if pid:
        print(f"✓ scheduler daemon 运行中 (PID={pid})")
        print(f"  日志: {LOG_FILE}")
        print(f"  尾部日志（tail -20）:")
        try:
            with open(LOG_FILE) as f:
                lines = f.readlines()
                for line in lines[-20:]:
                    print(f"    {line.rstrip()}")
        except Exception as e:
            print(f"    （读日志失败：{e}）")
        return 0
    else:
        print(f"✗ scheduler daemon 未运行")
        if PID_FILE.exists():
            print(f"  （PID 文件残留：{PID_FILE.read_text().strip()}，已失效）")
        return 1


def cmd_stop():
    pid = _read_pid()
    if not pid:
        print("scheduler daemon 未运行，无需停止")
        if PID_FILE.exists():
            PID_FILE.unlink()
        return 0
    print(f"正在停止 scheduler daemon (PID={pid})...")
    try:
        os.kill(pid, signal.SIGTERM)
        # 等最多 10s
        for _ in range(20):
            if not _is_running(pid):
                break
            time.sleep(0.5)
        if _is_running(pid):
            print(f"  SIGTERM 无效，发送 SIGKILL")
            os.kill(pid, signal.SIGKILL)
            time.sleep(1)
        print(f"✓ 已停止")
        if PID_FILE.exists():
            PID_FILE.unlink()
        return 0
    except Exception as e:
        print(f"✗ 停止失败: {e}")
        return 1


def cmd_start(bootstrap: str = "default", tick: int = 60):
    """启动 daemon 化的 scheduler 进程，立即返回不阻塞"""
    pid = _read_pid()
    if pid:
        print(f"✗ scheduler daemon 已在运行 (PID={pid})，请先 stop 或 restart")
        return 1

    # 清掉旧 PID 文件
    if PID_FILE.exists():
        PID_FILE.unlink()

    print(f"正在启动 scheduler daemon...")
    print(f"  bootstrap: {bootstrap}")
    print(f"  tick: {tick}s")
    print(f"  log: {LOG_FILE}")

    # 打开日志（追加模式，daemon 会一直写）
    log_fp = open(LOG_FILE, "ab", buffering=0)

    cmd = [
        sys.executable, "-u", str(SCHEDULER_SCRIPT),
        "--tick", str(tick),
        "--bootstrap", bootstrap,
    ]

    # 关键：start_new_session=True → Popen 内部会调 os.setsid()
    # 子进程成为新 session leader + 新 process group leader
    # 父 shell 退出/Ctrl+C 都不会通过 SIGHUP 影响子进程
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=log_fp,
        stderr=subprocess.STDOUT,
        cwd=str(ROOT),
        start_new_session=True,
        close_fds=True,
    )

    # 写 PID
    PID_FILE.write_text(str(proc.pid))

    # 等 2s 看是否立即崩溃
    time.sleep(2)
    if not _is_running(proc.pid):
        print(f"✗ 启动后立即退出，查看日志：tail {LOG_FILE}")
        if PID_FILE.exists():
            PID_FILE.unlink()
        return 1

    print(f"✓ scheduler daemon 已后台启动 (PID={proc.pid})")
    print(f"  现在可以安全关闭 codebuddy / 终端，daemon 不会被杀")
    print(f"  查看日志：tail -f {LOG_FILE}")
    print(f"  查看状态：python3 code/data/scheduler_daemon.py status")
    print(f"  停止：python3 code/data/scheduler_daemon.py stop")
    return 0


def cmd_restart(bootstrap: str = "default", tick: int = 60):
    cmd_stop()
    time.sleep(1)
    return cmd_start(bootstrap=bootstrap, tick=tick)


def main():
    ap = argparse.ArgumentParser(description="Scheduler 守护进程启动器（脱离 shell 会话）")
    ap.add_argument("action", choices=["start", "stop", "status", "restart"])
    ap.add_argument("--bootstrap", choices=["none", "default", "full"], default="default",
                    help="冷启动全量刷新策略（仅 start/restart 有效）")
    ap.add_argument("--tick", type=int, default=60, help="scheduler 主循环 tick 秒数")
    args = ap.parse_args()

    if args.action == "status":
        sys.exit(cmd_status())
    elif args.action == "stop":
        sys.exit(cmd_stop())
    elif args.action == "start":
        sys.exit(cmd_start(bootstrap=args.bootstrap, tick=args.tick))
    elif args.action == "restart":
        sys.exit(cmd_restart(bootstrap=args.bootstrap, tick=args.tick))


if __name__ == "__main__":
    main()
