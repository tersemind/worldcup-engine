"""
Scheduler 弹性启动器（断点续跑专用）
======================================

设计目标：
  - 适用于 launchd KeepAlive 场景：开机自启、崩溃自动重启
  - 启动时先**等待网络**（最多 5 分钟），联网后再启 scheduler
  - 若已有 daemon 在跑（PID 文件 + 探活），直接 exit 0（让 launchd 不重复拉起）
  - 直接前台运行 scheduler.py（不脱离会话）—— launchd 本身就是守护，不需要再 setsid
  - bootstrap=default：每次冷启动都做一次依赖序刷新，断点续跑首次刷新会把停机期间漏掉的数据补齐

用法（由 launchd 调用）：
    python3 code/data/scheduler_resilient_start.py [--bootstrap=default] [--tick=60]

手动调试：
    python3 code/data/scheduler_resilient_start.py --bootstrap=default --foreground
"""
import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
PID_FILE = ROOT / "data" / "scheduler.pid"
SCHEDULER_SCRIPT = ROOT / "code" / "data" / "scheduler.py"
RESILIENT_LOG = ROOT / "data" / "scheduler.resilient.log"


def _log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} [resilient] {msg}"
    print(line, flush=True)
    try:
        with open(RESILIENT_LOG, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _existing_daemon_pid():
    if not PID_FILE.exists():
        return None
    try:
        pid = int(PID_FILE.read_text().strip())
        return pid if _is_running(pid) else None
    except Exception:
        return None


def _wait_for_network(max_wait_s: int = 300, probe_interval_s: int = 5) -> bool:
    """
    等待联网。尝试解析多个公共域名，任一成功即认为联网。
    超时返回 False，但仍允许 scheduler 启动（很多任务有缓存可以离线工作）。
    """
    probes = [
        ("gamma-api.polymarket.com", 443),
        ("api.elevenlabs.io", 443),  # 备用 SSL endpoint
        ("8.8.8.8", 53),  # DNS 兜底
        ("1.1.1.1", 53),
    ]
    start = time.time()
    attempt = 0
    while time.time() - start < max_wait_s:
        attempt += 1
        for host, port in probes:
            try:
                sock = socket.create_connection((host, port), timeout=3)
                sock.close()
                _log(f"✓ 联网就绪 (probe={host}:{port}, 第{attempt}次, 耗时{int(time.time()-start)}s)")
                return True
            except Exception:
                continue
        _log(f"  等待联网... 第{attempt}次失败，{probe_interval_s}s 后重试")
        time.sleep(probe_interval_s)
    _log(f"⚠ 等待网络超时（{max_wait_s}s），仍尝试启动 scheduler（可能离线运行）")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", choices=["none", "default", "full"], default="default")
    ap.add_argument("--tick", type=int, default=60)
    ap.add_argument("--foreground", action="store_true",
                    help="前台运行（手动调试用，否则由 launchd 守护）")
    ap.add_argument("--max-wait-net", type=int, default=300,
                    help="最长等待联网秒数")
    args = ap.parse_args()

    _log("=" * 60)
    _log(f"🌅 弹性启动器触发（bootstrap={args.bootstrap}, tick={args.tick}）")

    # 1) 检查是否已有 daemon 在跑（防止 launchd 重复拉起）
    existing = _existing_daemon_pid()
    if existing:
        _log(f"✓ 已有 scheduler 在运行 (PID={existing})，本次弹性启动器退出")
        return 0

    # PID 文件残留清理
    if PID_FILE.exists():
        _log(f"  清理失效 PID 文件: {PID_FILE.read_text().strip()}")
        try:
            PID_FILE.unlink()
        except Exception:
            pass

    # 2) 等待联网
    _wait_for_network(max_wait_s=args.max_wait_net)

    # 3) 启动 scheduler.py（前台 exec，由 launchd 守护）
    cmd = [
        sys.executable, "-u", str(SCHEDULER_SCRIPT),
        "--tick", str(args.tick),
        "--bootstrap", args.bootstrap,
    ]
    _log(f"🚀 exec scheduler: {' '.join(cmd)}")

    # 写 PID（exec 前先记自己 PID，因为 execv 会保留 PID）
    PID_FILE.write_text(str(os.getpid()))

    # 使用 execv 替换进程，scheduler 直接接管，launchd 看到的 PID 不变
    try:
        os.execv(sys.executable, cmd)
    except Exception as e:
        _log(f"✗ exec 失败: {e}")
        # 清理 PID
        try:
            PID_FILE.unlink()
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    sys.exit(main())
