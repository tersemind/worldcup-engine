#!/bin/bash
# WorldCup Scheduler - 开机自启 + 断点续跑 安装/管理脚本
# =========================================================
# 用法:
#   ./scheduler_autostart.sh install   # 装到 LaunchAgents 并立即加载
#   ./scheduler_autostart.sh uninstall # 卸载（关闭自启）
#   ./scheduler_autostart.sh status    # 查看 launchd 状态 + scheduler 进程
#   ./scheduler_autostart.sh restart   # 重启（重新加载 plist）
#   ./scheduler_autostart.sh log       # 实时查看日志

set -e

PLIST_SRC="/Users/kego/.codebuddy/worldcup-predict/code/data/com.kego.worldcup-scheduler.plist"
PLIST_DST="$HOME/Library/LaunchAgents/com.kego.worldcup-scheduler.plist"
LABEL="com.kego.worldcup-scheduler"
LOG_FILE="/Users/kego/.codebuddy/worldcup-predict/data/scheduler.daemon.log"
RESILIENT_LOG="/Users/kego/.codebuddy/worldcup-predict/data/scheduler.resilient.log"
PID_FILE="/Users/kego/.codebuddy/worldcup-predict/data/scheduler.pid"

cmd_install() {
  echo "📦 安装 LaunchAgent..."

  # 如果当前有 daemon 在跑（非 launchd 启动的），先停掉避免冲突
  if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE" 2>/dev/null || echo "")
    if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
      echo "  检测到现有 daemon (PID=$OLD_PID)，先停止以免和 launchd 冲突"
      python3 /Users/kego/.codebuddy/worldcup-predict/code/data/scheduler_daemon.py stop || true
      sleep 2
    fi
  fi

  # 卸载已有的（若存在）
  if [ -f "$PLIST_DST" ]; then
    echo "  发现已有 plist，先卸载"
    launchctl unload "$PLIST_DST" 2>/dev/null || true
  fi

  cp "$PLIST_SRC" "$PLIST_DST"
  echo "  ✓ plist 复制到 $PLIST_DST"

  launchctl load -w "$PLIST_DST"
  echo "  ✓ launchctl load -w 完成（已启用开机自启）"

  sleep 2
  cmd_status
}

cmd_uninstall() {
  echo "🗑  卸载 LaunchAgent..."
  if [ -f "$PLIST_DST" ]; then
    launchctl unload -w "$PLIST_DST" 2>/dev/null || true
    rm -f "$PLIST_DST"
    echo "  ✓ 已卸载并删除 plist"
  else
    echo "  （plist 不存在，无需卸载）"
  fi
  # 残留 daemon 进程
  if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE" 2>/dev/null || echo "")
    if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
      echo "  仍有 scheduler 进程 (PID=$OLD_PID)，发送 SIGTERM"
      kill -TERM "$OLD_PID" 2>/dev/null || true
    fi
  fi
}

cmd_status() {
  echo "📊 launchd 状态："
  if launchctl list | grep -q "$LABEL"; then
    launchctl list | grep "$LABEL" | awk '{printf "  PID=%s  ExitCode=%s  Label=%s\n", $1, $2, $3}'
  else
    echo "  ✗ launchd 未加载（执行 install 启用开机自启）"
  fi

  echo ""
  echo "📊 scheduler 进程："
  if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE" 2>/dev/null || echo "")
    if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
      echo "  ✓ 运行中 (PID=$PID)"
      ps -p "$PID" -o pid,etime,sess,pgid,ppid,stat,command 2>/dev/null | tail -1
    else
      echo "  ✗ PID 文件 $PID 已失效"
    fi
  else
    echo "  （无 PID 文件）"
  fi

  echo ""
  echo "📜 最近日志（tail -15）："
  if [ -f "$LOG_FILE" ]; then
    tail -15 "$LOG_FILE" | sed 's/^/    /'
  else
    echo "    （日志文件还没生成）"
  fi
}

cmd_restart() {
  echo "🔄 重启 LaunchAgent..."
  if [ -f "$PLIST_DST" ]; then
    launchctl unload "$PLIST_DST" 2>/dev/null || true
    sleep 2
    launchctl load -w "$PLIST_DST"
    sleep 2
    cmd_status
  else
    echo "  ✗ plist 不存在，先执行 install"
    return 1
  fi
}

cmd_log() {
  echo "📜 实时日志（Ctrl+C 退出）— $LOG_FILE"
  tail -f "$LOG_FILE"
}

cmd_help() {
  cat <<EOF
WorldCup Scheduler 开机自启 + 断点续跑

用法:
  $0 install     安装 LaunchAgent，启用开机自启
  $0 uninstall   卸载 LaunchAgent，关闭开机自启
  $0 status      查看 launchd + scheduler 进程状态
  $0 restart     重新加载 plist（修改 plist 后用）
  $0 log         实时跟踪日志

断点续跑工作流：
  1. install 后，scheduler 立即启动，并写 LaunchAgent
  2. 每次登录/开机，launchd 自动拉起 scheduler_resilient_start.py
  3. 弹性启动器先等网（最长 5min），再 exec scheduler.py
  4. scheduler 启动时 bootstrap=default：按依赖序刷新所有板块
  5. 进程崩溃 → launchd 自动重启（ThrottleInterval=30s）
EOF
}

case "${1:-help}" in
  install)   cmd_install ;;
  uninstall) cmd_uninstall ;;
  status)    cmd_status ;;
  restart)   cmd_restart ;;
  log)       cmd_log ;;
  help|--help|-h) cmd_help ;;
  *) echo "未知命令: $1"; cmd_help; exit 1 ;;
esac
