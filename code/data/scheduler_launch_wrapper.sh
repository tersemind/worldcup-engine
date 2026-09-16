#!/bin/bash
# launchd 启动 wrapper：在调用 scheduler_resilient_start.py 之前注入 LLM 凭据
# 这样 ~/.secrets/llm_keys.env 是唯一的 key 来源，plist 不需要存明文
set -e

# 加载用户级密钥（如不存在不报错，让下面的健康检查统一处理）
if [ -f "$HOME/.secrets/llm_keys.env" ]; then
  . "$HOME/.secrets/llm_keys.env"
fi

# ── LLM key 健康检查 ──
# 缺 key 时直接退出+落盘日志，避免 daemon 静默启动后所有 LLMAgent 都走规则降级
LOG_DIR="/Users/kego/.codebuddy/worldcup-engine/code/data/logs"
mkdir -p "$LOG_DIR"
HEALTH_LOG="$LOG_DIR/scheduler_launch_health.log"
TS="$(date '+%Y-%m-%d %H:%M:%S')"

if [ -z "${LINGYA_API_KEY:-}" ]; then
  echo "[$TS] FATAL: LINGYA_API_KEY 未设置，请检查 ~/.secrets/llm_keys.env (mode 600)" >> "$HEALTH_LOG"
  echo "[$TS] FATAL: LINGYA_API_KEY 未设置" >&2
  exit 78  # EX_CONFIG: launchd 会按 KeepAlive/SuccessfulExit 重试策略处理
fi

if [ -z "${LINGYA_BASE_URL:-}" ]; then
  echo "[$TS] WARN: LINGYA_BASE_URL 未设置，client.py 会用默认值" >> "$HEALTH_LOG"
fi

echo "[$TS] OK: LLM key 健康检查通过 (model=${WORLDCUP_LLM_MODEL:-default}, fallback=${WORLDCUP_LLM_FALLBACK_MODEL:-default})" >> "$HEALTH_LOG"

# LLM Quorum：8 个 LLMAgent 中允许最多 25% (即 2 个) fallback
# 主模型已经有备用模型自动降级（client.py 内部），单 agent 失败已很罕见
# 如需更严：=0.0；如需放宽：=0.5；完全禁用：=1.0
: "${WORLDCUP_LLM_REQUIRE:=0.25}"
export WORLDCUP_LLM_REQUIRE

exec /usr/bin/python3 -u \
  /Users/kego/.codebuddy/worldcup-engine/code/data/scheduler_resilient_start.py \
  "$@"
