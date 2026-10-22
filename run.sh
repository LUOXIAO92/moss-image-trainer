#!/usr/bin/env bash
# 产品环境引导：没有 .venv 就当场建好，然后启动 Web 控制台。
#   ./run.sh --port 7869
#   ./run.sh --port 7869 --stop     # 停掉监听该端口的实例
#   ./run.sh --setup-only           # 只准备环境，不启动
#   PYTHON=/usr/bin/python3.13 ./run.sh
set -euo pipefail
cd "$(dirname "$0")"

VENV=.venv
SETUP_ONLY=0
STOP=0
PORT=7861
ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --setup-only) SETUP_ONLY=1 ;;
    --stop) STOP=1 ;;
    --port) PORT="$2"; ARGS+=("$1" "$2"); shift ;;
    --port=*) PORT="${1#*=}"; ARGS+=("$1") ;;
    *) ARGS+=("$1") ;;
  esac
  shift
done

have_venv() {
  [ -x "$VENV/bin/python" ] && [ -f "$VENV/pyvenv.cfg" ]
}

listener_pid() {
  ss -ltnp 2>/dev/null | grep ":$1 " | sed -n 's/.*pid=\([0-9]*\).*/\1/p' | head -1
}

PID=$(listener_pid "$PORT" || true)
if [ "$STOP" = "1" ]; then
  if [ -n "${PID:-}" ]; then
    kill "$PID" && echo "已停止端口 $PORT 上的实例（pid $PID）"
  else
    echo "端口 $PORT 上没有运行中的实例"
  fi
  exit 0
fi

if [ -n "${PID:-}" ]; then
  echo "端口 $PORT 已被占用（pid $PID）。" >&2
  echo "  复用现有实例： http://127.0.0.1:$PORT/" >&2
  echo "  先停掉它：     ./run.sh --port $PORT --stop" >&2
  exit 1
fi

if ! have_venv; then
  echo "首次运行：正在创建产品环境 .venv（安装依赖需要几分钟）..."
  if ! ./setup.sh; then
    echo "环境创建失败。可以手动重试：" >&2
    echo "  PYTHON=python3.13 ./setup.sh" >&2
    exit 1
  fi
fi

if ! have_venv; then
  echo "setup.sh 结束但仍然没有可用的 .venv，请检查上面的输出。" >&2
  exit 1
fi

if [ "$SETUP_ONLY" = "1" ]; then
  echo "环境已就绪：$VENV"
  exit 0
fi

echo "启动 WebUI: http://127.0.0.1:$PORT/  (pid $$, 用 ./run.sh --port $PORT --stop 停止)"
exec "$VENV/bin/python" -m moss_lora.cli web "${ARGS[@]}"
