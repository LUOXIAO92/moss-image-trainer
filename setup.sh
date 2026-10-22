#!/usr/bin/env bash
# 创建（或修复）产品环境 .venv 并安装依赖。
#   ./setup.sh                      # 自动挑一个 Python 3.11+
#   PYTHON=/usr/bin/python3.13 ./setup.sh
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130 ./setup.sh   # 指定 CUDA 版 torch
#   ./setup.sh --force              # 删掉重建
#   SETUP_DRY_RUN=1 ./setup.sh      # 只解析依赖，不真正下载安装
set -euo pipefail
cd "$(dirname "$0")"

FORCE=0
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
  esac
done

pick_python() {
  if [ -n "${PYTHON:-}" ]; then printf '%s' "$PYTHON"; return; fi
  for candidate in python3.14 python3.13 python3.12 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then printf '%s' "$candidate"; return; fi
  done
  return 1
}

PY=$(pick_python) || {
  echo "找不到 Python 3.11+。请安装后重试，或用 PYTHON=/path/to/python 指定。" >&2
  exit 1
}

if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'; then
  echo "$PY 版本过低（需要 3.11+）。" >&2
  exit 1
fi
echo "Python: $("$PY" -V 2>&1)"

if [ "$FORCE" = "1" ]; then
  rm -rf .venv
fi

if [ ! -x .venv/bin/python ] || [ ! -f .venv/pyvenv.cfg ]; then
  echo "创建 .venv ..."
  "$PY" -m venv .venv
fi

PIP=(.venv/bin/python -m pip)

if [ "${SETUP_DRY_RUN:-0}" = "1" ]; then
  echo "（dry-run：以下命令不会执行）"
  echo "  .venv/bin/python -m pip install --upgrade pip"
  echo "  .venv/bin/python -m pip install -r requirements.txt"
  echo "环境已就绪（dry-run）：$PWD/.venv"
  exit 0
fi

.venv/bin/python -m pip install --upgrade pip

if [ -n "${TORCH_INDEX_URL:-}" ]; then
  echo "torch 从 $TORCH_INDEX_URL 安装"
  "${PIP[@]}" install torch torchvision --index-url "$TORCH_INDEX_URL"
  "${PIP[@]}" install -r requirements.txt
else
  "${PIP[@]}" install -r requirements.txt
  echo "注意：torch 用的是 requirements 里的默认轮子；要用 CUDA 版请下次用 TORCH_INDEX_URL 重装。" >&2
fi

if [ "${SETUP_DRY_RUN:-0}" != "1" ]; then
  .venv/bin/python - <<'PY'
import torch
print(f"torch {torch.__version__} · CUDA {'可用' if torch.cuda.is_available() else '不可用'}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
PY

  # FlashAttention-2：按当前 CUDA / torch / Python / 算力匹配预编译轮子并装进 .venv。
  # 装不上不阻断 setup（训练会退回没有 varlen 的 SDPA 路径），但会明确告警。
  if [ "${SKIP_FLASH_ATTN:-0}" != "1" ]; then
    echo
    echo "FlashAttention-2"
    if ! .venv/bin/python setup_fa2.py install; then
      echo "警告：FlashAttention-2 未安装，训练将退回 SDPA（没有 varlen，打包训练会变慢）。" >&2
    fi
  fi
fi

echo "环境已就绪：./run.sh --port 7869"
