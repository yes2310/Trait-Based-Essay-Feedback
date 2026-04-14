#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$WORKSPACE_DIR/.venv"
VENV_PYTHON="$VENV_DIR/bin/python"

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    return 0
  fi

  echo "uv가 설치되어 있지 않습니다." >&2
  echo "예: python3 -m pip install uv" >&2
  exit 1
}

ensure_venv() {
  ensure_uv

  if [ ! -x "$VENV_PYTHON" ]; then
    echo "[leaf_workspace] uv 가상환경이 없어 새로 생성합니다: $VENV_DIR"
    uv venv "$VENV_DIR"
  fi
}

sync_requirements() {
  ensure_venv
  echo "[leaf_workspace] 의존성을 설치합니다."
  uv pip install --python "$VENV_PYTHON" -r "$WORKSPACE_DIR/requirements.txt"
}

run_workspace_python_module() {
  ensure_venv
  export PYTHONPATH="$WORKSPACE_DIR/runtime${PYTHONPATH:+:$PYTHONPATH}"
  cd "$WORKSPACE_DIR"
  "$VENV_PYTHON" -m "$@"
}
