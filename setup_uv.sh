#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

sync_requirements

echo "[leaf_workspace] uv 환경 준비가 완료되었습니다."
echo "가상환경 Python: $VENV_PYTHON"
