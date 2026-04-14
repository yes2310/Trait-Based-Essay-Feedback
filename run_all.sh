#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[leaf_workspace] 1/4 uv 환경 준비"
"$WORKSPACE_DIR/setup_uv.sh"

echo "[leaf_workspace] 2/4 trait 사전학습"
"$WORKSPACE_DIR/run_pretrain.sh"

echo "[leaf_workspace] 3/4 holistic 학습"
"$WORKSPACE_DIR/run_holistic.sh"

echo "[leaf_workspace] 4/4 trait-score 학습"
"$WORKSPACE_DIR/run_trait_score.sh"

echo "[leaf_workspace] 전체 실행이 완료되었습니다."
