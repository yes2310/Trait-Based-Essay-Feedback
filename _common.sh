#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$WORKSPACE_DIR/.venv"
VENV_PYTHON="$VENV_DIR/bin/python"

LEAF_SEMANTIC_TRAITS=(
  grammar_accuracy
  appropriateness_of_word_use
  elasticity_of_sentence_expression
  appropriateness_of_structure_within_a_paragraph
  adequacy_of_inter_paragraph_structure
  consistency_of_structure
  appropriateness_of_portion_size
  clarity_of_topic
  specificity_of_explanation
  creativity_of_thought
)

LEAF_TRAIT_GROUPS="grammar_accuracy,appropriateness_of_word_use,elasticity_of_sentence_expression:3;appropriateness_of_structure_within_a_paragraph,adequacy_of_inter_paragraph_structure,consistency_of_structure,appropriateness_of_portion_size:4;clarity_of_topic,specificity_of_explanation,creativity_of_thought:3"
LEAF_CLASS_BALANCE_MODE="loss_and_sampler"
LEAF_MODEL_VARIANT="canonical_moe"
LEAF_EVOLUTION_STAGE="full"
LEAF_WARMUP_EPOCHS="3"

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
