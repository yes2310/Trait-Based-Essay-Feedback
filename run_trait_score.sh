#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

run_workspace_python_module promptaes2.cli train-trait-score \
  --dataset leaf \
  --trait_checkpoint_dir "$WORKSPACE_DIR/results/trait_pretrain" \
  --csv_path "$WORKSPACE_DIR/data/leaf_merged.csv" \
  --predefined_split_column split \
  --trait_groups "$LEAF_TRAIT_GROUPS" \
  --target_traits "${LEAF_SEMANTIC_TRAITS[@]}" \
  --class_balance_mode "$LEAF_CLASS_BALANCE_MODE" \
  --model_variant "$LEAF_MODEL_VARIANT" \
  --evolution_stage "$LEAF_EVOLUTION_STAGE" \
  --warmup_epochs "$LEAF_WARMUP_EPOCHS" \
  --epochs 20 \
  --batch_size 32 \
  --output_dir "$WORKSPACE_DIR/results/trait_score"
