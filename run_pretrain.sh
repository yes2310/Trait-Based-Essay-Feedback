#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

run_workspace_python_module promptaes2.pretrain_cli \
  --dataset leaf \
  --data_path "$WORKSPACE_DIR/data/leaf_merged.csv" \
  --predefined_split_column split \
  --class_balance_mode "$LEAF_CLASS_BALANCE_MODE" \
  --print_epoch_metrics \
  --traits "${LEAF_SEMANTIC_TRAITS[@]}" \
  --epochs 5 \
  --batch_size 8 \
  --cpu_workers 0 \
  --output_dir "$WORKSPACE_DIR/results/trait_pretrain"
