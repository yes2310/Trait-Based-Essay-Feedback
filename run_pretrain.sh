#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

run_workspace_python_module promptaes2.pretrain_cli \
  --dataset leaf \
  --data_path "$WORKSPACE_DIR/data/leaf_merged.csv" \
  --predefined_split_column split \
  --traits alignment_with_topic spelling_grammar_style clarity_of_view_point arguments_supporting_details \
  --epochs 5 \
  --batch_size 8 \
  --cpu_workers 0 \
  --output_dir "$WORKSPACE_DIR/results/trait_pretrain"
