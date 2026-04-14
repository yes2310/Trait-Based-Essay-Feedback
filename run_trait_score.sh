#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

run_workspace_python_module promptaes2.cli train-trait-score \
  --dataset leaf \
  --trait_checkpoint_dir "$WORKSPACE_DIR/results/trait_pretrain" \
  --csv_path "$WORKSPACE_DIR/data/leaf_merged.csv" \
  --predefined_split_column split \
  --trait_groups "alignment_with_topic,arguments_supporting_details:2;clarity_of_view_point,spelling_grammar_style:2" \
  --target_traits alignment_with_topic arguments_supporting_details \
  --ablation_mode homo_hetero \
  --epochs 20 \
  --batch_size 32 \
  --output_dir "$WORKSPACE_DIR/results/trait_score"
