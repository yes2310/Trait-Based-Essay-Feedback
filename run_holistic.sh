#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

run_workspace_python_module promptaes2.cli train-holistic \
  --dataset leaf \
  --trait_checkpoint_dir "$WORKSPACE_DIR/results/trait_pretrain_3class" \
  --csv_path "$WORKSPACE_DIR/data/leaf_merged_3class.csv" \
  --predefined_split_column split \
  --imbalance_mitigation \
  --imbalance_max_weight 5.0 \
  --trait_groups "alignment_with_topic,arguments_supporting_details:2;clarity_of_view_point,spelling_grammar_style:2" \
  --ablation_mode homo_hetero \
  --epochs 20 \
  --batch_size 32 \
  --output_dir "$WORKSPACE_DIR/results/holistic_3class"
