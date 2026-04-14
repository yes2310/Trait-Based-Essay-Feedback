from __future__ import annotations

import argparse

from promptaes2.config import (
    ConfigError,
    DATASET_PRESETS,
    DataAlignmentError,
    MissingColumnError,
    parse_dropout_rates,
    parse_hidden_sizes,
    parse_trait_groups,
)
from promptaes2.training.holistic import run_holistic_training
from promptaes2.training.trait_score import run_trait_score_training


def _add_holistic_parser(subparsers):
    parser = subparsers.add_parser("train-holistic", help="Train holistic scorer from precomputed trait embeddings")
    parser.add_argument("--dataset", type=str, choices=sorted(DATASET_PRESETS.keys()), required=True)

    parser.add_argument("--npz_path", type=str, default=None)
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--trait_groups", type=str, required=True)
    parser.add_argument("--split_by_column", type=str, default=None)
    parser.add_argument("--predefined_split_column", type=str, default=None)
    parser.add_argument("--trait_checkpoint_dir", type=str, default=None)
    parser.add_argument("--trait_model_name", type=str, default=None)
    parser.add_argument("--embedding_batch_size", type=int, default=32)
    parser.add_argument("--embedding_max_seq_length", type=int, default=512)
    parser.add_argument("--backbone_mode", type=str, choices=["frozen", "e2e"], default="frozen")
    parser.add_argument("--backbone_lr", type=float, default=1e-5)
    parser.add_argument("--unfreeze_epoch", type=int, default=0)

    parser.add_argument("--embedding_dim", type=int, default=768)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=6e-5)
    parser.add_argument("--hidden_sizes", type=str, default="512-512")
    parser.add_argument("--dropout", type=str, default="0.1-0.1")

    parser.add_argument("--loss_type", type=str, choices=["combined", "cross_entropy"], default="combined")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta1", type=float, default=2.0)
    parser.add_argument("--beta2", type=float, default=1.0)
    parser.add_argument("--beta3", type=float, default=1.0)
    parser.add_argument("--beta4", type=float, default=1.0)
    parser.add_argument("--lambda1", type=float, default=5.0)
    parser.add_argument("--lambda2", type=float, default=10.0)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--combined_stage1_epochs", type=int, default=1)
    parser.add_argument(
        "--cl_score_binning",
        type=str,
        choices=["auto", "none", "force6"],
        default="auto",
    )
    parser.add_argument("--moe_aux_weight", type=float, default=0.01)
    parser.add_argument("--model_variant", type=str, choices=["legacy", "canonical_moe"], default="legacy")
    parser.add_argument("--group_num_experts", type=int, default=8)
    parser.add_argument("--classifier_num_experts", type=int, default=8)
    parser.add_argument("--router_top_k", type=int, default=2)

    parser.add_argument("--scheduler", type=str, choices=["cosine", "none"], default="cosine")
    parser.add_argument("--scheduler_t0", type=int, default=10)
    parser.add_argument("--scheduler_tmult", type=int, default=2)
    parser.add_argument("--scheduler_eta_min", type=float, default=1e-6)
    parser.add_argument("--imbalance_mitigation", action="store_true", default=False)
    parser.add_argument("--imbalance_max_weight", type=float, default=5.0)

    parser.add_argument("--evolution_stage", type=str, choices=["baseline", "cross_attention", "full"], default="full")
    parser.add_argument("--warmup_epochs", type=int, default=20)
    parser.add_argument(
        "--ablation_mode",
        type=str,
        choices=["none", "homo_only", "hetero_only", "homo_hetero"],
        default="none",
    )

    parser.add_argument("--use_skip1", action="store_true", default=False)
    parser.add_argument("--use_skip2", action="store_true", default=False)
    parser.add_argument("--use_skip3", action="store_true", default=False)
    parser.add_argument("--use_pre_homo_skip", action="store_true", default=False)
    parser.add_argument("--use_pre_hetero_skip", action="store_true", default=False)

    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--cpu_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="results/holistic")
    parser.add_argument(
        "--save_checkpoints",
        dest="save_checkpoints",
        action="store_true",
        default=True,
        help="Save best checkpoints/models during training (default: enabled).",
    )
    parser.add_argument(
        "--no_save_checkpoints",
        dest="save_checkpoints",
        action="store_false",
        help="Disable checkpoint/model file saving.",
    )

    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project_name", type=str, default="creativity-score_2")
    parser.add_argument("--run_name", type=str, default=None)

    parser.set_defaults(func=run_holistic_training)
    return parser


def _add_trait_score_parser(subparsers):
    parser = subparsers.add_parser(
        "train-trait-score",
        help="Train CreativityScorer to predict trait scores (classification) from trait embeddings",
    )
    parser.add_argument("--dataset", type=str, choices=sorted(DATASET_PRESETS.keys()), required=True)
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--trait_groups", type=str, required=True)
    parser.add_argument("--target_traits", nargs="+", required=True)

    parser.add_argument("--npz_path", type=str, default=None)
    parser.add_argument("--predefined_split_column", type=str, default=None)
    parser.add_argument("--trait_checkpoint_dir", type=str, default=None)
    parser.add_argument("--trait_model_name", type=str, default=None)
    parser.add_argument("--embedding_batch_size", type=int, default=32)
    parser.add_argument("--embedding_max_seq_length", type=int, default=512)

    parser.add_argument("--embedding_dim", type=int, default=768)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=6e-5)
    parser.add_argument("--hidden_sizes", type=str, default="512-512")
    parser.add_argument("--dropout", type=str, default="0.1-0.1")
    parser.add_argument("--moe_aux_weight", type=float, default=0.01)
    parser.add_argument("--model_variant", type=str, choices=["legacy", "canonical_moe"], default="legacy")
    parser.add_argument("--group_num_experts", type=int, default=8)
    parser.add_argument("--classifier_num_experts", type=int, default=8)
    parser.add_argument("--router_top_k", type=int, default=2)

    parser.add_argument("--scheduler", type=str, choices=["cosine", "none"], default="cosine")
    parser.add_argument("--scheduler_t0", type=int, default=10)
    parser.add_argument("--scheduler_tmult", type=int, default=2)
    parser.add_argument("--scheduler_eta_min", type=float, default=1e-6)
    parser.add_argument("--imbalance_mitigation", action="store_true", default=False)
    parser.add_argument("--imbalance_max_weight", type=float, default=5.0)

    parser.add_argument("--evolution_stage", type=str, choices=["baseline", "cross_attention", "full"], default="full")
    parser.add_argument("--warmup_epochs", type=int, default=20)
    parser.add_argument(
        "--ablation_mode",
        type=str,
        choices=["none", "homo_only", "hetero_only", "homo_hetero"],
        default="none",
    )

    parser.add_argument("--use_skip1", action="store_true", default=False)
    parser.add_argument("--use_skip2", action="store_true", default=False)
    parser.add_argument("--use_skip3", action="store_true", default=False)
    parser.add_argument("--use_pre_homo_skip", action="store_true", default=False)
    parser.add_argument("--use_pre_hetero_skip", action="store_true", default=False)

    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--cpu_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="results/trait_score")
    parser.add_argument(
        "--save_checkpoints",
        dest="save_checkpoints",
        action="store_true",
        default=True,
        help="Save best checkpoints/models during training (default: enabled).",
    )
    parser.add_argument(
        "--no_save_checkpoints",
        dest="save_checkpoints",
        action="store_false",
        help="Disable checkpoint/model file saving.",
    )

    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project_name", type=str, default="promptaes2-trait-score")
    parser.add_argument("--run_name", type=str, default=None)

    parser.set_defaults(func=run_trait_score_training)
    return parser


def _apply_ablation_mode(args) -> None:
    mode = getattr(args, "ablation_mode", "none")
    if mode == "none":
        return
    if mode == "homo_only":
        args.evolution_stage = "baseline"
        args.warmup_epochs = 0
        return
    if mode == "hetero_only":
        args.evolution_stage = "cross_attention"
        args.warmup_epochs = 0
        return
    if mode == "homo_hetero":
        args.evolution_stage = "full"
        args.warmup_epochs = 0
        return


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PromptAES2 CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_holistic_parser(subparsers)
    _add_trait_score_parser(subparsers)

    return parser


def _validate_parsed_args(args, parser):
    if args.command in {"train-holistic", "train-trait-score"}:
        _apply_ablation_mode(args)

        split_by_column = getattr(args, "split_by_column", None)
        if split_by_column is not None:
            split_by_column = split_by_column.strip()
            if not split_by_column:
                parser.error("--split_by_column must not be empty")
            args.split_by_column = split_by_column
        predefined_split_column = getattr(args, "predefined_split_column", None)
        if predefined_split_column is not None:
            predefined_split_column = predefined_split_column.strip()
            if not predefined_split_column:
                parser.error("--predefined_split_column must not be empty")
            args.predefined_split_column = predefined_split_column
        if (
            split_by_column is not None
            and predefined_split_column is not None
            and split_by_column == predefined_split_column
        ):
            parser.error("--split_by_column and --predefined_split_column must refer to different columns")
        try:
            args.hidden_sizes = parse_hidden_sizes(args.hidden_sizes)
            args.dropout_rates = parse_dropout_rates(args.dropout)
            args.trait_groups = parse_trait_groups(args.trait_groups)
        except ConfigError as exc:
            parser.error(str(exc))

        if predefined_split_column is None:
            split_ratios = {
                "train": args.train_ratio,
                "val": args.val_ratio,
                "test": args.test_ratio,
            }
            for split_name, split_ratio in split_ratios.items():
                if split_ratio <= 0 or split_ratio >= 1:
                    parser.error(f"--{split_name}_ratio must be > 0 and < 1")
            ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
            if abs(ratio_sum - 1.0) > 1e-6:
                parser.error("--train_ratio + --val_ratio + --test_ratio must equal 1.0")
        if args.epochs < 1:
            parser.error("--epochs must be >= 1")
        if args.batch_size < 1:
            parser.error("--batch_size must be >= 1")
        if args.embedding_batch_size < 1:
            parser.error("--embedding_batch_size must be >= 1")
        if args.embedding_max_seq_length < 1:
            parser.error("--embedding_max_seq_length must be >= 1")
        if args.warmup_epochs < 0:
            parser.error("--warmup_epochs must be >= 0")
        if args.moe_aux_weight < 0:
            parser.error("--moe_aux_weight must be >= 0")
        if args.group_num_experts < 1:
            parser.error("--group_num_experts must be >= 1")
        if args.classifier_num_experts < 1:
            parser.error("--classifier_num_experts must be >= 1")
        if args.router_top_k < 1:
            parser.error("--router_top_k must be >= 1")
        if args.imbalance_max_weight <= 0:
            parser.error("--imbalance_max_weight must be > 0")
        if args.command == "train-holistic":
            if args.backbone_lr <= 0:
                parser.error("--backbone_lr must be > 0")
            if args.unfreeze_epoch < 0:
                parser.error("--unfreeze_epoch must be >= 0")
            if args.backbone_mode == "e2e":
                if args.trait_checkpoint_dir is None:
                    parser.error("--trait_checkpoint_dir is required when --backbone_mode e2e")
                if args.npz_path is not None:
                    parser.error("--npz_path cannot be used with --backbone_mode e2e")
        if args.command == "train-holistic" and args.loss_type == "combined":
            if args.lambda1 < 0:
                parser.error("--lambda1 must be >= 0")
            if args.lambda2 < 0:
                parser.error("--lambda2 must be >= 0")
            if args.combined_stage1_epochs < 0:
                parser.error("--combined_stage1_epochs must be >= 0")
        if args.npz_path is None and args.trait_checkpoint_dir is None:
            parser.error("Provide either --npz_path or --trait_checkpoint_dir")
        if args.command == "train-trait-score":
            if not args.target_traits:
                parser.error("--target_traits must contain at least one trait")
        if args.patience < 1:
            parser.error("--patience must be >= 1")


def main(argv: list[str] | None = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_parsed_args(args, parser)

    try:
        return args.func(args)
    except (ConfigError, MissingColumnError, DataAlignmentError, RuntimeError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
