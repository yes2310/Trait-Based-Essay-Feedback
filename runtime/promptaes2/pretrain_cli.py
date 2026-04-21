from __future__ import annotations

import argparse

from promptaes2.config import (
    ConfigError,
    DATASET_PRESETS,
    DataAlignmentError,
    MissingColumnError,
)
from promptaes2.training.trait_pretrain import run_trait_pretrain


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PromptAES2 pretrain CLI")
    parser.add_argument("--dataset", type=str, choices=sorted(DATASET_PRESETS.keys()), required=True)
    parser.add_argument("--traits", nargs="+", default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--split_by_column", type=str, default=None)
    parser.add_argument("--predefined_split_column", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument(
        "--class_balance_mode",
        type=str,
        choices=["none", "loss", "loss_and_sampler"],
        default="none",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--auto_stop", action="store_true", default=False)
    parser.add_argument("--cpu_workers", type=int, default=4)
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument("--scheduler", type=str, choices=["cosine", "none"], default="none")
    parser.add_argument("--scheduler_t0", type=int, default=10)
    parser.add_argument("--scheduler_tmult", type=int, default=2)
    parser.add_argument("--scheduler_eta_min", type=float, default=1e-6)
    parser.add_argument("--print_epoch_metrics", action="store_true", default=False)
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project_name", type=str, default="promptaes2-pretrain")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="results/pretrain")
    return parser


def _validate_parsed_args(args, parser: argparse.ArgumentParser) -> None:
    if args.split_by_column is not None:
        args.split_by_column = args.split_by_column.strip()
        if not args.split_by_column:
            parser.error("--split_by_column must not be empty")
    if args.predefined_split_column is not None:
        args.predefined_split_column = args.predefined_split_column.strip()
        if not args.predefined_split_column:
            parser.error("--predefined_split_column must not be empty")
    if (
        args.split_by_column is not None
        and args.predefined_split_column is not None
        and args.split_by_column == args.predefined_split_column
    ):
        parser.error("--split_by_column and --predefined_split_column must refer to different columns")

    if args.epochs is None:
        args.epochs = 200 if args.auto_stop else 20

    if args.epochs < 1:
        parser.error("--epochs must be >= 1")
    if args.batch_size < 1:
        parser.error("--batch_size must be >= 1")
    if args.max_seq_length < 1:
        parser.error("--max_seq_length must be >= 1")
    if args.early_stopping_patience < 1:
        parser.error("--early_stopping_patience must be >= 1")
    if args.scheduler_t0 < 1:
        parser.error("--scheduler_t0 must be >= 1")
    if args.scheduler_tmult < 1:
        parser.error("--scheduler_tmult must be >= 1")


def main(argv: list[str] | None = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_parsed_args(args, parser)

    try:
        return run_trait_pretrain(args)
    except (ConfigError, MissingColumnError, DataAlignmentError, RuntimeError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
