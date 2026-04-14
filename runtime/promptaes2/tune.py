from __future__ import annotations

import argparse
import json
import shlex
from datetime import datetime
from pathlib import Path

from promptaes2.cli import _validate_parsed_args, build_parser as build_train_parser
from promptaes2.config import DATASET_PRESETS, ConfigError, DataAlignmentError, MissingColumnError


def _parse_csv_values(raw: str, cast, field_name: str):
    values: list = []
    for item in raw.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        try:
            values.append(cast(stripped))
        except ValueError as exc:
            raise ValueError(f"Invalid value '{stripped}' in {field_name}.") from exc
    if not values:
        raise ValueError(f"{field_name} must contain at least one value.")
    return values


def _parse_csv_str_choices(raw: str, field_name: str) -> list[str]:
    return _parse_csv_values(raw, str, field_name)


def _parse_csv_int_choices(raw: str, field_name: str) -> list[int]:
    return _parse_csv_values(raw, int, field_name)


def _parse_csv_float_choices(raw: str, field_name: str) -> list[float]:
    return _parse_csv_values(raw, float, field_name)


def _parse_csv_bool_choices(raw: str, field_name: str) -> list[bool]:
    mapping = {"true": True, "false": False, "1": True, "0": False}
    values: list[bool] = []
    for item in raw.split(","):
        stripped = item.strip().lower()
        if not stripped:
            continue
        if stripped not in mapping:
            raise ValueError(
                f"Invalid boolean value '{item}' in {field_name}. Use true/false."
            )
        values.append(mapping[stripped])
    if not values:
        raise ValueError(f"{field_name} must contain at least one value.")
    return values


def _extract_objective_score(mode: str, result: dict) -> float:
    if mode == "holistic":
        if "best_val_qwk" in result:
            return float(result["best_val_qwk"])
        grouped = result.get("groups", {})
        if not grouped:
            raise ValueError("Holistic result does not contain val qwk values.")
        weighted_sum = 0.0
        total_rows = 0
        for payload in grouped.values():
            rows = int(payload.get("rows", 0))
            group_result = payload.get("result", {})
            qwk = float(group_result["best_val_qwk"])
            weighted_sum += rows * qwk
            total_rows += rows
        if total_rows <= 0:
            raise ValueError("Holistic grouped result rows are empty.")
        return weighted_sum / float(total_rows)

    trait_results = result.get("results", {})
    if not trait_results:
        raise ValueError("Trait-score result does not contain per-trait results.")
    values = [float(payload["best_val_qwk"]) for payload in trait_results.values()]
    return sum(values) / float(len(values))


def _build_base_command(args) -> list[str]:
    command = "train-holistic" if args.mode == "holistic" else "train-trait-score"
    cmd = [
        command,
        "--dataset",
        args.dataset,
        "--csv_path",
        args.csv_path,
        "--trait_groups",
        args.trait_groups,
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--cpu_workers",
        str(args.cpu_workers),
        "--hidden_sizes",
        args.hidden_sizes,
        "--dropout",
        args.dropout,
        "--scheduler_t0",
        str(args.scheduler_t0),
        "--scheduler_tmult",
        str(args.scheduler_tmult),
        "--scheduler_eta_min",
        str(args.scheduler_eta_min),
        "--train_ratio",
        str(args.train_ratio),
        "--val_ratio",
        str(args.val_ratio),
        "--test_ratio",
        str(args.test_ratio),
        "--embedding_batch_size",
        str(args.embedding_batch_size),
        "--embedding_max_seq_length",
        str(args.embedding_max_seq_length),
    ]
    if args.npz_path is not None:
        cmd.extend(["--npz_path", args.npz_path])
    if args.trait_checkpoint_dir is not None:
        cmd.extend(["--trait_checkpoint_dir", args.trait_checkpoint_dir])
    if args.trait_model_name is not None:
        cmd.extend(["--trait_model_name", args.trait_model_name])
    if args.device is not None:
        cmd.extend(["--device", args.device])
    if args.split_by_column is not None and args.mode == "holistic":
        cmd.extend(["--split_by_column", args.split_by_column])

    if args.mode == "trait-score":
        cmd.append("--target_traits")
        cmd.extend(args.target_traits)

    if args.mode == "holistic":
        cmd.extend(
            [
                "--alpha",
                str(args.alpha),
                "--beta1",
                str(args.beta1),
                "--beta2",
                str(args.beta2),
                "--beta3",
                str(args.beta3),
                "--beta4",
                str(args.beta4),
                "--lambda1",
                str(args.lambda1),
                "--lambda2",
                str(args.lambda2),
                "--margin",
                str(args.margin),
                "--combined_stage1_epochs",
                str(args.combined_stage1_epochs),
                "--cl_score_binning",
                str(args.cl_score_binning),
            ]
        )

    if args.save_checkpoints:
        cmd.append("--save_checkpoints")
    else:
        cmd.append("--no_save_checkpoints")

    return cmd


def _build_trial_command(args, trial, study_dir: Path) -> tuple[list[str], dict]:
    params: dict[str, object] = {}
    params["batch_size"] = trial.suggest_categorical("batch_size", args.batch_size_choices)
    params["embedding_dim"] = trial.suggest_categorical("embedding_dim", args.embedding_dim_choices)
    params["learning_rate"] = trial.suggest_float(
        "learning_rate",
        args.lr_min,
        args.lr_max,
        log=True,
    )
    params["scheduler"] = trial.suggest_categorical("scheduler", args.scheduler_choices)
    params["ablation_mode"] = trial.suggest_categorical("ablation_mode", args.ablation_choices)
    params["warmup_epochs"] = trial.suggest_categorical("warmup_epochs", args.warmup_epochs_choices)
    params["moe_aux_weight"] = trial.suggest_float(
        "moe_aux_weight",
        args.moe_aux_weight_min,
        args.moe_aux_weight_max,
    )
    params["use_pre_homo_skip"] = trial.suggest_categorical(
        "use_pre_homo_skip",
        args.pre_homo_skip_choices,
    )
    params["use_pre_hetero_skip"] = trial.suggest_categorical(
        "use_pre_hetero_skip",
        args.pre_hetero_skip_choices,
    )
    if args.mode == "holistic":
        params["loss_type"] = trial.suggest_categorical("loss_type", args.loss_type_choices)
        params["backbone_mode"] = trial.suggest_categorical("backbone_mode", args.backbone_mode_choices)
        if params["backbone_mode"] == "e2e":
            params["backbone_lr"] = trial.suggest_float(
                "backbone_lr",
                args.backbone_lr_min,
                args.backbone_lr_max,
                log=True,
            )
            params["unfreeze_epoch"] = trial.suggest_categorical(
                "unfreeze_epoch",
                args.unfreeze_epoch_choices,
            )
        else:
            params["backbone_lr"] = args.backbone_lr_min
            params["unfreeze_epoch"] = 0

    trial_dir = study_dir / f"trial_{trial.number:04d}"
    trial_seed = int(args.seed + trial.number)

    cmd = _build_base_command(args)
    cmd.extend(
        [
            "--batch_size",
            str(params["batch_size"]),
            "--embedding_dim",
            str(params["embedding_dim"]),
            "--learning_rate",
            str(params["learning_rate"]),
            "--scheduler",
            str(params["scheduler"]),
            "--ablation_mode",
            str(params["ablation_mode"]),
            "--warmup_epochs",
            str(params["warmup_epochs"]),
            "--moe_aux_weight",
            str(params["moe_aux_weight"]),
            "--seed",
            str(trial_seed),
            "--output_dir",
            str(trial_dir),
        ]
    )
    if params["use_pre_homo_skip"]:
        cmd.append("--use_pre_homo_skip")
    if params["use_pre_hetero_skip"]:
        cmd.append("--use_pre_hetero_skip")
    if args.mode == "holistic":
        cmd.extend(["--loss_type", str(params["loss_type"])])
        cmd.extend(
            [
                "--backbone_mode",
                str(params["backbone_mode"]),
                "--backbone_lr",
                str(params["backbone_lr"]),
                "--unfreeze_epoch",
                str(params["unfreeze_epoch"]),
            ]
        )

    if args.use_wandb:
        cmd.extend(
            [
                "--use_wandb",
                "--wandb_project_name",
                args.wandb_project_name,
                "--run_name",
                f"{args.run_name_prefix}-trial{trial.number:04d}",
            ]
        )

    return cmd, params


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PromptAES2 Optuna tuner")
    parser.add_argument("--mode", type=str, choices=["holistic", "trait-score"], required=True)
    parser.add_argument("--dataset", type=str, choices=sorted(DATASET_PRESETS.keys()), required=True)
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--trait_groups", type=str, required=True)
    parser.add_argument("--target_traits", nargs="+", default=None)

    parser.add_argument("--npz_path", type=str, default=None)
    parser.add_argument("--trait_checkpoint_dir", type=str, default=None)
    parser.add_argument("--trait_model_name", type=str, default=None)
    parser.add_argument("--split_by_column", type=str, default=None)

    parser.add_argument("--output_dir", type=str, default="results/tuning")
    parser.add_argument("--study_name", type=str, default=None)
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--timeout_sec", type=int, default=None)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--cpu_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save_checkpoints", dest="save_checkpoints", action="store_true", default=False)
    parser.add_argument("--no_save_checkpoints", dest="save_checkpoints", action="store_false")

    parser.add_argument("--embedding_batch_size", type=int, default=32)
    parser.add_argument("--embedding_max_seq_length", type=int, default=512)
    parser.add_argument("--hidden_sizes", type=str, default="512-512")
    parser.add_argument("--dropout", type=str, default="0.1-0.1")
    parser.add_argument("--scheduler_t0", type=int, default=10)
    parser.add_argument("--scheduler_tmult", type=int, default=2)
    parser.add_argument("--scheduler_eta_min", type=float, default=1e-6)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)

    parser.add_argument("--batch_size_choices", type=str, default="16,32,64")
    parser.add_argument("--embedding_dim_choices", type=str, default="768")
    parser.add_argument("--scheduler_choices", type=str, default="none,cosine")
    parser.add_argument("--ablation_choices", type=str, default="homo_only,hetero_only,homo_hetero")
    parser.add_argument("--backbone_mode_choices", type=str, default="frozen")
    parser.add_argument("--unfreeze_epoch_choices", type=str, default="0")
    parser.add_argument("--warmup_epochs_choices", type=str, default="0")
    parser.add_argument("--pre_homo_skip_choices", type=str, default="false,true")
    parser.add_argument("--pre_hetero_skip_choices", type=str, default="false,true")
    parser.add_argument("--lr_min", type=float, default=1e-5)
    parser.add_argument("--lr_max", type=float, default=3e-4)
    parser.add_argument("--backbone_lr_min", type=float, default=1e-6)
    parser.add_argument("--backbone_lr_max", type=float, default=3e-5)
    parser.add_argument("--moe_aux_weight_min", type=float, default=0.0)
    parser.add_argument("--moe_aux_weight_max", type=float, default=0.02)

    parser.add_argument("--loss_type_choices", type=str, default="cross_entropy,combined")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta1", type=float, default=2.0)
    parser.add_argument("--beta2", type=float, default=1.0)
    parser.add_argument("--beta3", type=float, default=1.0)
    parser.add_argument("--beta4", type=float, default=1.0)
    parser.add_argument("--lambda1", type=float, default=5.0)
    parser.add_argument("--lambda2", type=float, default=10.0)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--combined_stage1_epochs", type=int, default=1)
    parser.add_argument("--cl_score_binning", type=str, choices=["auto", "none", "force6"], default="auto")

    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project_name", type=str, default="promptaes2-hpo")
    parser.add_argument("--run_name_prefix", type=str, default="hpo")
    return parser


def _validate_args(args, parser: argparse.ArgumentParser) -> None:
    if args.mode == "trait-score" and (not args.target_traits):
        parser.error("--target_traits is required for --mode trait-score")

    if args.n_trials < 1:
        parser.error("--n_trials must be >= 1")
    if args.timeout_sec is not None and args.timeout_sec < 1:
        parser.error("--timeout_sec must be >= 1")
    if args.epochs < 1:
        parser.error("--epochs must be >= 1")
    if args.patience < 1:
        parser.error("--patience must be >= 1")
    if args.lr_min <= 0 or args.lr_max <= 0:
        parser.error("--lr_min and --lr_max must be > 0")
    if args.lr_min >= args.lr_max:
        parser.error("--lr_min must be smaller than --lr_max")
    if args.backbone_lr_min <= 0 or args.backbone_lr_max <= 0:
        parser.error("--backbone_lr_min and --backbone_lr_max must be > 0")
    if args.backbone_lr_min >= args.backbone_lr_max:
        parser.error("--backbone_lr_min must be smaller than --backbone_lr_max")
    if args.moe_aux_weight_min < 0 or args.moe_aux_weight_max < 0:
        parser.error("--moe_aux_weight_min/max must be >= 0")
    if args.moe_aux_weight_min > args.moe_aux_weight_max:
        parser.error("--moe_aux_weight_min must be <= --moe_aux_weight_max")
    if args.split_by_column is not None:
        args.split_by_column = args.split_by_column.strip()
        if not args.split_by_column:
            parser.error("--split_by_column must not be empty")
    if args.npz_path is None and args.trait_checkpoint_dir is None:
        parser.error("Provide either --npz_path or --trait_checkpoint_dir")

    try:
        args.batch_size_choices = _parse_csv_int_choices(args.batch_size_choices, "batch_size_choices")
        args.embedding_dim_choices = _parse_csv_int_choices(args.embedding_dim_choices, "embedding_dim_choices")
        args.scheduler_choices = _parse_csv_str_choices(args.scheduler_choices, "scheduler_choices")
        args.ablation_choices = _parse_csv_str_choices(args.ablation_choices, "ablation_choices")
        args.backbone_mode_choices = _parse_csv_str_choices(args.backbone_mode_choices, "backbone_mode_choices")
        args.unfreeze_epoch_choices = _parse_csv_int_choices(args.unfreeze_epoch_choices, "unfreeze_epoch_choices")
        args.warmup_epochs_choices = _parse_csv_int_choices(args.warmup_epochs_choices, "warmup_epochs_choices")
        args.pre_homo_skip_choices = _parse_csv_bool_choices(args.pre_homo_skip_choices, "pre_homo_skip_choices")
        args.pre_hetero_skip_choices = _parse_csv_bool_choices(
            args.pre_hetero_skip_choices, "pre_hetero_skip_choices"
        )
        args.loss_type_choices = _parse_csv_str_choices(args.loss_type_choices, "loss_type_choices")
    except ValueError as exc:
        parser.error(str(exc))

    allowed_schedulers = {"none", "cosine"}
    for scheduler in args.scheduler_choices:
        if scheduler not in allowed_schedulers:
            parser.error(f"Unsupported scheduler choice '{scheduler}'. Allowed: {sorted(allowed_schedulers)}")

    allowed_ablations = {"none", "homo_only", "hetero_only", "homo_hetero"}
    for ablation in args.ablation_choices:
        if ablation not in allowed_ablations:
            parser.error(f"Unsupported ablation choice '{ablation}'. Allowed: {sorted(allowed_ablations)}")

    allowed_backbone_modes = {"frozen", "e2e"}
    for backbone_mode in args.backbone_mode_choices:
        if backbone_mode not in allowed_backbone_modes:
            parser.error(
                f"Unsupported backbone_mode choice '{backbone_mode}'. "
                f"Allowed: {sorted(allowed_backbone_modes)}"
            )
    if any(epoch < 0 for epoch in args.unfreeze_epoch_choices):
        parser.error("--unfreeze_epoch_choices must contain non-negative integers")
    if args.mode == "holistic" and "e2e" in args.backbone_mode_choices:
        if args.trait_checkpoint_dir is None:
            parser.error("--trait_checkpoint_dir is required when backbone_mode_choices include e2e")
        if args.npz_path is not None:
            parser.error("--npz_path cannot be used when backbone_mode_choices include e2e")

    allowed_losses = {"cross_entropy", "combined"}
    for loss_name in args.loss_type_choices:
        if loss_name not in allowed_losses:
            parser.error(f"Unsupported loss_type choice '{loss_name}'. Allowed: {sorted(allowed_losses)}")


def main(argv: list[str] | None = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(args, parser)

    try:
        import optuna  # pylint: disable=import-outside-toplevel
    except ImportError:
        parser.exit(
            2,
            "Error: optuna is not installed. Install with `python -m pip install optuna`.\n",
        )

    train_parser = build_train_parser()

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    study_name = (
        args.study_name
        if args.study_name is not None
        else f"{args.mode}_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    study_dir = output_root / study_name
    study_dir.mkdir(parents=True, exist_ok=True)

    print(f"HPO mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Trials: {args.n_trials}")
    print(f"Study: {study_name}")
    print(f"Output: {study_dir}")
    print("Objective: maximize best_val_qwk")

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=min(5, max(1, args.n_trials // 4)))
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=args.storage,
        load_if_exists=True,
    )

    def objective(trial):
        trial_cmd, sampled_params = _build_trial_command(args, trial, study_dir=study_dir)
        command_text = "python -m promptaes2.cli " + " ".join(shlex.quote(token) for token in trial_cmd)
        trial.set_user_attr("command", command_text)
        trial.set_user_attr("sampled_params", sampled_params)

        parsed = train_parser.parse_args(trial_cmd)
        _validate_parsed_args(parsed, train_parser)
        try:
            result = parsed.func(parsed)
            score = _extract_objective_score(args.mode, result)
        except (ConfigError, MissingColumnError, DataAlignmentError, RuntimeError, ValueError) as exc:
            trial.set_user_attr("error", str(exc))
            return -1e9
        trial.set_user_attr("score", score)
        return score

    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout_sec,
        gc_after_trial=True,
        show_progress_bar=True,
    )

    best = study.best_trial
    summary = {
        "study_name": study_name,
        "mode": args.mode,
        "dataset": args.dataset,
        "objective": "best_val_qwk",
        "best_value": float(best.value),
        "best_params": best.params,
        "best_command": best.user_attrs.get("command"),
        "n_trials": len(study.trials),
    }
    summary_path = study_dir / "best_trial.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)

    print("\nBest trial:")
    print(f"  value: {best.value:.6f}")
    print(f"  params: {best.params}")
    print(f"  command: {best.user_attrs.get('command')}")
    print(f"Saved summary: {summary_path}")

    return summary


if __name__ == "__main__":
    main()
