from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from promptaes2.config import ConfigError, validate_required_columns
from promptaes2.data.alignment import align_npz_and_csv, extract_trait_embeddings
from promptaes2.data.datasets import MultiEmbeddingDataset
from promptaes2.models.factory import build_scoring_model
from promptaes2.training.holistic import _extract_embeddings_from_trait_checkpoints
from promptaes2.utils.checkpoint import EarlyStopping, build_checkpoint_name
from promptaes2.utils.metrics import calculate_accuracy_qwk


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _trait_groups_to_names(trait_groups):
    names: list[str] = []
    for trait_names, dim in trait_groups:
        if dim != len(trait_names):
            raise ConfigError(
                f"Trait group dimension mismatch for {trait_names}: dim={dim}, len={len(trait_names)}"
            )
        names.extend(trait_names)
    return names


def _format_label_distribution(labels: np.ndarray) -> str:
    counts = pd.Series(labels).value_counts(dropna=False)
    sorted_items = sorted(counts.items(), key=lambda item: str(item[0]))
    total = int(counts.sum())
    if total == 0:
        return "n/a"

    parts: list[str] = []
    for label, count in sorted_items:
        ratio = (float(count) / float(total)) * 100.0
        label_text = "NaN" if pd.isna(label) else str(label)
        parts.append(f"{label_text}:{int(count)} ({ratio:.1f}%)")
    return ", ".join(parts)


def _print_split_distributions(train_labels: np.ndarray, val_labels: np.ndarray, test_labels: np.ndarray) -> None:
    total = len(train_labels) + len(val_labels) + len(test_labels)
    if total == 0:
        return

    print(
        "Split sizes: "
        f"train={len(train_labels)} ({(len(train_labels) / total) * 100:.1f}%), "
        f"val={len(val_labels)} ({(len(val_labels) / total) * 100:.1f}%), "
        f"test={len(test_labels)} ({(len(test_labels) / total) * 100:.1f}%)"
    )
    print(f"Train distribution: {_format_label_distribution(train_labels)}")
    print(f"Valid distribution: {_format_label_distribution(val_labels)}")
    print(f"Test  distribution: {_format_label_distribution(test_labels)}")


_PREDEFINED_SPLIT_ALIASES = {
    "train": "train",
    "dev": "val",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "test",
}


def _normalize_predefined_split_value(value) -> str | None:
    if pd.isna(value):
        return None
    normalized = str(value).strip().lower()
    if not normalized:
        return None
    return _PREDEFINED_SPLIT_ALIASES.get(normalized)


def _build_predefined_split_indices(
    split_values: pd.Series,
    *,
    split_column: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normalized_splits = split_values.map(_normalize_predefined_split_value)
    invalid_mask = normalized_splits.isna()
    if invalid_mask.any():
        invalid_values = sorted(
            {
                "NaN" if pd.isna(raw_value) else str(raw_value)
                for raw_value in split_values[invalid_mask]
            }
        )
        invalid_text = ", ".join(invalid_values)
        raise ConfigError(
            f"Predefined split column '{split_column}' contains invalid values: {invalid_text}. "
            "Expected train/dev/test (or val/valid/validation)."
        )

    split_array = normalized_splits.to_numpy(dtype=object)
    train_idx = np.flatnonzero(split_array == "train")
    val_idx = np.flatnonzero(split_array == "val")
    test_idx = np.flatnonzero(split_array == "test")

    missing_splits = [
        split_name
        for split_name, split_indices in (("train", train_idx), ("val", val_idx), ("test", test_idx))
        if len(split_indices) == 0
    ]
    if missing_splits:
        missing_text = ", ".join(missing_splits)
        raise ValueError(
            f"predefined split column '{split_column}' is missing {missing_text} rows after filtering"
        )

    return train_idx, val_idx, test_idx


def _maybe_init_wandb(args, target_trait: str):
    if not bool(getattr(args, "use_wandb", False)):
        return None

    try:
        import wandb  # pylint: disable=import-outside-toplevel
    except ImportError:
        print("wandb is not installed. Continuing without wandb logging.")
        return None

    run_name = getattr(args, "run_name", None)
    if run_name:
        run_name = f"{run_name}-{target_trait}"

    run = wandb.init(
        project=args.wandb_project_name,
        name=run_name,
        config={
            "mode": "trait-score",
            "dataset": args.dataset,
            "target_trait": target_trait,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "scheduler": args.scheduler,
            "epochs": args.epochs,
            "patience": args.patience,
            "evolution_stage": args.evolution_stage,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "predefined_split_column": getattr(args, "predefined_split_column", None),
            "moe_aux_weight": args.moe_aux_weight,
            "model_variant": getattr(args, "model_variant", "legacy"),
            "group_num_experts": int(getattr(args, "group_num_experts", 8)),
            "classifier_num_experts": int(getattr(args, "classifier_num_experts", 8)),
            "router_top_k": int(getattr(args, "router_top_k", 2)),
        },
    )
    return run


def _log_wandb(run, payload: dict) -> None:
    if run is not None:
        run.log(payload)


def _resolve_embeddings_and_essay(args, required_embedding_traits: list[str], device: torch.device):
    if args.npz_path is not None:
        aligned_npz, essay, _ = align_npz_and_csv(args.npz_path, args.csv_path)
        embeddings_dict = extract_trait_embeddings(aligned_npz)
        return embeddings_dict, essay

    embeddings_dict, essay = _extract_embeddings_from_trait_checkpoints(
        args=args,
        trait_names=required_embedding_traits,
        device=device,
    )
    return embeddings_dict, essay


def _build_loader(embeddings_dict: dict[str, np.ndarray], labels: np.ndarray, indices: np.ndarray, args, shuffle: bool):
    subset_embeddings = {trait: emb[indices] for trait, emb in embeddings_dict.items()}
    subset_labels = labels[indices]
    dataset = MultiEmbeddingDataset(subset_embeddings, subset_labels)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=args.cpu_workers,
    )


def _train_one_epoch(model, loader: DataLoader, optimizer, criterion, device: torch.device, moe_aux_weight: float):
    model.train()
    total_loss = 0.0
    main_loss_total = 0.0
    aux_loss_total = 0.0
    predictions: list[int] = []
    labels_all: list[int] = []

    for batch_embeddings, labels_batch in loader:
        optimizer.zero_grad()
        batch_embeddings = {trait: emb.to(device) for trait, emb in batch_embeddings.items()}
        labels_batch = labels_batch.to(device)

        output = model(batch_embeddings)
        main_loss = criterion(output, labels_batch)
        moe_aux_loss = model.moe_aux_loss
        if moe_aux_loss is None:
            moe_aux_loss = output.new_tensor(0.0)

        loss = main_loss + (moe_aux_weight * moe_aux_loss)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        main_loss_total += float(main_loss.item())
        aux_loss_total += float(moe_aux_loss.item())
        predictions.extend(torch.argmax(output, dim=1).cpu().numpy().tolist())
        labels_all.extend(labels_batch.cpu().numpy().tolist())

    acc, qwk = calculate_accuracy_qwk(labels_all, predictions)
    return (
        total_loss / len(loader),
        main_loss_total / len(loader),
        aux_loss_total / len(loader),
        acc,
        qwk,
    )


def _validate_one_epoch(model, loader: DataLoader, criterion, device: torch.device, moe_aux_weight: float):
    model.eval()
    total_loss = 0.0
    main_loss_total = 0.0
    aux_loss_total = 0.0
    predictions: list[int] = []
    labels_all: list[int] = []

    with torch.no_grad():
        for batch_embeddings, labels_batch in loader:
            batch_embeddings = {trait: emb.to(device) for trait, emb in batch_embeddings.items()}
            labels_batch = labels_batch.to(device)

            output = model(batch_embeddings)
            main_loss = criterion(output, labels_batch)
            moe_aux_loss = model.moe_aux_loss
            if moe_aux_loss is None:
                moe_aux_loss = output.new_tensor(0.0)
            loss = main_loss + (moe_aux_weight * moe_aux_loss)

            total_loss += float(loss.item())
            main_loss_total += float(main_loss.item())
            aux_loss_total += float(moe_aux_loss.item())
            predictions.extend(torch.argmax(output, dim=1).cpu().numpy().tolist())
            labels_all.extend(labels_batch.cpu().numpy().tolist())

    acc, qwk = calculate_accuracy_qwk(labels_all, predictions)
    return (
        total_loss / len(loader),
        main_loss_total / len(loader),
        aux_loss_total / len(loader),
        acc,
        qwk,
    )


def _print_final_summary(final_results: dict[str, dict[str, float | int]]) -> dict[str, float]:
    print("\n" + "=" * 100)
    print("FINAL TRAIT-SCORE SUMMARY")
    print("=" * 100)
    print(f"{'Trait':20s} {'Val QWK':>10s} {'Test QWK':>10s} {'Test Acc':>10s} {'Epoch':>8s} {'Labels':>8s}")
    print("-" * 100)

    sorted_results = sorted(final_results.items(), key=lambda item: item[1]["test_qwk"], reverse=True)
    for trait_name, result in sorted_results:
        print(
            f"{trait_name:20s} {result['best_val_qwk']:>10.4f} {result['test_qwk']:>10.4f} "
            f"{result['test_acc']:>10.4f} {int(result['best_epoch']):>8d} {int(result['labels']):>8d}"
        )

    test_qwk_values = [float(result["test_qwk"]) for result in final_results.values()]
    avg_qwk = float(np.mean(test_qwk_values))
    best_qwk = float(np.max(test_qwk_values))
    worst_qwk = float(np.min(test_qwk_values))

    print("-" * 100)
    print(f"Average Test QWK: {avg_qwk:.4f}")
    print(f"Best Test QWK:    {best_qwk:.4f}")
    print(f"Worst Test QWK:   {worst_qwk:.4f}")
    return {
        "average_test_qwk": avg_qwk,
        "best_test_qwk": best_qwk,
        "worst_test_qwk": worst_qwk,
    }


def run_trait_score_training(args):
    _set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    required_embedding_traits = _trait_groups_to_names(args.trait_groups)

    embeddings_dict, essay = _resolve_embeddings_and_essay(
        args=args,
        required_embedding_traits=required_embedding_traits,
        device=device,
    )

    missing_embedding_traits = [name for name in required_embedding_traits if name not in embeddings_dict]
    if missing_embedding_traits:
        available = ", ".join(sorted(embeddings_dict.keys()))
        missing = ", ".join(missing_embedding_traits)
        raise ConfigError(
            f"Missing required embeddings for trait groups: {missing}. Available embeddings: {available}"
        )

    validate_required_columns(essay.columns, args.target_traits)

    invalid_targets = [trait for trait in args.target_traits if trait not in required_embedding_traits]
    if invalid_targets:
        invalid_text = ", ".join(invalid_targets)
        raise ConfigError(
            f"Target traits must be included in --trait_groups. Invalid targets: {invalid_text}"
        )

    moe_aux_weight = float(getattr(args, "moe_aux_weight", 0.01))
    save_checkpoints = bool(getattr(args, "save_checkpoints", True))
    predefined_split_column = getattr(args, "predefined_split_column", None)
    if predefined_split_column is not None:
        validate_required_columns(essay.columns, [predefined_split_column])

    print(f"Dataset: {args.dataset}")
    print(f"Mode: trait-score")
    print(f"Target traits: {args.target_traits}")
    print(f"Trait groups: {args.trait_groups}")
    model_variant = str(getattr(args, "model_variant", "legacy"))
    print(f"Model variant: {model_variant}")
    if model_variant == "canonical_moe":
        print(
            "Canonical MoE settings: "
            f"group_num_experts={int(getattr(args, 'group_num_experts', 8))}, "
            f"classifier_num_experts={int(getattr(args, 'classifier_num_experts', 8))}, "
            f"router_top_k={int(getattr(args, 'router_top_k', 2))}"
        )
    print(f"Ablation mode: {getattr(args, 'ablation_mode', 'none')}")
    print(
        f"Evolution stage: {args.evolution_stage} (warmup_epochs={args.warmup_epochs})"
    )
    print(
        "Pre-skip toggles: "
        f"pre_homo={bool(getattr(args, 'use_pre_homo_skip', False))}, "
        f"pre_hetero={bool(getattr(args, 'use_pre_hetero_skip', False))}"
    )
    print(f"Save checkpoints: {save_checkpoints}")
    if predefined_split_column is None:
        print(
            "Split ratios: "
            f"train={args.train_ratio:.2f}, val={args.val_ratio:.2f}, test={args.test_ratio:.2f}"
        )
    else:
        print(f"Predefined split column: {predefined_split_column}")
    print(f"Output dir: {output_dir}")

    final_results: dict[str, dict[str, float | int]] = {}
    skipped_traits: list[tuple[str, str]] = []

    for target_trait in args.target_traits:
        labels_series = essay[target_trait]
        valid_mask = ~labels_series.isna()
        valid_indices = np.where(valid_mask.to_numpy())[0]
        if len(valid_indices) == 0:
            skipped_traits.append((target_trait, "no non-null labels"))
            print(f"Skipping '{target_trait}': no non-null labels.")
            continue

        trait_labels_raw = labels_series.iloc[valid_indices].to_numpy()
        unique_labels = np.unique(trait_labels_raw)
        if len(unique_labels) < 2:
            skipped_traits.append((target_trait, "less than 2 unique labels"))
            print(f"Skipping '{target_trait}': requires at least 2 unique labels.")
            continue

        label_mapping = {old: new for new, old in enumerate(sorted(unique_labels))}
        labels = np.array([label_mapping[label] for label in trait_labels_raw])

        try:
            if predefined_split_column is not None:
                split_values = essay.iloc[valid_indices][predefined_split_column].reset_index(drop=True)
                train_idx, val_idx, test_idx = _build_predefined_split_indices(
                    split_values,
                    split_column=predefined_split_column,
                )
            else:
                holdout_ratio = float(args.val_ratio + args.test_ratio)
                test_ratio_within_holdout = float(args.test_ratio / holdout_ratio)
                indices = np.arange(len(labels))
                train_idx, holdout_idx = train_test_split(
                    indices,
                    test_size=holdout_ratio,
                    random_state=args.seed,
                    stratify=labels,
                )
                val_idx, test_idx = train_test_split(
                    holdout_idx,
                    test_size=test_ratio_within_holdout,
                    random_state=args.seed,
                    stratify=labels[holdout_idx],
                )
        except ValueError as exc:
            reason = (
                f"predefined split failed ({exc})"
                if predefined_split_column is not None
                else f"stratified split failed ({exc})"
            )
            skipped_traits.append((target_trait, reason))
            print(f"Skipping '{target_trait}': {reason}.")
            print(f"Label distribution: {_format_label_distribution(trait_labels_raw)}")
            continue

        trait_embeddings = {
            trait_name: emb[valid_indices]
            for trait_name, emb in embeddings_dict.items()
        }
        train_loader = _build_loader(trait_embeddings, labels, train_idx, args, shuffle=True)
        val_loader = _build_loader(trait_embeddings, labels, val_idx, args, shuffle=False)
        test_loader = _build_loader(trait_embeddings, labels, test_idx, args, shuffle=False)

        if len(train_loader) == 0 or len(val_loader) == 0 or len(test_loader) == 0:
            skipped_traits.append((target_trait, "empty loader after split"))
            print(f"Skipping '{target_trait}': empty loader after split.")
            continue

        print(f"\nTraining trait-score target: {target_trait} ({len(unique_labels)} classes)")
        _print_split_distributions(labels[train_idx], labels[val_idx], labels[test_idx])

        try:
            model = build_scoring_model(
                model_variant=model_variant,
                embedding_dim=args.embedding_dim,
                hidden_sizes=args.hidden_sizes,
                num_classes=len(unique_labels),
                dropout_rates=args.dropout_rates,
                use_skip1=args.use_skip1,
                use_skip2=args.use_skip2,
                use_skip3=args.use_skip3,
                use_pre_homo_skip=bool(getattr(args, "use_pre_homo_skip", False)),
                use_pre_hetero_skip=bool(getattr(args, "use_pre_hetero_skip", False)),
                trait_groups=args.trait_groups,
                evolution_stage=args.evolution_stage,
                warmup_epochs=args.warmup_epochs,
                group_num_experts=int(getattr(args, "group_num_experts", 8)),
                classifier_num_experts=int(getattr(args, "classifier_num_experts", 8)),
                router_top_k=int(getattr(args, "router_top_k", 2)),
            ).to(device)
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc

        optimizer = torch.optim.RAdam(model.parameters(), lr=args.learning_rate)
        scheduler = (
            CosineAnnealingWarmRestarts(
                optimizer,
                T_0=args.scheduler_t0,
                T_mult=args.scheduler_tmult,
                eta_min=args.scheduler_eta_min,
            )
            if args.scheduler == "cosine"
            else None
        )
        criterion = nn.CrossEntropyLoss().to(device)

        checkpoint_path = output_dir / build_checkpoint_name(args.dataset, "trait_score", target_trait)
        stopper = (
            EarlyStopping(
                patience=args.patience,
                path=checkpoint_path,
                metric="qwk",
                verbose=True,
            )
            if save_checkpoints
            else None
        )

        best_val_qwk = -np.inf
        best_epoch = 0
        best_state_dict: dict[str, torch.Tensor] | None = None
        no_improve_epochs = 0
        wandb_run = _maybe_init_wandb(args, target_trait=target_trait)
        try:
            for epoch in tqdm(range(args.epochs), desc=f"trait-score:{target_trait}"):
                if hasattr(model, "set_epoch"):
                    model.set_epoch(epoch)

                train_total, train_main, train_aux, train_acc, train_qwk = _train_one_epoch(
                    model,
                    train_loader,
                    optimizer,
                    criterion,
                    device,
                    moe_aux_weight,
                )
                if scheduler is not None:
                    scheduler.step(epoch + 1)
                lr = optimizer.param_groups[0]["lr"]

                val_total, val_main, val_aux, val_acc, val_qwk = _validate_one_epoch(
                    model,
                    val_loader,
                    criterion,
                    device,
                    moe_aux_weight,
                )

                _log_wandb(
                    wandb_run,
                    {
                        "epoch": epoch + 1,
                        "train/total_loss": train_total,
                        "train/main_loss": train_main,
                        "train/aux_loss": train_aux,
                        "train/acc": train_acc,
                        "train/qwk": train_qwk,
                        "val/total_loss": val_total,
                        "val/main_loss": val_main,
                        "val/aux_loss": val_aux,
                        "val/acc": val_acc,
                        "val/qwk": val_qwk,
                        "lr": lr,
                    },
                )

                print(
                    f"Epoch {epoch + 1}/{args.epochs} - "
                    f"train_total={train_total:.4f}, train_main={train_main:.4f}, train_aux={train_aux:.4f}, "
                    f"val_total={val_total:.4f}, val_main={val_main:.4f}, val_aux={val_aux:.4f}, "
                    f"val_acc={val_acc:.4f}, val_qwk={val_qwk:.4f}, lr={lr:.6g}"
                )

                if val_qwk > best_val_qwk:
                    best_val_qwk = val_qwk
                    best_epoch = epoch + 1
                    if not save_checkpoints:
                        best_state_dict = {
                            key: value.detach().cpu().clone()
                            for key, value in model.state_dict().items()
                        }
                        no_improve_epochs = 0
                elif not save_checkpoints:
                    no_improve_epochs += 1

                if save_checkpoints:
                    assert stopper is not None
                    stop = stopper.step(val_qwk, model)
                    if stop.early_stop:
                        print(f"Early stopping triggered for '{target_trait}'.")
                        break
                elif no_improve_epochs >= args.patience:
                    print(
                        f"Early stopping triggered for '{target_trait}' "
                        "(in-memory best model, checkpoint saving disabled)."
                    )
                    break
        finally:
            if wandb_run is not None:
                wandb_run.finish()

        if save_checkpoints:
            if not checkpoint_path.exists():
                skipped_traits.append((target_trait, "checkpoint not created"))
                print(f"Skipping '{target_trait}': checkpoint was not created.")
                continue
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        elif best_state_dict is None:
            skipped_traits.append((target_trait, "best in-memory model not available"))
            print(f"Skipping '{target_trait}': best in-memory model not available.")
            continue
        else:
            model.load_state_dict(best_state_dict)

        test_total, test_main, test_aux, test_acc, test_qwk = _validate_one_epoch(
            model,
            test_loader,
            criterion,
            device,
            moe_aux_weight,
        )

        final_model_path = output_dir / build_checkpoint_name(args.dataset, "trait_score_model", target_trait)
        if save_checkpoints:
            torch.save(model.state_dict(), final_model_path)

        final_results[target_trait] = {
            "best_val_qwk": float(best_val_qwk),
            "best_epoch": int(best_epoch),
            "test_qwk": float(test_qwk),
            "test_acc": float(test_acc),
            "test_total_loss": float(test_total),
            "test_main_loss": float(test_main),
            "test_aux_loss": float(test_aux),
            "labels": int(len(unique_labels)),
            "checkpoint_path": str(checkpoint_path) if save_checkpoints else None,
            "model_path": str(final_model_path) if save_checkpoints else None,
        }

        print(
            f"Completed '{target_trait}': best_val_qwk={best_val_qwk:.4f} "
            f"(epoch {best_epoch}), test_qwk={test_qwk:.4f}, test_acc={test_acc:.4f}"
        )

    if not final_results:
        detail = "; ".join(f"{name}: {reason}" for name, reason in skipped_traits) or "unknown reason"
        raise ConfigError(f"No target traits were trained. Details: {detail}")

    stats = _print_final_summary(final_results)
    if skipped_traits:
        print("\nSkipped traits:")
        for name, reason in skipped_traits:
            print(f"- {name}: {reason}")

    return {
        "target_traits": list(args.target_traits),
        "results": final_results,
        "stats": stats,
        "skipped_traits": [{"trait": name, "reason": reason} for name, reason in skipped_traits],
    }
