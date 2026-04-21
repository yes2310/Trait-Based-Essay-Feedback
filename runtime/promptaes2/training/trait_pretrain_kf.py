from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer, RobertaConfig, RobertaForSequenceClassification

from promptaes2.config import ConfigError, get_dataset_preset, validate_required_columns
from promptaes2.utils.checkpoint import EarlyStopping, build_checkpoint_name
from promptaes2.utils.metrics import calculate_accuracy_qwk


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_tokenizer_and_model_name(model_name: str):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to load tokenizer '{model_name}'. "
            "Check internet connectivity or pre-download to local cache."
        ) from exc
    return tokenizer


def _convert_dataframe_to_tensors(
    data: pd.DataFrame,
    tokenizer,
    max_seq_length: int,
    batch_size: int,
    cpu_workers: int,
    label_name: str,
) -> DataLoader:
    inputs = tokenizer(
        data["text"].tolist(),
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=max_seq_length,
    )

    labels_raw = data[label_name].tolist()
    min_label = min(labels_raw)
    labels_normalized = [int(label - min_label) for label in labels_raw]
    labels = torch.tensor(labels_normalized).long()

    dataset = torch.utils.data.TensorDataset(inputs.input_ids, inputs.attention_mask, labels)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=RandomSampler(dataset),
        num_workers=cpu_workers,
    )


def _format_label_distribution(labels: pd.Series) -> str:
    counts = labels.value_counts(dropna=False)
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


def _print_trait_data_summary(trait_name: str, selected_data: pd.DataFrame, total_rows: int) -> None:
    sample_count = len(selected_data)
    missing_count = total_rows - sample_count
    print(f"\n[Data Summary] trait='{trait_name}'")
    print(f"Samples: {sample_count}/{total_rows} (missing={missing_count})")
    if sample_count > 0:
        print(f"Label distribution: {_format_label_distribution(selected_data[trait_name])}")


def _print_split_distributions(
    trait_name: str,
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> None:
    total = len(train_data) + len(val_data) + len(test_data)
    if total == 0:
        return

    print(
        "Split sizes: "
        f"train={len(train_data)} ({(len(train_data) / total) * 100:.1f}%), "
        f"val={len(val_data)} ({(len(val_data) / total) * 100:.1f}%), "
        f"test={len(test_data)} ({(len(test_data) / total) * 100:.1f}%)"
    )
    print(f"Train distribution: {_format_label_distribution(train_data[trait_name])}")
    print(f"Valid distribution: {_format_label_distribution(val_data[trait_name])}")
    print(f"Test  distribution: {_format_label_distribution(test_data[trait_name])}")


def _train_epoch(model, dataloader: DataLoader, optimizer, device: torch.device) -> tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    all_labels: list[int] = []
    all_predictions: list[int] = []

    for batch in tqdm(dataloader, leave=False):
        inputs, masks, labels = [item.to(device) for item in batch]

        optimizer.zero_grad()
        outputs = model(input_ids=inputs, attention_mask=masks, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        predictions = torch.argmax(outputs.logits, dim=1).cpu().numpy().tolist()
        labels_np = labels.cpu().numpy().tolist()

        all_predictions.extend(predictions)
        all_labels.extend(labels_np)

    accuracy, qwk = calculate_accuracy_qwk(all_labels, all_predictions)
    return total_loss / len(dataloader), accuracy, qwk


def _validate_epoch(model, dataloader: DataLoader, device: torch.device) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    all_labels: list[int] = []
    all_predictions: list[int] = []

    with torch.no_grad():
        for batch in dataloader:
            inputs, masks, labels = [item.to(device) for item in batch]
            outputs = model(input_ids=inputs, attention_mask=masks, labels=labels)
            total_loss += float(outputs.loss.item())

            predictions = torch.argmax(outputs.logits, dim=1).cpu().numpy().tolist()
            labels_np = labels.cpu().numpy().tolist()

            all_predictions.extend(predictions)
            all_labels.extend(labels_np)

    accuracy, qwk = calculate_accuracy_qwk(all_labels, all_predictions)
    return total_loss / len(dataloader), accuracy, qwk


def _resolve_traits_and_paths(args):
    preset = get_dataset_preset(args.dataset)
    traits = args.traits if args.traits else preset.default_traits
    data_path = args.data_path if args.data_path else preset.default_data_path
    model_name = args.model_name if args.model_name else preset.default_model_name
    return traits, data_path, model_name


def _sanitize_for_path(value: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z._-]+", "_", value).strip("._-")
    if not normalized:
        return "value"
    return normalized[:64]


def _maybe_init_wandb(args, *, trait_name: str, run_name_override: str | None = None):
    if not bool(getattr(args, "use_wandb", False)):
        return None

    try:
        import wandb  # pylint: disable=import-outside-toplevel
    except ImportError:
        print("wandb is not installed. Continuing without wandb logging.")
        return None

    run = wandb.init(
        project=str(getattr(args, "wandb_project_name", "promptaes2-trait")),
        name=run_name_override if run_name_override is not None else getattr(args, "run_name", None),
        config={
            "dataset": args.dataset,
            "trait_name": trait_name,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "max_seq_length": args.max_seq_length,
            "scheduler": getattr(args, "scheduler", "none"),
            "epochs": int(getattr(args, "epochs", 20)),
            "auto_stop": bool(getattr(args, "auto_stop", False)),
            "early_stopping_patience": int(getattr(args, "early_stopping_patience", 5)),
        },
    )
    return run


def _log_wandb(run, payload: dict) -> None:
    if run is not None:
        run.log(payload)


def run_trait_pretrain(args):
    _set_seed(args.seed)

    traits, data_path, model_name = _resolve_traits_and_paths(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = int(getattr(args, "epochs", 20))
    auto_stop = bool(getattr(args, "auto_stop", False))
    early_stopping_patience = int(getattr(args, "early_stopping_patience", 5))
    scheduler_type = str(getattr(args, "scheduler", "none"))
    scheduler_t0 = int(getattr(args, "scheduler_t0", 10))
    scheduler_tmult = int(getattr(args, "scheduler_tmult", 2))
    scheduler_eta_min = float(getattr(args, "scheduler_eta_min", 1e-6))
    print_epoch_metrics = bool(getattr(args, "print_epoch_metrics", False))
    use_wandb = bool(getattr(args, "use_wandb", False))

    print(f"Dataset: {args.dataset}")
    print(f"Traits: {traits}")
    print(f"Data path: {data_path}")
    print(f"Model name: {model_name}")
    print(f"Epochs: {epochs}")
    print(f"Scheduler: {scheduler_type}")
    print(f"Auto stop: {auto_stop} (patience={early_stopping_patience})")
    print(f"Print epoch metrics: {print_epoch_metrics}")
    print(f"W&B logging: {use_wandb}")
    if use_wandb:
        print(f"W&B project: {getattr(args, 'wandb_project_name', 'promptaes2-trait')}")
    print(
        "Note: The Transformers log about newly initialized "
        "'classifier.*' weights is expected for trait-specific heads."
    )

    data = pd.read_csv(data_path)
    validate_required_columns(data.columns, ["text", *traits])

    tokenizer = _load_tokenizer_and_model_name(model_name)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    final_results: dict[str, dict[str, float | int]] = {}

    for trait_name in traits:
        selected_data = data[["text", trait_name]].dropna()
        _print_trait_data_summary(trait_name, selected_data, total_rows=len(data))
        if selected_data.empty:
            print(f"Skipping '{trait_name}': no non-null samples.")
            continue

        unique_labels = sorted(selected_data[trait_name].unique())
        unique_label_count = len(unique_labels)
        if unique_label_count < 2:
            print(f"Skipping '{trait_name}': requires at least 2 unique labels.")
            continue

        try:
            train_data, temp_data = train_test_split(
                selected_data,
                test_size=0.3,
                random_state=args.seed,
                stratify=selected_data[trait_name],
            )
            val_data, test_data = train_test_split(
                temp_data,
                test_size=0.5,
                random_state=args.seed,
                stratify=temp_data[trait_name],
            )
        except ValueError as exc:
            print(f"Skipping '{trait_name}': stratified split failed ({exc}).")
            print(f"Label distribution: {_format_label_distribution(selected_data[trait_name])}")
            print("Hint: each label should have enough samples in train/val/test during stratified split.")
            continue

        _print_split_distributions(trait_name, train_data, val_data, test_data)

        try:
            model_config = RobertaConfig.from_pretrained(model_name, num_labels=unique_label_count)
            model = RobertaForSequenceClassification.from_pretrained(
                model_name,
                config=model_config,
                ignore_mismatched_sizes=True,
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to load model '{model_name}'. "
                "Retry with internet enabled or use an offline cached model path."
            ) from exc

        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        scheduler = (
            CosineAnnealingWarmRestarts(
                optimizer,
                T_0=scheduler_t0,
                T_mult=scheduler_tmult,
                eta_min=scheduler_eta_min,
            )
            if scheduler_type == "cosine"
            else None
        )

        train_loader = _convert_dataframe_to_tensors(
            train_data,
            tokenizer,
            args.max_seq_length,
            args.batch_size,
            args.cpu_workers,
            trait_name,
        )
        val_loader = _convert_dataframe_to_tensors(
            val_data,
            tokenizer,
            args.max_seq_length,
            args.batch_size,
            args.cpu_workers,
            trait_name,
        )
        test_loader = _convert_dataframe_to_tensors(
            test_data,
            tokenizer,
            args.max_seq_length,
            args.batch_size,
            args.cpu_workers,
            trait_name,
        )

        checkpoint_name = build_checkpoint_name(args.dataset, "trait", trait_name)
        checkpoint_path = output_dir / checkpoint_name
        stopper_patience = early_stopping_patience if auto_stop else max(epochs + 1, early_stopping_patience)
        stopper = EarlyStopping(
            patience=stopper_patience,
            path=checkpoint_path,
            metric="qwk",
            verbose=True,
        )

        writer = SummaryWriter(log_dir=str(output_dir / "tensorboard" / f"{args.dataset}_{trait_name}"))
        wandb_run_name = None
        base_run_name = getattr(args, "run_name", None)
        if base_run_name:
            wandb_run_name = f"{base_run_name}-{_sanitize_for_path(trait_name)}"
        wandb_run = _maybe_init_wandb(
            args,
            trait_name=trait_name,
            run_name_override=wandb_run_name,
        )

        best_epoch = 0
        best_val_qwk = -1.0

        print(f"\nTraining trait: {trait_name} ({unique_label_count} classes)")
        try:
            for epoch in range(epochs):
                train_loss, train_acc, train_qwk = _train_epoch(model, train_loader, optimizer, device)
                val_loss, val_acc, val_qwk = _validate_epoch(model, val_loader, device)
                if scheduler is not None:
                    scheduler.step(epoch + 1)
                current_lr = optimizer.param_groups[0]["lr"]

                writer.add_scalar(f"{trait_name}/train_loss", train_loss, epoch)
                writer.add_scalar(f"{trait_name}/train_acc", train_acc, epoch)
                writer.add_scalar(f"{trait_name}/train_qwk", train_qwk, epoch)
                writer.add_scalar(f"{trait_name}/val_loss", val_loss, epoch)
                writer.add_scalar(f"{trait_name}/val_acc", val_acc, epoch)
                writer.add_scalar(f"{trait_name}/val_qwk", val_qwk, epoch)
                writer.add_scalar(f"{trait_name}/lr", current_lr, epoch)

                _log_wandb(
                    wandb_run,
                    {
                        "epoch": epoch + 1,
                        "train/loss": train_loss,
                        "train/acc": train_acc,
                        "train/qwk": train_qwk,
                        "val/loss": val_loss,
                        "val/acc": val_acc,
                        "val/qwk": val_qwk,
                        "lr": current_lr,
                    },
                )

                if print_epoch_metrics:
                    print(
                        f"Epoch {epoch + 1}/{epochs} - "
                        f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, train_qwk={train_qwk:.4f}, "
                        f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, val_qwk={val_qwk:.4f}, lr={current_lr:.6g}"
                    )

                if val_qwk > best_val_qwk:
                    best_val_qwk = val_qwk
                    best_epoch = epoch + 1

                stop = stopper.step(val_qwk, model)
                if auto_stop and stop.early_stop:
                    print(f"Early stopping triggered for '{trait_name}'.")
                    break

            if not checkpoint_path.exists():
                raise ConfigError(
                    f"Checkpoint for trait '{trait_name}' was not created: {checkpoint_path}"
                )

            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            test_loss, test_acc, test_qwk = _validate_epoch(model, test_loader, device)

            model_name_out = build_checkpoint_name(args.dataset, "trait_model", trait_name)
            model_output_path = output_dir / model_name_out
            torch.save(model.state_dict(), model_output_path)

            final_results[trait_name] = {
                "best_val_qwk": best_val_qwk,
                "best_epoch": best_epoch,
                "test_qwk": test_qwk,
                "test_acc": test_acc,
                "test_loss": test_loss,
                "labels": unique_label_count,
            }

            _log_wandb(
                wandb_run,
                {
                    "best/val_qwk": best_val_qwk,
                    "best/epoch": best_epoch,
                    "test/qwk": test_qwk,
                    "test/acc": test_acc,
                    "test/loss": test_loss,
                },
            )

            print(
                f"Completed '{trait_name}': best_val_qwk={best_val_qwk:.4f} "
                f"(epoch {best_epoch}), test_qwk={test_qwk:.4f}, test_acc={test_acc:.4f}"
            )
        finally:
            writer.close()
            if wandb_run is not None:
                wandb_run.finish()

    if not final_results:
        print("No traits were trained. Check trait names and label distributions.")
        return final_results

    print("\n" + "=" * 100)
    print("FINAL TRAIT PRETRAIN SUMMARY")
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
    print("-" * 100)
    print(f"Average Test QWK: {np.mean(test_qwk_values):.4f}")
    print(f"Best Test QWK:    {np.max(test_qwk_values):.4f}")
    print(f"Worst Test QWK:   {np.min(test_qwk_values):.4f}")

    return final_results
