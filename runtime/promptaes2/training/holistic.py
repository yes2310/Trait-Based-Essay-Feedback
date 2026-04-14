from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm.auto import tqdm

from promptaes2.config import ConfigError, get_dataset_preset, validate_required_columns
from promptaes2.data.alignment import align_npz_and_csv, extract_trait_embeddings
from promptaes2.data.datasets import MultiEmbeddingDataset
from promptaes2.losses.combined import CombinedLoss
from promptaes2.models.factory import build_scoring_model
from promptaes2.utils.checkpoint import EarlyStopping, build_checkpoint_name
from promptaes2.utils.imbalance import (
    build_class_weight_tensor,
    build_weighted_sampler,
    format_class_weight_summary,
)
from promptaes2.utils.metrics import calculate_accuracy_qwk


class _HolisticTextDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        labels: np.ndarray,
        contrastive_scores: np.ndarray | None = None,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.contrastive_scores = contrastive_scores

        if len(self.texts) != len(self.labels):
            raise ValueError(f"texts and labels length mismatch: {len(self.texts)} vs {len(self.labels)}")
        if self.contrastive_scores is not None and len(self.contrastive_scores) != len(self.labels):
            raise ValueError(
                "contrastive_scores and labels length mismatch: "
                f"{len(self.contrastive_scores)} vs {len(self.labels)}"
            )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        if self.contrastive_scores is None:
            return self.texts[idx], self.labels[idx]
        return self.texts[idx], self.labels[idx], self.contrastive_scores[idx]


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


def _resolve_embedding_anchor(embeddings_dict: dict[str, np.ndarray]) -> str:
    if "content" in embeddings_dict:
        return "content"
    return sorted(embeddings_dict.keys())[0]


def _infer_num_labels_from_state(state: dict[str, torch.Tensor], checkpoint_path: Path) -> int:
    preferred_keys = ("classifier.out_proj.weight", "classifier.weight")
    for key in preferred_keys:
        tensor = state.get(key)
        if isinstance(tensor, torch.Tensor) and tensor.ndim == 2:
            return int(tensor.size(0))

    for key, tensor in state.items():
        if "classifier" in key and key.endswith("weight") and isinstance(tensor, torch.Tensor) and tensor.ndim == 2:
            return int(tensor.size(0))

    raise ConfigError(
        f"Failed to infer num_labels from checkpoint '{checkpoint_path}'. "
        "Expected a classifier weight tensor in state dict."
    )


def _resolve_trait_checkpoint_path(dataset: str, trait_name: str, checkpoint_dir: Path) -> Path:
    candidates = [
        checkpoint_dir / build_checkpoint_name(dataset, "trait_model", trait_name),
        checkpoint_dir / build_checkpoint_name(dataset, "trait", trait_name),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    candidate_text = ", ".join(str(path) for path in candidates)
    raise ConfigError(
        f"Trait checkpoint for '{trait_name}' not found. Checked: {candidate_text}"
    )


def _state_dict_to_cpu(module: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _load_trait_backbones_for_e2e(
    args,
    *,
    trait_names: list[str],
    device: torch.device,
):
    trait_checkpoint_dir = getattr(args, "trait_checkpoint_dir", None)
    if trait_checkpoint_dir is None:
        raise ConfigError("Trait checkpoint directory is required for --backbone_mode e2e.")

    checkpoint_dir = Path(trait_checkpoint_dir)
    if not checkpoint_dir.exists():
        raise ConfigError(f"Trait checkpoint directory does not exist: {checkpoint_dir}")

    preset = get_dataset_preset(args.dataset)
    model_name = args.trait_model_name if args.trait_model_name else preset.default_model_name

    try:
        from transformers import AutoTokenizer, RobertaConfig, RobertaForSequenceClassification
    except ImportError as exc:
        raise RuntimeError(
            "Transformers is required to run end-to-end holistic training."
        ) from exc

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to load tokenizer '{model_name}' for end-to-end holistic training."
        ) from exc

    trait_models: dict[str, nn.Module] = {}
    for trait_name in trait_names:
        checkpoint_path = _resolve_trait_checkpoint_path(args.dataset, trait_name, checkpoint_dir)
        loaded = torch.load(checkpoint_path, map_location="cpu")
        state = loaded.get("model_state_dict", loaded) if isinstance(loaded, dict) else loaded
        if not isinstance(state, dict):
            raise ConfigError(f"Invalid checkpoint format: {checkpoint_path}")
        if any(key.startswith("module.") for key in state):
            state = {key.removeprefix("module."): value for key, value in state.items()}

        num_labels = _infer_num_labels_from_state(state, checkpoint_path)
        try:
            config = RobertaConfig.from_pretrained(model_name, num_labels=num_labels)
            trait_model = RobertaForSequenceClassification(config)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to initialize model '{model_name}' for trait '{trait_name}'."
            ) from exc

        missing_keys, unexpected_keys = trait_model.load_state_dict(state, strict=False)
        allowed_missing = {"roberta.embeddings.position_ids"}
        filtered_missing = [key for key in missing_keys if key not in allowed_missing]
        if filtered_missing or unexpected_keys:
            missing_text = ", ".join(filtered_missing) if filtered_missing else "none"
            unexpected_text = ", ".join(unexpected_keys) if unexpected_keys else "none"
            raise ConfigError(
                f"Checkpoint/model key mismatch for trait '{trait_name}'. "
                f"missing=[{missing_text}], unexpected=[{unexpected_text}]"
            )

        trait_model.to(device)
        trait_models[trait_name] = trait_model

    return tokenizer, trait_models


def _extract_embeddings_from_trait_checkpoints(
    args,
    trait_names: list[str],
    device: torch.device,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    trait_checkpoint_dir = getattr(args, "trait_checkpoint_dir", None)
    if trait_checkpoint_dir is None:
        raise ConfigError("Trait checkpoint directory is required when --npz_path is not provided.")

    checkpoint_dir = Path(trait_checkpoint_dir)
    if not checkpoint_dir.exists():
        raise ConfigError(f"Trait checkpoint directory does not exist: {checkpoint_dir}")

    essay = pd.read_csv(args.csv_path)
    validate_required_columns(essay.columns, ["text", "total_score"])
    texts = essay["text"].fillna("").astype(str).tolist()
    if not texts:
        raise ConfigError(f"No rows found in CSV: {args.csv_path}")

    preset = get_dataset_preset(args.dataset)
    model_name = args.trait_model_name if args.trait_model_name else preset.default_model_name
    max_seq_length = int(getattr(args, "embedding_max_seq_length", 512))
    embedding_batch_size = int(getattr(args, "embedding_batch_size", 32))

    try:
        from transformers import AutoTokenizer, RobertaConfig, RobertaForSequenceClassification
    except ImportError as exc:
        raise RuntimeError(
            "Transformers is required to extract embeddings from trait checkpoints."
        ) from exc

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to load tokenizer '{model_name}' for trait checkpoint embedding extraction."
        ) from exc

    text_batches = [
        texts[idx : idx + embedding_batch_size]
        for idx in range(0, len(texts), embedding_batch_size)
    ]

    print("NPZ not provided. Generating trait embeddings from trait checkpoints.")
    print(f"Trait checkpoint dir: {checkpoint_dir}")
    print(f"Trait backbone model: {model_name}")

    embeddings_dict: dict[str, np.ndarray] = {}
    for trait_name in trait_names:
        checkpoint_path = _resolve_trait_checkpoint_path(args.dataset, trait_name, checkpoint_dir)
        loaded = torch.load(checkpoint_path, map_location="cpu")
        state = loaded.get("model_state_dict", loaded) if isinstance(loaded, dict) else loaded
        if not isinstance(state, dict):
            raise ConfigError(f"Invalid checkpoint format: {checkpoint_path}")
        if any(key.startswith("module.") for key in state):
            state = {key.removeprefix("module."): value for key, value in state.items()}

        num_labels = _infer_num_labels_from_state(state, checkpoint_path)
        try:
            config = RobertaConfig.from_pretrained(model_name, num_labels=num_labels)
            model = RobertaForSequenceClassification(config)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to initialize model '{model_name}' for trait '{trait_name}'."
            ) from exc

        missing_keys, unexpected_keys = model.load_state_dict(state, strict=False)
        allowed_missing = {"roberta.embeddings.position_ids"}
        filtered_missing = [key for key in missing_keys if key not in allowed_missing]
        if filtered_missing or unexpected_keys:
            missing_text = ", ".join(filtered_missing) if filtered_missing else "none"
            unexpected_text = ", ".join(unexpected_keys) if unexpected_keys else "none"
            raise ConfigError(
                f"Checkpoint/model key mismatch for trait '{trait_name}'. "
                f"missing=[{missing_text}], unexpected=[{unexpected_text}]"
            )

        model.to(device)
        model.eval()

        trait_embeddings: list[np.ndarray] = []
        with torch.no_grad():
            for text_batch in text_batches:
                encoded = tokenizer(
                    text_batch,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=max_seq_length,
                )
                outputs = model.roberta(
                    input_ids=encoded["input_ids"].to(device),
                    attention_mask=encoded["attention_mask"].to(device),
                    return_dict=True,
                )
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy().astype(np.float32)
                trait_embeddings.append(cls_embeddings)

        embeddings_dict[trait_name] = np.concatenate(trait_embeddings, axis=0)

    return embeddings_dict, essay


def _validate_embedding_dimensions(
    embeddings_dict: dict[str, np.ndarray],
    expected_dim: int,
) -> None:
    dims = {
        trait_name: int(embedding.shape[1])
        for trait_name, embedding in embeddings_dict.items()
    }
    unique_dims = sorted(set(dims.values()))
    if len(unique_dims) != 1:
        detail = ", ".join(f"{name}:{dim}" for name, dim in sorted(dims.items()))
        raise ConfigError(f"Inconsistent embedding dimensions detected: {detail}")

    actual_dim = unique_dims[0]
    if actual_dim != expected_dim:
        raise ConfigError(
            f"Embedding dimension mismatch: got {actual_dim}, expected {expected_dim}. "
            "Use --embedding_dim to match your trait embedding size."
        )


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


def _print_split_distributions(
    train_labels: np.ndarray,
    val_labels: np.ndarray,
    test_labels: np.ndarray,
) -> None:
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
    essay: pd.DataFrame,
    split_column: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normalized_splits = essay[split_column].map(_normalize_predefined_split_value)
    invalid_mask = normalized_splits.isna()
    if invalid_mask.any():
        invalid_values = sorted(
            {
                "NaN" if pd.isna(raw_value) else str(raw_value)
                for raw_value in essay.loc[invalid_mask, split_column]
            }
        )
        invalid_text = ", ".join(invalid_values)
        raise ConfigError(
            f"Predefined split column '{split_column}' contains invalid values: {invalid_text}. "
            "Expected train/dev/test (or val/valid/validation)."
        )

    split_values = normalized_splits.to_numpy(dtype=object)
    train_idx = np.flatnonzero(split_values == "train")
    val_idx = np.flatnonzero(split_values == "val")
    test_idx = np.flatnonzero(split_values == "test")

    missing_splits = [
        split_name
        for split_name, split_indices in (("train", train_idx), ("val", val_idx), ("test", test_idx))
        if len(split_indices) == 0
    ]
    if missing_splits:
        missing_text = ", ".join(missing_splits)
        raise ConfigError(
            f"Predefined split column '{split_column}' is missing {missing_text} rows."
        )

    return train_idx, val_idx, test_idx


def _maybe_init_wandb(args, run_name_override: str | None = None):
    if not args.use_wandb:
        return None

    moe_aux_weight = float(getattr(args, "moe_aux_weight", 0.01))

    try:
        import wandb  # pylint: disable=import-outside-toplevel
    except ImportError:
        print("wandb is not installed. Continuing without wandb logging.")
        return None

    run = wandb.init(
        project=args.wandb_project_name,
        name=run_name_override if run_name_override is not None else args.run_name,
        config={
            "dataset": args.dataset,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "loss_type": args.loss_type,
            "hidden_sizes": args.hidden_sizes,
            "dropout_rates": args.dropout_rates,
            "scheduler": args.scheduler,
            "evolution_stage": args.evolution_stage,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "predefined_split_column": getattr(args, "predefined_split_column", None),
            "imbalance_mitigation": bool(getattr(args, "imbalance_mitigation", False)),
            "imbalance_max_weight": float(getattr(args, "imbalance_max_weight", 5.0)),
            "epochs": args.epochs,
            "moe_aux_weight": moe_aux_weight,
            "model_variant": getattr(args, "model_variant", "legacy"),
            "group_num_experts": int(getattr(args, "group_num_experts", 8)),
            "classifier_num_experts": int(getattr(args, "classifier_num_experts", 8)),
            "router_top_k": int(getattr(args, "router_top_k", 2)),
        },
    )
    return run


def _log_wandb(run, payload: dict):
    if run is not None:
        run.log(payload)


def _sanitize_for_path(value: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z._-]+", "_", value).strip("._-")
    if not normalized:
        return "value"
    return normalized[:64]


def _format_trait_groups_for_log(trait_groups) -> str:
    parts: list[str] = []
    for trait_names, dim in trait_groups:
        joined = ",".join(trait_names)
        parts.append(f"{joined}:{dim}")
    return "; ".join(parts)


def _print_holistic_startup_info(args) -> None:
    csv_path = Path(args.csv_path)
    npz_path = getattr(args, "npz_path", None)
    trait_checkpoint_dir = getattr(args, "trait_checkpoint_dir", None)
    split_by_column = getattr(args, "split_by_column", None)
    predefined_split_column = getattr(args, "predefined_split_column", None)

    print("\n" + "=" * 100, flush=True)
    print("HOLISTIC TRAINING STARTUP", flush=True)
    print("=" * 100, flush=True)
    print(f"Dataset: {args.dataset}", flush=True)
    print(f"CSV path: {csv_path}", flush=True)
    print(f"CSV exists: {csv_path.exists()}", flush=True)
    print(f"Trait groups: {_format_trait_groups_for_log(args.trait_groups)}", flush=True)
    print(
        "Input mode: "
        + (
            f"npz ({npz_path})"
            if npz_path is not None
            else f"trait checkpoints ({trait_checkpoint_dir})"
        ),
        flush=True,
    )
    if predefined_split_column is None:
        print(
            f"Split ratios: train={args.train_ratio:.2f}, val={args.val_ratio:.2f}, test={args.test_ratio:.2f}",
            flush=True,
        )
    else:
        print(f"Predefined split column: {predefined_split_column}", flush=True)
    print(f"Split by column: {split_by_column if split_by_column is not None else 'none'}", flush=True)
    print(f"Ablation mode: {getattr(args, 'ablation_mode', 'none')}", flush=True)
    print(f"Backbone mode: {getattr(args, 'backbone_mode', 'frozen')}", flush=True)
    if getattr(args, "backbone_mode", "frozen") == "e2e":
        print(
            f"Backbone lr: {float(getattr(args, 'backbone_lr', 1e-5))} "
            f"(unfreeze_epoch={int(getattr(args, 'unfreeze_epoch', 0))})",
            flush=True,
        )
    print(
        f"Evolution stage: {args.evolution_stage} (warmup_epochs={args.warmup_epochs})",
        flush=True,
    )
    print(
        "Imbalance mitigation: "
        f"{bool(getattr(args, 'imbalance_mitigation', False))} "
        f"(max_weight={float(getattr(args, 'imbalance_max_weight', 5.0)):.2f})",
        flush=True,
    )
    print(
        "Pre-skip toggles: "
        f"pre_homo={bool(getattr(args, 'use_pre_homo_skip', False))}, "
        f"pre_hetero={bool(getattr(args, 'use_pre_hetero_skip', False))}",
        flush=True,
    )
    model_variant = str(getattr(args, "model_variant", "legacy"))
    print(f"Model variant: {model_variant}", flush=True)
    if model_variant == "canonical_moe":
        print(
            "Canonical MoE settings: "
            f"group_num_experts={int(getattr(args, 'group_num_experts', 8))}, "
            f"classifier_num_experts={int(getattr(args, 'classifier_num_experts', 8))}, "
            f"router_top_k={int(getattr(args, 'router_top_k', 2))}",
            flush=True,
        )
    print(f"Save checkpoints: {bool(getattr(args, 'save_checkpoints', True))}", flush=True)
    print(f"Output dir: {args.output_dir}", flush=True)

    if csv_path.exists():
        try:
            columns = pd.read_csv(csv_path, nrows=0).columns.astype(str).tolist()
            columns_preview = ", ".join(columns[:20])
            if len(columns) > 20:
                columns_preview = f"{columns_preview}, ..."
            print(f"CSV columns ({len(columns)}): {columns_preview}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"CSV header preview failed: {exc}", flush=True)

    print("-" * 100, flush=True)


_WIDE_RANGE_PROMPTS = {1, 7, 8}


def _normalize_prompt_value(value) -> int | str | None:
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if text.startswith("prompt"):
        text = text.replace("prompt", "", 1).strip()
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return text


def _to_six_bins(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores.astype(np.float32)
    min_score = float(np.min(scores))
    max_score = float(np.max(scores))
    if max_score <= min_score:
        return np.ones_like(scores, dtype=np.float32)
    edges = np.linspace(min_score, max_score, num=7)
    return (np.digitize(scores, edges[1:-1], right=True) + 1).astype(np.float32)


def _should_apply_score_binning(
    scores: np.ndarray,
    *,
    binning_mode: str,
    prompt_value: int | str | None = None,
) -> bool:
    if binning_mode == "none":
        return False
    if binning_mode == "force6":
        return True

    normalized_prompt = _normalize_prompt_value(prompt_value)
    if isinstance(normalized_prompt, int) and normalized_prompt in _WIDE_RANGE_PROMPTS:
        return True

    score_span = float(np.max(scores) - np.min(scores))
    return score_span >= 10.0


def _build_contrastive_scores(
    essay: pd.DataFrame,
    *,
    binning_mode: str,
) -> tuple[np.ndarray, list[str]]:
    raw_scores = essay["total_score"].to_numpy(dtype=np.float32, copy=True)
    converted_scores = raw_scores.copy()
    applied_messages: list[str] = []

    if "prompt" in essay.columns:
        prompt_values = essay["prompt"].to_numpy()
        for prompt_value in pd.unique(prompt_values):
            mask = prompt_values == prompt_value
            prompt_scores = raw_scores[mask]
            if prompt_scores.size == 0:
                continue
            if _should_apply_score_binning(
                prompt_scores,
                binning_mode=binning_mode,
                prompt_value=prompt_value,
            ):
                converted_scores[mask] = _to_six_bins(prompt_scores)
                prompt_label = "NaN" if pd.isna(prompt_value) else str(prompt_value)
                applied_messages.append(
                    f"Applied 1~6 contrastive score binning for prompt={prompt_label} "
                    f"(range={float(np.min(prompt_scores)):.1f}~{float(np.max(prompt_scores)):.1f})."
                )
    elif _should_apply_score_binning(raw_scores, binning_mode=binning_mode):
        converted_scores = _to_six_bins(raw_scores)
        applied_messages.append(
            "Applied 1~6 contrastive score binning for entire split "
            f"(range={float(np.min(raw_scores)):.1f}~{float(np.max(raw_scores)):.1f})."
        )

    return converted_scores.astype(np.float32), applied_messages


def _run_single_holistic_training(
    args,
    *,
    embeddings_dict: dict[str, np.ndarray],
    essay: pd.DataFrame,
    output_dir: Path,
    run_label: str | None = None,
):
    _set_seed(args.seed)
    moe_aux_weight = float(getattr(args, "moe_aux_weight", 0.01))

    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))

    if "total_score" not in essay.columns:
        available = ", ".join(essay.columns.astype(str).tolist())
        raise ConfigError(
            f"CSV must contain 'total_score' column. Available columns: {available}"
        )

    labels_raw = essay["total_score"].values
    unique_labels = np.unique(labels_raw)
    label_mapping = {old: new for new, old in enumerate(sorted(unique_labels))}
    labels = np.array([label_mapping[label] for label in labels_raw])
    contrastive_scores, binning_messages = _build_contrastive_scores(
        essay,
        binning_mode=str(getattr(args, "cl_score_binning", "auto")),
    )

    unique_labels_count = len(unique_labels)
    predefined_split_column = getattr(args, "predefined_split_column", None)
    if predefined_split_column is not None:
        train_idx, val_idx, test_idx = _build_predefined_split_indices(essay, predefined_split_column)
    else:
        indices = np.arange(len(labels))
        holdout_ratio = float(args.val_ratio + args.test_ratio)
        test_ratio_within_holdout = float(args.test_ratio / holdout_ratio)

        try:
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
            raise ConfigError(
                f"Stratified split failed ({exc}). "
                f"Label distribution: {_format_label_distribution(labels_raw)}. "
                "Ensure each label has enough samples for train/val/test."
            ) from exc

    wandb_run_name = None
    if run_label is not None and getattr(args, "run_name", None):
        wandb_run_name = f"{args.run_name}-{_sanitize_for_path(run_label)}"
    run = _maybe_init_wandb(args, run_name_override=wandb_run_name)
    anchor_trait = _resolve_embedding_anchor(embeddings_dict)

    if run_label is not None:
        print(f"Run label: {run_label}")
    print(f"Dataset: {args.dataset}")
    print(f"Aligned samples: {len(labels)}")
    print(f"Unique labels: {unique_labels_count}")
    if predefined_split_column is None:
        print(
            "Split ratios: "
            f"train={args.train_ratio:.2f}, val={args.val_ratio:.2f}, test={args.test_ratio:.2f}"
        )
    else:
        print(f"Predefined split column: {predefined_split_column}")
    print(f"Anchor trait for CombinedLoss: {anchor_trait}")
    print(f"MoE auxiliary loss weight: {moe_aux_weight}")
    for message in binning_messages:
        print(message)

    _print_split_distributions(labels[train_idx], labels[val_idx], labels[test_idx])

    train_embeddings = {trait: emb[train_idx] for trait, emb in embeddings_dict.items()}
    val_embeddings = {trait: emb[val_idx] for trait, emb in embeddings_dict.items()}
    test_embeddings = {trait: emb[test_idx] for trait, emb in embeddings_dict.items()}

    imbalance_mitigation = bool(getattr(args, "imbalance_mitigation", False))
    imbalance_max_weight = float(getattr(args, "imbalance_max_weight", 5.0))
    class_weights: torch.Tensor | None = None
    train_sampler: WeightedRandomSampler | None = None
    if imbalance_mitigation:
        class_weights, counts = build_class_weight_tensor(
            labels[train_idx],
            num_classes=unique_labels_count,
            max_weight=imbalance_max_weight,
            device=device,
        )
        train_sampler, weights_np, counts_np = build_weighted_sampler(
            labels[train_idx],
            num_classes=unique_labels_count,
            max_weight=imbalance_max_weight,
        )
        class_names = [str(label) for label in sorted(unique_labels)]
        print(
            "Train class weights: "
            + format_class_weight_summary(counts_np, weights_np, class_labels=class_names)
        )

    train_dataset = MultiEmbeddingDataset(
        train_embeddings,
        labels[train_idx],
        contrastive_scores=contrastive_scores[train_idx] if args.loss_type == "combined" else None,
    )
    val_dataset = MultiEmbeddingDataset(
        val_embeddings,
        labels[val_idx],
        contrastive_scores=contrastive_scores[val_idx] if args.loss_type == "combined" else None,
    )
    test_dataset = MultiEmbeddingDataset(
        test_embeddings,
        labels[test_idx],
        contrastive_scores=contrastive_scores[test_idx] if args.loss_type == "combined" else None,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=False,
        num_workers=args.cpu_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.cpu_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.cpu_workers,
    )

    if len(train_loader) == 0:
        raise ConfigError("Train loader is empty. Reduce --batch_size or adjust split configuration.")
    if len(val_loader) == 0:
        raise ConfigError("Validation loader is empty. Reduce --batch_size or adjust split configuration.")
    if len(test_loader) == 0:
        raise ConfigError("Test loader is empty. Reduce --batch_size or adjust split configuration.")

    try:
        model = build_scoring_model(
            model_variant=str(getattr(args, "model_variant", "legacy")),
            embedding_dim=args.embedding_dim,
            hidden_sizes=args.hidden_sizes,
            num_classes=unique_labels_count,
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

    criterion = (
        CombinedLoss(
            lambda1=args.lambda1,
            lambda2=args.lambda2,
            alpha=args.alpha,
            beta1=args.beta1,
            beta2=args.beta2,
            beta3=args.beta3,
            beta4=args.beta4,
            margin=args.margin,
            class_weights=class_weights,
        ).to(device)
        if args.loss_type == "combined"
        else nn.CrossEntropyLoss(weight=class_weights).to(device)
    )
    ce_only_criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)

    def _unpack_holistic_batch(batch):
        if args.loss_type == "combined":
            batch_embeddings, labels_batch, contrastive_scores_batch = batch
            return batch_embeddings, labels_batch, contrastive_scores_batch
        batch_embeddings, labels_batch = batch
        return batch_embeddings, labels_batch, None

    stage1_epochs = int(getattr(args, "combined_stage1_epochs", 1)) if args.loss_type == "combined" else 0
    if args.loss_type == "combined":
        if stage1_epochs > 0:
            print(f"Stage 1/2 (CE-only) epochs: {stage1_epochs}")
            for stage1_epoch in tqdm(range(stage1_epochs), desc="Holistic Stage1"):
                if hasattr(model, "set_epoch"):
                    model.set_epoch(stage1_epoch)

                model.train()
                stage1_total_loss = 0.0
                stage1_main_loss = 0.0
                stage1_aux_loss = 0.0

                for batch in train_loader:
                    optimizer.zero_grad()
                    batch_embeddings, labels_batch, _ = _unpack_holistic_batch(batch)
                    batch_embeddings = {trait: emb.to(device) for trait, emb in batch_embeddings.items()}
                    labels_batch = labels_batch.to(device)

                    output = model(batch_embeddings)
                    main_loss = ce_only_criterion(output, labels_batch)

                    moe_aux_loss = model.moe_aux_loss
                    if moe_aux_loss is None:
                        moe_aux_loss = output.new_tensor(0.0)
                    total_loss = main_loss + (moe_aux_weight * moe_aux_loss)

                    total_loss.backward()
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()

                    stage1_total_loss += float(total_loss.item())
                    stage1_main_loss += float(main_loss.item())
                    stage1_aux_loss += float(moe_aux_loss.item())

                avg_stage1_total = stage1_total_loss / len(train_loader)
                avg_stage1_main = stage1_main_loss / len(train_loader)
                avg_stage1_aux = stage1_aux_loss / len(train_loader)
                _log_wandb(
                    run,
                    {
                        "stage1/train_loss": avg_stage1_total,
                        "stage1/train_main_loss": avg_stage1_main,
                        "stage1/train_moe_aux_loss": avg_stage1_aux,
                        "stage1/epoch": stage1_epoch + 1,
                    },
                )
                print(
                    f"Stage1 Epoch {stage1_epoch + 1}/{stage1_epochs} - "
                    f"train_total={avg_stage1_total:.4f}, train_main={avg_stage1_main:.4f}, "
                    f"train_aux={avg_stage1_aux:.4f}"
                )
        else:
            print("Skipping Stage 1 CE-only fine-tuning (--combined_stage1_epochs=0).")

        print("Initializing contrastive score means from full training split.")
        model.eval()
        init_embeddings: list[torch.Tensor] = []
        init_scores: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in train_loader:
                batch_embeddings, _, contrastive_scores_batch = _unpack_holistic_batch(batch)
                if contrastive_scores_batch is None:
                    raise RuntimeError("Combined loss requires contrastive scores in the dataloader.")
                batch_embeddings = {trait: emb.to(device) for trait, emb in batch_embeddings.items()}
                contrastive_scores_batch = contrastive_scores_batch.to(device).float()
                _, holistic_embeddings = model(batch_embeddings, return_holistic_embedding=True)
                init_embeddings.append(holistic_embeddings.detach())
                init_scores.append(contrastive_scores_batch.detach())

        criterion.update_score_embeddings(
            torch.cat(init_embeddings, dim=0),
            torch.cat(init_scores, dim=0),
            reset=True,
        )

    best_qwk = -np.inf
    best_acc = 0.0
    save_checkpoints = bool(getattr(args, "save_checkpoints", True))
    best_state_dict: dict[str, torch.Tensor] | None = None
    no_improve_epochs = 0

    checkpoint_path = output_dir / build_checkpoint_name(args.dataset, "holistic", "best")
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

    for epoch in tqdm(range(args.epochs), desc="Holistic"):
        if hasattr(model, "set_epoch"):
            model.set_epoch(stage1_epochs + epoch)

        model.train()
        epoch_total_loss = 0.0
        epoch_main_loss = 0.0
        epoch_aux_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            batch_embeddings, labels_batch, contrastive_scores_batch = _unpack_holistic_batch(batch)
            batch_embeddings = {trait: emb.to(device) for trait, emb in batch_embeddings.items()}
            labels_batch = labels_batch.to(device)

            if args.loss_type == "combined":
                if contrastive_scores_batch is None:
                    raise RuntimeError("Combined loss requires contrastive scores in the dataloader.")
                contrastive_scores_batch = contrastive_scores_batch.to(device).float()
                output, holistic_embeddings = model(batch_embeddings, return_holistic_embedding=True)
                main_loss = criterion(
                    output,
                    labels_batch,
                    holistic_embeddings,
                    contrastive_scores_batch,
                )
            else:
                output = model(batch_embeddings)
                main_loss = criterion(output, labels_batch)

            moe_aux_loss = model.moe_aux_loss
            if moe_aux_loss is None:
                moe_aux_loss = output.new_tensor(0.0)
            total_loss = main_loss + (moe_aux_weight * moe_aux_loss)

            total_loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            epoch_total_loss += float(total_loss.item())
            epoch_main_loss += float(main_loss.item())
            epoch_aux_loss += float(moe_aux_loss.item())

        avg_train_total_loss = epoch_total_loss / len(train_loader)
        avg_train_main_loss = epoch_main_loss / len(train_loader)
        avg_train_aux_loss = epoch_aux_loss / len(train_loader)

        model.eval()
        val_predictions: list[int] = []
        val_labels: list[int] = []
        val_total_loss = 0.0
        val_main_loss = 0.0
        val_aux_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                batch_embeddings, labels_batch, contrastive_scores_batch = _unpack_holistic_batch(batch)
                batch_embeddings = {trait: emb.to(device) for trait, emb in batch_embeddings.items()}
                labels_batch = labels_batch.to(device)

                if args.loss_type == "combined":
                    if contrastive_scores_batch is None:
                        raise RuntimeError("Combined loss requires contrastive scores in the dataloader.")
                    contrastive_scores_batch = contrastive_scores_batch.to(device).float()
                    output, holistic_embeddings = model(batch_embeddings, return_holistic_embedding=True)
                    main_loss = criterion(
                        output,
                        labels_batch,
                        holistic_embeddings,
                        contrastive_scores_batch,
                    )
                else:
                    output = model(batch_embeddings)
                    main_loss = criterion(output, labels_batch)

                moe_aux_loss = model.moe_aux_loss
                if moe_aux_loss is None:
                    moe_aux_loss = output.new_tensor(0.0)
                total_loss = main_loss + (moe_aux_weight * moe_aux_loss)

                val_total_loss += float(total_loss.item())
                val_main_loss += float(main_loss.item())
                val_aux_loss += float(moe_aux_loss.item())
                val_predictions.extend(torch.argmax(output, dim=1).cpu().numpy().tolist())
                val_labels.extend(labels_batch.cpu().numpy().tolist())

        avg_val_total_loss = val_total_loss / len(val_loader)
        avg_val_main_loss = val_main_loss / len(val_loader)
        avg_val_aux_loss = val_aux_loss / len(val_loader)
        val_accuracy, val_qwk = calculate_accuracy_qwk(val_labels, val_predictions)

        _log_wandb(
            run,
            {
                "train_loss": avg_train_total_loss,
                "val_loss": avg_val_total_loss,
                "train_main_loss": avg_train_main_loss,
                "train_moe_aux_loss": avg_train_aux_loss,
                "val_main_loss": avg_val_main_loss,
                "val_moe_aux_loss": avg_val_aux_loss,
                "val_accuracy": val_accuracy,
                "val_qwk": val_qwk,
                "epoch": epoch,
            },
        )

        print(
            f"Epoch {epoch + 1}/{args.epochs} - "
            f"train_total={avg_train_total_loss:.4f}, train_main={avg_train_main_loss:.4f}, "
            f"train_aux={avg_train_aux_loss:.4f}, "
            f"val_total={avg_val_total_loss:.4f}, val_main={avg_val_main_loss:.4f}, "
            f"val_aux={avg_val_aux_loss:.4f}, "
            f"val_acc={val_accuracy:.4f}, val_qwk={val_qwk:.4f}"
        )

        if val_qwk > best_qwk:
            best_qwk = val_qwk
            best_acc = val_accuracy
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
                print("Early stopping triggered.")
                break
        elif no_improve_epochs >= args.patience:
            print("Early stopping triggered (in-memory best model, checkpoint saving disabled).")
            break

    if save_checkpoints and checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    elif (not save_checkpoints) and best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    model.eval()
    test_predictions: list[int] = []
    test_labels: list[int] = []
    test_total_loss = 0.0
    test_main_loss = 0.0
    test_aux_loss = 0.0

    with torch.no_grad():
        for batch in test_loader:
            batch_embeddings, labels_batch, contrastive_scores_batch = _unpack_holistic_batch(batch)
            batch_embeddings = {trait: emb.to(device) for trait, emb in batch_embeddings.items()}
            labels_batch = labels_batch.to(device)

            if args.loss_type == "combined":
                if contrastive_scores_batch is None:
                    raise RuntimeError("Combined loss requires contrastive scores in the dataloader.")
                contrastive_scores_batch = contrastive_scores_batch.to(device).float()
                output, holistic_embeddings = model(batch_embeddings, return_holistic_embedding=True)
                main_loss = criterion(
                    output,
                    labels_batch,
                    holistic_embeddings,
                    contrastive_scores_batch,
                )
            else:
                output = model(batch_embeddings)
                main_loss = criterion(output, labels_batch)

            moe_aux_loss = model.moe_aux_loss
            if moe_aux_loss is None:
                moe_aux_loss = output.new_tensor(0.0)
            total_loss = main_loss + (moe_aux_weight * moe_aux_loss)

            test_total_loss += float(total_loss.item())
            test_main_loss += float(main_loss.item())
            test_aux_loss += float(moe_aux_loss.item())
            test_predictions.extend(torch.argmax(output, dim=1).cpu().numpy().tolist())
            test_labels.extend(labels_batch.cpu().numpy().tolist())

    avg_test_total_loss = test_total_loss / len(test_loader)
    avg_test_main_loss = test_main_loss / len(test_loader)
    avg_test_aux_loss = test_aux_loss / len(test_loader)
    test_accuracy, test_qwk = calculate_accuracy_qwk(test_labels, test_predictions)

    _log_wandb(
        run,
        {
            "best_val_qwk": best_qwk,
            "best_val_accuracy": best_acc,
            "test_loss": avg_test_total_loss,
            "test_main_loss": avg_test_main_loss,
            "test_moe_aux_loss": avg_test_aux_loss,
            "test_qwk": test_qwk,
            "test_accuracy": test_accuracy,
        },
    )

    if run is not None:
        run.finish()

    print("=" * 40)
    print(f"best val QWK: {best_qwk:.4f}")
    print(f"best val ACC: {best_acc:.4f}")
    print(f"test QWK: {test_qwk:.4f}")
    print(f"test ACC: {test_accuracy:.4f}")

    return {
        "best_val_qwk": float(best_qwk),
        "best_val_accuracy": float(best_acc),
        "test_qwk": float(test_qwk),
        "test_accuracy": float(test_accuracy),
        "test_loss": float(avg_test_total_loss),
        "test_main_loss": float(avg_test_main_loss),
        "test_moe_aux_loss": float(avg_test_aux_loss),
        "split_sizes": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "checkpoint_path": str(checkpoint_path) if save_checkpoints else None,
        "checkpoints_saved": save_checkpoints,
    }


def _run_single_holistic_training_e2e(
    args,
    *,
    essay: pd.DataFrame,
    output_dir: Path,
    trait_names: list[str],
    run_label: str | None = None,
):
    _set_seed(args.seed)
    moe_aux_weight = float(getattr(args, "moe_aux_weight", 0.01))

    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))

    validate_required_columns(essay.columns, ["text", "total_score"])
    texts_all = essay["text"].fillna("").astype(str).tolist()
    if not texts_all:
        raise ConfigError("No text rows found for end-to-end holistic training.")

    labels_raw = essay["total_score"].values
    unique_labels = np.unique(labels_raw)
    label_mapping = {old: new for new, old in enumerate(sorted(unique_labels))}
    labels = np.array([label_mapping[label] for label in labels_raw])
    contrastive_scores, binning_messages = _build_contrastive_scores(
        essay,
        binning_mode=str(getattr(args, "cl_score_binning", "auto")),
    )

    unique_labels_count = len(unique_labels)
    predefined_split_column = getattr(args, "predefined_split_column", None)
    if predefined_split_column is not None:
        train_idx, val_idx, test_idx = _build_predefined_split_indices(essay, predefined_split_column)
    else:
        indices = np.arange(len(labels))
        holdout_ratio = float(args.val_ratio + args.test_ratio)
        test_ratio_within_holdout = float(args.test_ratio / holdout_ratio)

        try:
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
            raise ConfigError(
                f"Stratified split failed ({exc}). "
                f"Label distribution: {_format_label_distribution(labels_raw)}. "
                "Ensure each label has enough samples for train/val/test."
            ) from exc

    wandb_run_name = None
    if run_label is not None and getattr(args, "run_name", None):
        wandb_run_name = f"{args.run_name}-{_sanitize_for_path(run_label)}"
    run = _maybe_init_wandb(args, run_name_override=wandb_run_name)

    if run_label is not None:
        print(f"Run label: {run_label}")
    print(f"Dataset: {args.dataset}")
    print(f"Aligned samples: {len(labels)}")
    print(f"Unique labels: {unique_labels_count}")
    print(f"Backbone mode: e2e (lr={float(args.backbone_lr)}, unfreeze_epoch={int(args.unfreeze_epoch)})")
    if predefined_split_column is None:
        print(
            "Split ratios: "
            f"train={args.train_ratio:.2f}, val={args.val_ratio:.2f}, test={args.test_ratio:.2f}"
        )
    else:
        print(f"Predefined split column: {predefined_split_column}")
    print(f"MoE auxiliary loss weight: {moe_aux_weight}")
    for message in binning_messages:
        print(message)

    _print_split_distributions(labels[train_idx], labels[val_idx], labels[test_idx])

    train_texts = [texts_all[idx] for idx in train_idx]
    val_texts = [texts_all[idx] for idx in val_idx]
    test_texts = [texts_all[idx] for idx in test_idx]

    imbalance_mitigation = bool(getattr(args, "imbalance_mitigation", False))
    imbalance_max_weight = float(getattr(args, "imbalance_max_weight", 5.0))
    class_weights: torch.Tensor | None = None
    train_sampler: WeightedRandomSampler | None = None
    if imbalance_mitigation:
        class_weights, counts = build_class_weight_tensor(
            labels[train_idx],
            num_classes=unique_labels_count,
            max_weight=imbalance_max_weight,
            device=device,
        )
        train_sampler, weights_np, counts_np = build_weighted_sampler(
            labels[train_idx],
            num_classes=unique_labels_count,
            max_weight=imbalance_max_weight,
        )
        class_names = [str(label) for label in sorted(unique_labels)]
        print(
            "Train class weights: "
            + format_class_weight_summary(counts_np, weights_np, class_labels=class_names)
        )

    train_dataset = _HolisticTextDataset(
        train_texts,
        labels[train_idx],
        contrastive_scores=contrastive_scores[train_idx] if args.loss_type == "combined" else None,
    )
    val_dataset = _HolisticTextDataset(
        val_texts,
        labels[val_idx],
        contrastive_scores=contrastive_scores[val_idx] if args.loss_type == "combined" else None,
    )
    test_dataset = _HolisticTextDataset(
        test_texts,
        labels[test_idx],
        contrastive_scores=contrastive_scores[test_idx] if args.loss_type == "combined" else None,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=False,
        num_workers=args.cpu_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.cpu_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.cpu_workers,
    )

    if len(train_loader) == 0:
        raise ConfigError("Train loader is empty. Reduce --batch_size or adjust split configuration.")
    if len(val_loader) == 0:
        raise ConfigError("Validation loader is empty. Reduce --batch_size or adjust split configuration.")
    if len(test_loader) == 0:
        raise ConfigError("Test loader is empty. Reduce --batch_size or adjust split configuration.")

    tokenizer, trait_backbones = _load_trait_backbones_for_e2e(
        args,
        trait_names=trait_names,
        device=device,
    )

    try:
        model = build_scoring_model(
            model_variant=str(getattr(args, "model_variant", "legacy")),
            embedding_dim=args.embedding_dim,
            hidden_sizes=args.hidden_sizes,
            num_classes=unique_labels_count,
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

    backbone_params: list[nn.Parameter] = []
    for trait_backbone in trait_backbones.values():
        backbone_params.extend(list(trait_backbone.roberta.parameters()))

    optimizer = torch.optim.RAdam(
        [
            {"params": model.parameters(), "lr": args.learning_rate},
            {"params": backbone_params, "lr": args.backbone_lr},
        ]
    )
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

    criterion = (
        CombinedLoss(
            lambda1=args.lambda1,
            lambda2=args.lambda2,
            alpha=args.alpha,
            beta1=args.beta1,
            beta2=args.beta2,
            beta3=args.beta3,
            beta4=args.beta4,
            margin=args.margin,
            class_weights=class_weights,
        ).to(device)
        if args.loss_type == "combined"
        else nn.CrossEntropyLoss(weight=class_weights).to(device)
    )
    ce_only_criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)

    unfreeze_epoch = int(getattr(args, "unfreeze_epoch", 0))
    backbones_unfrozen = unfreeze_epoch == 0

    def _set_backbone_train_mode(global_epoch: int) -> None:
        nonlocal backbones_unfrozen
        trainable = global_epoch >= unfreeze_epoch
        for trait_backbone in trait_backbones.values():
            trait_backbone.train(trainable)
            for param in trait_backbone.roberta.parameters():
                param.requires_grad = trainable
        if trainable and not backbones_unfrozen:
            print(f"Unfroze trait backbones at epoch {global_epoch + 1}.")
            backbones_unfrozen = True

    def _set_backbone_eval_mode() -> None:
        for trait_backbone in trait_backbones.values():
            trait_backbone.eval()

    def _build_batch_embeddings(text_batch: list[str]) -> dict[str, torch.Tensor]:
        encoded = tokenizer(
            text_batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=int(getattr(args, "embedding_max_seq_length", 512)),
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        embeddings: dict[str, torch.Tensor] = {}
        for trait_name, trait_backbone in trait_backbones.items():
            output = trait_backbone.roberta(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            embeddings[trait_name] = output.last_hidden_state[:, 0, :]
        return embeddings

    def _unpack_text_batch(batch):
        if args.loss_type == "combined":
            text_batch, labels_batch, contrastive_scores_batch = batch
            return list(text_batch), labels_batch, contrastive_scores_batch
        text_batch, labels_batch = batch
        return list(text_batch), labels_batch, None

    save_checkpoints = bool(getattr(args, "save_checkpoints", True))
    checkpoint_path = output_dir / build_checkpoint_name(args.dataset, "holistic", "best")

    stage1_epochs = int(getattr(args, "combined_stage1_epochs", 1)) if args.loss_type == "combined" else 0
    if args.loss_type == "combined":
        if stage1_epochs > 0:
            print(f"Stage 1/2 (CE-only) epochs: {stage1_epochs}")
            for stage1_epoch in tqdm(range(stage1_epochs), desc="Holistic Stage1"):
                if hasattr(model, "set_epoch"):
                    model.set_epoch(stage1_epoch)
                _set_backbone_train_mode(stage1_epoch)

                model.train()
                stage1_total_loss = 0.0
                stage1_main_loss = 0.0
                stage1_aux_loss = 0.0

                for batch in train_loader:
                    optimizer.zero_grad()
                    text_batch, labels_batch, _ = _unpack_text_batch(batch)
                    batch_embeddings = _build_batch_embeddings(text_batch)
                    labels_batch = labels_batch.to(device)

                    output = model(batch_embeddings)
                    main_loss = ce_only_criterion(output, labels_batch)
                    moe_aux_loss = model.moe_aux_loss if model.moe_aux_loss is not None else output.new_tensor(0.0)
                    total_loss = main_loss + (moe_aux_weight * moe_aux_loss)

                    total_loss.backward()
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()

                    stage1_total_loss += float(total_loss.item())
                    stage1_main_loss += float(main_loss.item())
                    stage1_aux_loss += float(moe_aux_loss.item())

                avg_stage1_total = stage1_total_loss / len(train_loader)
                avg_stage1_main = stage1_main_loss / len(train_loader)
                avg_stage1_aux = stage1_aux_loss / len(train_loader)
                _log_wandb(
                    run,
                    {
                        "stage1/train_loss": avg_stage1_total,
                        "stage1/train_main_loss": avg_stage1_main,
                        "stage1/train_moe_aux_loss": avg_stage1_aux,
                        "stage1/epoch": stage1_epoch + 1,
                    },
                )
                print(
                    f"Stage1 Epoch {stage1_epoch + 1}/{stage1_epochs} - "
                    f"train_total={avg_stage1_total:.4f}, train_main={avg_stage1_main:.4f}, "
                    f"train_aux={avg_stage1_aux:.4f}"
                )
        else:
            print("Skipping Stage 1 CE-only fine-tuning (--combined_stage1_epochs=0).")

        print("Initializing contrastive score means from full training split.")
        model.eval()
        _set_backbone_eval_mode()
        init_embeddings: list[torch.Tensor] = []
        init_scores: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in train_loader:
                text_batch, _, contrastive_scores_batch = _unpack_text_batch(batch)
                if contrastive_scores_batch is None:
                    raise RuntimeError("Combined loss requires contrastive scores in the dataloader.")
                batch_embeddings = _build_batch_embeddings(text_batch)
                contrastive_scores_batch = contrastive_scores_batch.to(device).float()
                _, holistic_embeddings = model(batch_embeddings, return_holistic_embedding=True)
                init_embeddings.append(holistic_embeddings.detach())
                init_scores.append(contrastive_scores_batch.detach())

        criterion.update_score_embeddings(
            torch.cat(init_embeddings, dim=0),
            torch.cat(init_scores, dim=0),
            reset=True,
        )

    best_qwk = -np.inf
    best_acc = 0.0
    best_state_payload: dict[str, object] | None = None
    no_improve_epochs = 0

    for epoch in tqdm(range(args.epochs), desc="Holistic"):
        global_epoch = stage1_epochs + epoch
        if hasattr(model, "set_epoch"):
            model.set_epoch(global_epoch)
        _set_backbone_train_mode(global_epoch)

        model.train()
        epoch_total_loss = 0.0
        epoch_main_loss = 0.0
        epoch_aux_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            text_batch, labels_batch, contrastive_scores_batch = _unpack_text_batch(batch)
            batch_embeddings = _build_batch_embeddings(text_batch)
            labels_batch = labels_batch.to(device)

            if args.loss_type == "combined":
                if contrastive_scores_batch is None:
                    raise RuntimeError("Combined loss requires contrastive scores in the dataloader.")
                contrastive_scores_batch = contrastive_scores_batch.to(device).float()
                output, holistic_embeddings = model(batch_embeddings, return_holistic_embedding=True)
                main_loss = criterion(
                    output,
                    labels_batch,
                    holistic_embeddings,
                    contrastive_scores_batch,
                )
            else:
                output = model(batch_embeddings)
                main_loss = criterion(output, labels_batch)

            moe_aux_loss = model.moe_aux_loss if model.moe_aux_loss is not None else output.new_tensor(0.0)
            total_loss = main_loss + (moe_aux_weight * moe_aux_loss)

            total_loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            epoch_total_loss += float(total_loss.item())
            epoch_main_loss += float(main_loss.item())
            epoch_aux_loss += float(moe_aux_loss.item())

        avg_train_total_loss = epoch_total_loss / len(train_loader)
        avg_train_main_loss = epoch_main_loss / len(train_loader)
        avg_train_aux_loss = epoch_aux_loss / len(train_loader)

        model.eval()
        _set_backbone_eval_mode()
        val_predictions: list[int] = []
        val_labels: list[int] = []
        val_total_loss = 0.0
        val_main_loss = 0.0
        val_aux_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                text_batch, labels_batch, contrastive_scores_batch = _unpack_text_batch(batch)
                batch_embeddings = _build_batch_embeddings(text_batch)
                labels_batch = labels_batch.to(device)

                if args.loss_type == "combined":
                    if contrastive_scores_batch is None:
                        raise RuntimeError("Combined loss requires contrastive scores in the dataloader.")
                    contrastive_scores_batch = contrastive_scores_batch.to(device).float()
                    output, holistic_embeddings = model(batch_embeddings, return_holistic_embedding=True)
                    main_loss = criterion(
                        output,
                        labels_batch,
                        holistic_embeddings,
                        contrastive_scores_batch,
                    )
                else:
                    output = model(batch_embeddings)
                    main_loss = criterion(output, labels_batch)

                moe_aux_loss = model.moe_aux_loss if model.moe_aux_loss is not None else output.new_tensor(0.0)
                total_loss = main_loss + (moe_aux_weight * moe_aux_loss)

                val_total_loss += float(total_loss.item())
                val_main_loss += float(main_loss.item())
                val_aux_loss += float(moe_aux_loss.item())
                val_predictions.extend(torch.argmax(output, dim=1).cpu().numpy().tolist())
                val_labels.extend(labels_batch.cpu().numpy().tolist())

        avg_val_total_loss = val_total_loss / len(val_loader)
        avg_val_main_loss = val_main_loss / len(val_loader)
        avg_val_aux_loss = val_aux_loss / len(val_loader)
        val_accuracy, val_qwk = calculate_accuracy_qwk(val_labels, val_predictions)

        _log_wandb(
            run,
            {
                "train_loss": avg_train_total_loss,
                "val_loss": avg_val_total_loss,
                "train_main_loss": avg_train_main_loss,
                "train_moe_aux_loss": avg_train_aux_loss,
                "val_main_loss": avg_val_main_loss,
                "val_moe_aux_loss": avg_val_aux_loss,
                "val_accuracy": val_accuracy,
                "val_qwk": val_qwk,
                "epoch": epoch,
            },
        )
        print(
            f"Epoch {epoch + 1}/{args.epochs} - "
            f"train_total={avg_train_total_loss:.4f}, train_main={avg_train_main_loss:.4f}, "
            f"train_aux={avg_train_aux_loss:.4f}, "
            f"val_total={avg_val_total_loss:.4f}, val_main={avg_val_main_loss:.4f}, "
            f"val_aux={avg_val_aux_loss:.4f}, val_acc={val_accuracy:.4f}, val_qwk={val_qwk:.4f}"
        )

        if val_qwk > best_qwk:
            best_qwk = val_qwk
            best_acc = val_accuracy
            no_improve_epochs = 0
            best_state_payload = {
                "scorer_state_dict": _state_dict_to_cpu(model),
                "trait_backbone_state_dicts": {
                    trait_name: _state_dict_to_cpu(trait_backbone)
                    for trait_name, trait_backbone in trait_backbones.items()
                },
            }
            if save_checkpoints:
                torch.save(best_state_payload, checkpoint_path)
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= args.patience:
            print("Early stopping triggered.")
            break

    if save_checkpoints and checkpoint_path.exists():
        loaded = torch.load(checkpoint_path, map_location="cpu")
    elif best_state_payload is not None:
        loaded = best_state_payload
    else:
        raise ConfigError("Failed to keep best model state during end-to-end training.")

    model.load_state_dict(loaded["scorer_state_dict"])
    trait_backbone_states = loaded.get("trait_backbone_state_dicts", {})
    for trait_name, trait_backbone in trait_backbones.items():
        trait_state = trait_backbone_states.get(trait_name)
        if trait_state is None:
            raise ConfigError(f"Missing trait backbone state for '{trait_name}' in best checkpoint payload.")
        trait_backbone.load_state_dict(trait_state)

    model.eval()
    _set_backbone_eval_mode()
    test_predictions: list[int] = []
    test_labels: list[int] = []
    test_total_loss = 0.0
    test_main_loss = 0.0
    test_aux_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            text_batch, labels_batch, contrastive_scores_batch = _unpack_text_batch(batch)
            batch_embeddings = _build_batch_embeddings(text_batch)
            labels_batch = labels_batch.to(device)

            if args.loss_type == "combined":
                if contrastive_scores_batch is None:
                    raise RuntimeError("Combined loss requires contrastive scores in the dataloader.")
                contrastive_scores_batch = contrastive_scores_batch.to(device).float()
                output, holistic_embeddings = model(batch_embeddings, return_holistic_embedding=True)
                main_loss = criterion(
                    output,
                    labels_batch,
                    holistic_embeddings,
                    contrastive_scores_batch,
                )
            else:
                output = model(batch_embeddings)
                main_loss = criterion(output, labels_batch)

            moe_aux_loss = model.moe_aux_loss if model.moe_aux_loss is not None else output.new_tensor(0.0)
            total_loss = main_loss + (moe_aux_weight * moe_aux_loss)

            test_total_loss += float(total_loss.item())
            test_main_loss += float(main_loss.item())
            test_aux_loss += float(moe_aux_loss.item())
            test_predictions.extend(torch.argmax(output, dim=1).cpu().numpy().tolist())
            test_labels.extend(labels_batch.cpu().numpy().tolist())

    avg_test_total_loss = test_total_loss / len(test_loader)
    avg_test_main_loss = test_main_loss / len(test_loader)
    avg_test_aux_loss = test_aux_loss / len(test_loader)
    test_accuracy, test_qwk = calculate_accuracy_qwk(test_labels, test_predictions)

    _log_wandb(
        run,
        {
            "best_val_qwk": best_qwk,
            "best_val_accuracy": best_acc,
            "test_loss": avg_test_total_loss,
            "test_main_loss": avg_test_main_loss,
            "test_moe_aux_loss": avg_test_aux_loss,
            "test_qwk": test_qwk,
            "test_accuracy": test_accuracy,
        },
    )
    if run is not None:
        run.finish()

    print("=" * 40)
    print(f"best val QWK: {best_qwk:.4f}")
    print(f"best val ACC: {best_acc:.4f}")
    print(f"test QWK: {test_qwk:.4f}")
    print(f"test ACC: {test_accuracy:.4f}")

    return {
        "best_val_qwk": float(best_qwk),
        "best_val_accuracy": float(best_acc),
        "test_qwk": float(test_qwk),
        "test_accuracy": float(test_accuracy),
        "test_loss": float(avg_test_total_loss),
        "test_main_loss": float(avg_test_main_loss),
        "test_moe_aux_loss": float(avg_test_aux_loss),
        "split_sizes": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "checkpoint_path": str(checkpoint_path) if save_checkpoints else None,
        "checkpoints_saved": save_checkpoints,
    }


def _print_grouped_holistic_summary(
    grouped_results: dict[str, dict[str, object]],
    *,
    split_by_column: str,
    skipped_groups: list[tuple[str, str]],
) -> None:
    print("\n" + "=" * 100)
    print(f"FINAL GROUPED HOLISTIC SUMMARY (split by '{split_by_column}')")
    print("=" * 100)
    print(f"{'Group':30s} {'Rows':>8s} {'Val QWK':>10s} {'Test QWK':>10s} {'Test ACC':>10s}")
    print("-" * 100)

    sorted_grouped_results = sorted(grouped_results.items(), key=lambda item: item[0])
    all_test_qwks: list[float] = []
    weighted_qwk_sum = 0.0
    total_rows = 0

    for group_label, payload in sorted_grouped_results:
        rows = int(payload["rows"])
        result = payload["result"]
        test_qwk = float(result["test_qwk"])
        all_test_qwks.append(test_qwk)
        weighted_qwk_sum += rows * test_qwk
        total_rows += rows
        print(
            f"{group_label[:30]:30s} {rows:8d} {float(result['best_val_qwk']):10.4f} "
            f"{test_qwk:10.4f} {float(result['test_accuracy']):10.4f}"
        )

    print("-" * 100)
    print(f"Total trained groups: {len(grouped_results)}")
    if all_test_qwks:
        if total_rows > 0:
            weighted_avg_test_qwk = weighted_qwk_sum / float(total_rows)
            print(f"Average Test QWK: {weighted_avg_test_qwk:.4f} (weighted by rows)")
        else:
            print("Average Test QWK: n/a")
        print(f"Best Test QWK:    {np.max(all_test_qwks):.4f}")
        print(f"Worst Test QWK:   {np.min(all_test_qwks):.4f}")

    if skipped_groups:
        print("\nSkipped groups:")
        for group_label, reason in sorted(skipped_groups, key=lambda item: item[0]):
            print(f"- {group_label}: {reason}")


def run_holistic_training(args):
    _print_holistic_startup_info(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    split_by_column = getattr(args, "split_by_column", None)
    predefined_split_column = getattr(args, "predefined_split_column", None)
    backbone_mode = str(getattr(args, "backbone_mode", "frozen"))

    trait_names = _trait_groups_to_names(args.trait_groups)
    if backbone_mode == "e2e":
        essay = pd.read_csv(args.csv_path)
        if predefined_split_column is not None:
            validate_required_columns(essay.columns, [predefined_split_column])
        if split_by_column is None:
            return _run_single_holistic_training_e2e(
                args,
                essay=essay,
                output_dir=output_dir,
                trait_names=trait_names,
            )

        required_group_columns = [split_by_column]
        if predefined_split_column is not None:
            required_group_columns.append(predefined_split_column)
        validate_required_columns(essay.columns, required_group_columns)
        grouped_results: dict[str, dict[str, object]] = {}
        skipped_groups: list[tuple[str, str]] = []

        grouped = essay.groupby(split_by_column, dropna=False, sort=False)
        groups = list(grouped)
        if not groups:
            raise ConfigError(f"No groups found in split column '{split_by_column}'.")

        for idx, (raw_value, group_df) in enumerate(groups, start=1):
            group_label = "NaN" if pd.isna(raw_value) else str(raw_value)
            group_key = group_label if group_label not in grouped_results else f"{group_label} ({idx})"
            group_dir_name = f"group_{idx:03d}_{split_by_column}_{_sanitize_for_path(group_label)}"
            group_output_dir = output_dir / group_dir_name
            group_essay = group_df.reset_index(drop=True)

            print("\n" + "=" * 100)
            print(
                f"[Group {idx}/{len(groups)}] {split_by_column}='{group_label}' "
                f"(rows={len(group_essay)})"
            )
            print("=" * 100)

            try:
                group_result = _run_single_holistic_training_e2e(
                    args,
                    essay=group_essay,
                    output_dir=group_output_dir,
                    trait_names=trait_names,
                    run_label=f"{split_by_column}={group_label}",
                )
            except ConfigError as exc:
                skipped_groups.append((group_key, str(exc)))
                print(f"Skipping group '{group_key}': {exc}")
                continue

            grouped_results[group_key] = {
                "rows": int(len(group_essay)),
                "output_dir": str(group_output_dir),
                "result": group_result,
            }
    else:
        npz_path = getattr(args, "npz_path", None)
        if npz_path is not None:
            aligned_npz, essay, _ = align_npz_and_csv(npz_path, args.csv_path)
            embeddings_dict = extract_trait_embeddings(aligned_npz)
        else:
            embeddings_dict, essay = _extract_embeddings_from_trait_checkpoints(
                args=args,
                trait_names=trait_names,
                device=device,
            )

        missing_traits = [name for name in trait_names if name not in embeddings_dict]
        if missing_traits:
            available = ", ".join(sorted(embeddings_dict.keys()))
            missing = ", ".join(missing_traits)
            raise ConfigError(
                f"Trait embeddings missing for: {missing}. Available embeddings: {available}"
            )
        _validate_embedding_dimensions(embeddings_dict, expected_dim=int(args.embedding_dim))
        if predefined_split_column is not None:
            validate_required_columns(essay.columns, [predefined_split_column])

        if split_by_column is None:
            return _run_single_holistic_training(
                args,
                embeddings_dict=embeddings_dict,
                essay=essay,
                output_dir=output_dir,
            )

        required_group_columns = [split_by_column]
        if predefined_split_column is not None:
            required_group_columns.append(predefined_split_column)
        validate_required_columns(essay.columns, required_group_columns)
        grouped_results = {}
        skipped_groups = []

        grouped = essay.groupby(split_by_column, dropna=False, sort=False)
        groups = list(grouped)
        if not groups:
            raise ConfigError(f"No groups found in split column '{split_by_column}'.")

        for idx, (raw_value, group_df) in enumerate(groups, start=1):
            group_label = "NaN" if pd.isna(raw_value) else str(raw_value)
            group_key = group_label if group_label not in grouped_results else f"{group_label} ({idx})"
            group_dir_name = f"group_{idx:03d}_{split_by_column}_{_sanitize_for_path(group_label)}"
            group_output_dir = output_dir / group_dir_name
            group_indices = group_df.index.to_numpy(dtype=int)

            group_embeddings = {
                trait_name: trait_embedding[group_indices]
                for trait_name, trait_embedding in embeddings_dict.items()
            }
            group_essay = essay.iloc[group_indices].reset_index(drop=True)

            print("\n" + "=" * 100)
            print(
                f"[Group {idx}/{len(groups)}] {split_by_column}='{group_label}' "
                f"(rows={len(group_indices)})"
            )
            print("=" * 100)

            try:
                group_result = _run_single_holistic_training(
                    args,
                    embeddings_dict=group_embeddings,
                    essay=group_essay,
                    output_dir=group_output_dir,
                    run_label=f"{split_by_column}={group_label}",
                )
            except ConfigError as exc:
                skipped_groups.append((group_key, str(exc)))
                print(f"Skipping group '{group_key}': {exc}")
                continue

            grouped_results[group_key] = {
                "rows": int(len(group_indices)),
                "output_dir": str(group_output_dir),
                "result": group_result,
            }

    if not grouped_results:
        raise ConfigError(
            f"No groups from '{split_by_column}' could be trained successfully. "
            "Check per-group sample and label distributions."
        )

    _print_grouped_holistic_summary(
        grouped_results,
        split_by_column=split_by_column,
        skipped_groups=skipped_groups,
    )

    return {
        "group_by": split_by_column,
        "groups": grouped_results,
        "skipped_groups": [{"group": group, "reason": reason} for group, reason in skipped_groups],
    }
