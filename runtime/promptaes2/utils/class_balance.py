from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


CLASS_BALANCE_MODES = ("none", "loss", "loss_and_sampler")


def normalize_class_balance_mode(mode: str | None) -> str:
    normalized = "none" if mode is None else str(mode).strip().lower()
    if normalized not in CLASS_BALANCE_MODES:
        expected = ", ".join(CLASS_BALANCE_MODES)
        raise ValueError(f"Unsupported class_balance_mode '{mode}'. Expected one of: {expected}")
    return normalized


def _to_numpy_labels(labels: Sequence[int] | np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(labels, torch.Tensor):
        labels_np = labels.detach().cpu().numpy()
    else:
        labels_np = np.asarray(labels)
    if labels_np.ndim != 1:
        raise ValueError(f"labels must be 1-dimensional, got shape={labels_np.shape}")
    return labels_np.astype(np.int64, copy=False)


def build_class_weight_tensor(
    labels: Sequence[int] | np.ndarray | torch.Tensor,
    *,
    num_classes: int | None = None,
) -> torch.Tensor:
    labels_np = _to_numpy_labels(labels)
    if labels_np.size == 0:
        raise ValueError("labels must not be empty")

    inferred_num_classes = int(labels_np.max()) + 1
    class_count = inferred_num_classes if num_classes is None else int(num_classes)
    if class_count < inferred_num_classes:
        raise ValueError(
            f"num_classes={class_count} is smaller than inferred class count={inferred_num_classes}"
        )

    counts = np.bincount(labels_np, minlength=class_count).astype(np.float32)
    weights = np.zeros_like(counts)
    valid_mask = counts > 0
    weights[valid_mask] = 1.0 / np.sqrt(counts[valid_mask])
    if valid_mask.any():
        weights[valid_mask] = weights[valid_mask] / float(weights[valid_mask].mean())

    return torch.tensor(weights, dtype=torch.float32)


def maybe_build_class_weight_tensor(
    labels: Sequence[int] | np.ndarray | torch.Tensor,
    *,
    class_balance_mode: str,
    num_classes: int | None = None,
) -> torch.Tensor | None:
    normalized_mode = normalize_class_balance_mode(class_balance_mode)
    if normalized_mode == "none":
        return None
    return build_class_weight_tensor(labels, num_classes=num_classes)


def maybe_build_weighted_sampler(
    labels: Sequence[int] | np.ndarray | torch.Tensor,
    *,
    class_balance_mode: str,
    num_classes: int | None = None,
    seed: int | None = None,
) -> WeightedRandomSampler | None:
    normalized_mode = normalize_class_balance_mode(class_balance_mode)
    if normalized_mode != "loss_and_sampler":
        return None

    labels_np = _to_numpy_labels(labels)
    class_weights = build_class_weight_tensor(labels_np, num_classes=num_classes).cpu().numpy()
    sample_weights = torch.tensor(class_weights[labels_np], dtype=torch.double)

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )
