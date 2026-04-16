from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


def compute_class_counts(labels: np.ndarray | list[int], *, num_classes: int | None = None) -> np.ndarray:
    labels_np = np.asarray(labels, dtype=np.int64)
    if labels_np.size == 0:
        raise ValueError("labels must not be empty")
    if np.min(labels_np) < 0:
        raise ValueError("labels must be non-negative integers")

    if num_classes is None:
        num_classes = int(np.max(labels_np)) + 1
    return np.bincount(labels_np, minlength=num_classes)


def compute_class_weights(
    labels: np.ndarray | list[int],
    *,
    num_classes: int | None = None,
    max_weight: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    counts = compute_class_counts(labels, num_classes=num_classes)
    weights = np.zeros(len(counts), dtype=np.float32)
    present_mask = counts > 0
    if not np.any(present_mask):
        raise ValueError("at least one class must be present")

    inverse = 1.0 / counts[present_mask].astype(np.float64)
    inverse /= inverse.mean()
    if max_weight is not None:
        inverse = np.minimum(inverse, float(max_weight))
    weights[present_mask] = inverse.astype(np.float32)
    return weights, counts


def build_class_weight_tensor(
    labels: np.ndarray | list[int],
    *,
    num_classes: int | None = None,
    max_weight: float | None = None,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, np.ndarray]:
    weights, counts = compute_class_weights(labels, num_classes=num_classes, max_weight=max_weight)
    tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    return tensor, counts


def build_weighted_sampler(
    labels: np.ndarray | list[int],
    *,
    num_classes: int | None = None,
    max_weight: float | None = None,
    power: float = 1.0,
) -> tuple[WeightedRandomSampler, np.ndarray, np.ndarray]:
    labels_np = np.asarray(labels, dtype=np.int64)
    counts = compute_class_counts(labels_np, num_classes=num_classes)
    class_weights = np.zeros(len(counts), dtype=np.float32)
    present_mask = counts > 0
    inverse = np.power(counts[present_mask].astype(np.float64), -float(power))
    inverse /= inverse.mean()
    if max_weight is not None:
        inverse = np.minimum(inverse, float(max_weight))
    class_weights[present_mask] = inverse.astype(np.float32)
    sample_weights = class_weights[labels_np].astype(np.float64)
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler, class_weights, counts


def format_class_weight_summary(
    counts: np.ndarray,
    weights: np.ndarray,
    *,
    class_labels: list[str] | None = None,
) -> str:
    parts: list[str] = []
    for idx, (count, weight) in enumerate(zip(counts.tolist(), weights.tolist())):
        if count <= 0:
            continue
        label = class_labels[idx] if class_labels is not None else str(idx)
        parts.append(f"{label}:n={int(count)},w={float(weight):.3f}")
    return ", ".join(parts) if parts else "n/a"
