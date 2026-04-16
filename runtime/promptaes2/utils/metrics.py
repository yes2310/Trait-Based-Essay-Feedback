from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score


def calculate_accuracy_qwk(labels: list[int] | np.ndarray, predictions: list[int] | np.ndarray) -> tuple[float, float]:
    accuracy = accuracy_score(labels, predictions)
    qwk = cohen_kappa_score(labels, predictions, weights="quadratic")
    return float(accuracy), float(qwk)


def format_label_distribution(
    values: list[int] | np.ndarray,
    *,
    class_labels: list[str] | None = None,
) -> str:
    values_np = np.asarray(values, dtype=np.int64)
    if values_np.size == 0:
        return "n/a"
    counts = np.bincount(values_np)
    total = int(counts.sum())
    parts: list[str] = []
    for idx, count in enumerate(counts.tolist()):
        if count <= 0:
            continue
        label = class_labels[idx] if class_labels is not None and idx < len(class_labels) else str(idx)
        ratio = (float(count) / float(total)) * 100.0
        parts.append(f"{label}:{count} ({ratio:.1f}%)")
    return ", ".join(parts) if parts else "n/a"


def calculate_majority_baseline(
    labels: list[int] | np.ndarray,
) -> tuple[int, float, float]:
    labels_np = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels_np)
    majority_class = int(np.argmax(counts))
    predictions = np.full_like(labels_np, majority_class)
    accuracy, qwk = calculate_accuracy_qwk(labels_np, predictions)
    return majority_class, accuracy, qwk
