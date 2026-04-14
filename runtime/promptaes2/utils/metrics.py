from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score


def calculate_accuracy_qwk(labels: list[int] | np.ndarray, predictions: list[int] | np.ndarray) -> tuple[float, float]:
    accuracy = accuracy_score(labels, predictions)
    qwk = cohen_kappa_score(labels, predictions, weights="quadratic")
    return float(accuracy), float(qwk)
