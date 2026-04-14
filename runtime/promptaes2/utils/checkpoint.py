from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class EarlyStoppingResult:
    early_stop: bool
    improved: bool


class EarlyStopping:
    def __init__(
        self,
        patience: int,
        path: str | Path,
        metric: str = "qwk",
        delta: float = 0.0,
        verbose: bool = False,
    ) -> None:
        self.patience = patience
        self.path = Path(path)
        self.metric = metric
        self.delta = delta
        self.verbose = verbose

        self.counter = 0
        self.best_score: float | None = None
        self.best_value = -np.inf if metric != "loss" else np.inf

    def step(self, value: float, model: torch.nn.Module) -> EarlyStoppingResult:
        score = -value if self.metric == "loss" else value
        improved = self.best_score is None or score >= self.best_score + self.delta

        if improved:
            self.best_score = score
            self.best_value = value
            self.counter = 0
            self.path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), self.path)
            if self.verbose:
                print(f"Validation {self.metric} improved. Saved checkpoint: {self.path}")
            return EarlyStoppingResult(early_stop=False, improved=True)

        self.counter += 1
        if self.verbose:
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
        return EarlyStoppingResult(early_stop=self.counter >= self.patience, improved=False)


def build_checkpoint_name(dataset: str, stage: str, name: str, epoch: int | None = None) -> str:
    if epoch is None:
        return f"{dataset}_{stage}_{name}.pt"
    return f"{dataset}_{stage}_{name}_epoch{epoch:03d}.pt"
