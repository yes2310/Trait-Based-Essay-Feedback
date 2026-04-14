from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch


TraitGroup = List[Tuple[List[str], int]]
BatchEmbeddings = Dict[str, torch.Tensor]
HolisticBatch = Union[
    Tuple[BatchEmbeddings, torch.Tensor],
    Tuple[BatchEmbeddings, torch.Tensor, Dict[str, torch.Tensor]],
]


@dataclass(frozen=True)
class DatasetPreset:
    dataset: str
    default_traits: list[str]
    default_model_name: str
    default_data_path: str


@dataclass
class TrainArgs:
    epochs: int
    batch_size: int
    learning_rate: float
    cpu_workers: int
    seed: int
    device: Optional[str] = None
