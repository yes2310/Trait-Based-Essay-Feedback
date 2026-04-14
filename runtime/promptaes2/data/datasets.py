from __future__ import annotations

from typing import Optional

import numpy as np
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    def __init__(self, labels: np.ndarray, *embedding_list: np.ndarray):
        self.labels = labels
        self.embedding_list = embedding_list

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        embeddings = tuple(embedding[idx] for embedding in self.embedding_list)
        return embeddings[0], self.labels[idx]


class MultiEmbeddingDataset(Dataset):
    def __init__(
        self,
        embeddings_dict: dict[str, np.ndarray],
        labels: np.ndarray,
        trait_labels: Optional[dict[str, np.ndarray]] = None,
        contrastive_scores: Optional[np.ndarray] = None,
    ):
        self.embeddings_dict = embeddings_dict
        self.labels = labels
        self.trait_labels = trait_labels if trait_labels is not None else {}
        self.contrastive_scores = contrastive_scores

        first_key = next(iter(self.embeddings_dict))
        expected_len = len(self.embeddings_dict[first_key])
        if expected_len != len(self.labels):
            raise ValueError(
                f"Embeddings and labels length mismatch: {expected_len} vs {len(self.labels)}"
            )
        if self.contrastive_scores is not None and len(self.contrastive_scores) != len(self.labels):
            raise ValueError(
                "contrastive_scores and labels length mismatch: "
                f"{len(self.contrastive_scores)} vs {len(self.labels)}"
            )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        embeddings = {trait: embedding[idx] for trait, embedding in self.embeddings_dict.items()}
        holistic_label = self.labels[idx]

        if self.trait_labels:
            trait_label_dict = {
                trait: labels[idx] for trait, labels in self.trait_labels.items()
            }
            if self.contrastive_scores is not None:
                return embeddings, holistic_label, trait_label_dict, self.contrastive_scores[idx]
            return embeddings, holistic_label, trait_label_dict

        if self.contrastive_scores is not None:
            return embeddings, holistic_label, self.contrastive_scores[idx]
        return embeddings, holistic_label
