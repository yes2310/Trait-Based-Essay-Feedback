from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        holistic_criterion,
        alpha: float = 0.7,
        normalize_trait_weights: bool = True,
    ):
        super().__init__()
        self.holistic_criterion = holistic_criterion
        self.alpha = alpha
        self.normalize_trait_weights = normalize_trait_weights
        self.trait_criterion = nn.CrossEntropyLoss(reduction="mean")

    def forward(
        self,
        holistic_logits: torch.Tensor,
        holistic_targets: torch.Tensor,
        trait_logits: Dict[str, torch.Tensor],
        trait_targets: Dict[str, torch.Tensor],
        embeddings: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if hasattr(self.holistic_criterion, "update_score_embeddings"):
            if embeddings is None:
                raise ValueError("CombinedLoss requires embeddings for contrastive component")
            holistic_loss = self.holistic_criterion(
                holistic_logits, holistic_targets, embeddings, holistic_targets
            )
        else:
            holistic_loss = self.holistic_criterion(holistic_logits, holistic_targets)

        trait_losses: Dict[str, torch.Tensor] = {}
        total_trait_loss = torch.tensor(0.0, device=holistic_logits.device)
        num_valid_traits = 0

        for trait_name, logits in trait_logits.items():
            if trait_name not in trait_targets:
                continue

            targets = trait_targets[trait_name]
            valid_mask = ~torch.isnan(targets)
            if valid_mask.sum() == 0:
                continue

            logits_valid = logits[valid_mask]
            targets_valid = targets[valid_mask].long()

            trait_loss = self.trait_criterion(logits_valid, targets_valid)
            trait_losses[trait_name] = trait_loss
            total_trait_loss = total_trait_loss + trait_loss
            num_valid_traits += 1

        if num_valid_traits > 0 and self.normalize_trait_weights:
            total_trait_loss = total_trait_loss / num_valid_traits

        total_loss = self.alpha * holistic_loss + (1 - self.alpha) * total_trait_loss
        return total_loss, holistic_loss, trait_losses
