from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedLoss(nn.Module):
    def __init__(
        self,
        lambda1: float = 5.0,
        lambda2: float = 10.0,
        alpha: float = 0.5,
        beta1: float = 2.0,
        beta2: float = 1.0,
        beta3: float = 1.0,
        beta4: float = 1.0,
        margin: float = 0.0,
        ce_weight: torch.Tensor | None = None,
    ):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.beta4 = beta4
        self.margin = margin

        self.score_embeddings: dict[float, list[torch.Tensor]] = defaultdict(list)
        self.score_means: dict[float, torch.Tensor] = {}
        self.median_score: float | None = None
        self.is_initialized = False

        self.margin_ranking_fn = nn.MarginRankingLoss(margin=self.margin, reduction="mean")
        self.register_buffer("ce_weight", ce_weight.clone().detach() if ce_weight is not None else None)

    def cross_entropy_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, targets, weight=self.ce_weight)

    def mse_loss(self, pred: torch.Tensor, pred_prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_prob = F.softmax(pred, dim=1)
        target_onehot = F.one_hot(target.long(), num_classes=pred.size(1)).float()
        return F.mse_loss(pred_prob, target_onehot)

    def similarity_loss(self, pred: torch.Tensor, pred_prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_onehot = F.one_hot(target.long(), num_classes=pred.size(1)).float()
        pred_norm = F.normalize(pred_prob, dim=1)
        target_norm = F.normalize(target_onehot, dim=1)
        return 1 - torch.sum(pred_norm * target_norm, dim=1).mean()

    def margin_ranking_loss(
        self, pred: torch.Tensor, pred_prob: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        device = pred_prob.device
        pred_scores = pred_prob.max(dim=1)[0]

        batch_size = pred_scores.size(0)
        i_idx, j_idx = torch.triu_indices(batch_size, batch_size, offset=1, device=device)

        diff_mask = target[i_idx] != target[j_idx]
        i_idx = i_idx[diff_mask]
        j_idx = j_idx[diff_mask]

        if i_idx.numel() == 0:
            return torch.tensor(0.0, device=device)

        y = torch.sign(target[i_idx].float() - target[j_idx].float()).to(device)
        x1 = pred_scores[i_idx]
        x2 = pred_scores[j_idx]
        return self.margin_ranking_fn(x1, x2, y)

    def update_score_embeddings(
        self,
        embeddings: torch.Tensor,
        scores: torch.Tensor,
        *,
        reset: bool = True,
    ) -> None:
        if reset:
            self.score_embeddings.clear()
            self.score_means.clear()
            self.median_score = None
            self.is_initialized = False

        for emb, score in zip(embeddings, scores):
            score_value = float(score.item()) if isinstance(score, torch.Tensor) else float(score)
            self.score_embeddings[score_value].append(emb.detach())

        if not self.score_embeddings:
            raise ValueError("No score embeddings were provided to initialize contrastive samples.")

        unique_scores = torch.tensor(
            list(self.score_embeddings.keys()),
            dtype=torch.float32,
            device=embeddings.device,
        )
        self.median_score = float(unique_scores.median().item())

        self.score_means = {
            score: torch.stack(embs).mean(dim=0) for score, embs in self.score_embeddings.items()
        }
        self.is_initialized = True

    def get_contrastive_samples(self, input_score: float) -> tuple[torch.Tensor, torch.Tensor]:
        if self.median_score is None:
            raise ValueError("median_score is not initialized")

        positive_sample = self.score_means[input_score]
        is_input_above_median = input_score > self.median_score
        negative_scores = [
            score
            for score in self.score_means.keys()
            if (score > self.median_score) != is_input_above_median
        ]

        if negative_scores:
            negative_score = float(np.random.choice(negative_scores))
            negative_sample = self.score_means[negative_score]
        else:
            negative_sample = -positive_sample

        return positive_sample, negative_sample

    def contrastive_loss(
        self,
        input_embedding: torch.Tensor,
        positive_sample: torch.Tensor,
        negative_sample: torch.Tensor,
    ) -> torch.Tensor:
        positive_sample = positive_sample.to(input_embedding.device)
        negative_sample = negative_sample.to(input_embedding.device)
        dist_positive = F.pairwise_distance(input_embedding.unsqueeze(0), positive_sample.unsqueeze(0))
        dist_negative = F.pairwise_distance(input_embedding.unsqueeze(0), negative_sample.unsqueeze(0))
        p = dist_positive / (dist_positive + dist_negative + 1e-6)

        cos_sim = F.cosine_similarity(input_embedding.unsqueeze(0), negative_sample.unsqueeze(0))
        q = torch.abs(cos_sim)

        loss = p * self.lambda1 + q * self.lambda2
        return loss[0]

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        embeddings: torch.Tensor,
        scores: torch.Tensor,
    ) -> torch.Tensor:
        if not self.is_initialized:
            raise ValueError(
                "CombinedLoss is not initialized. "
                "Call update_score_embeddings() before the first forward pass."
            )

        logit_prob = F.softmax(logits, dim=1)
        mse = self.beta1 * self.mse_loss(logits, logit_prob, targets.float())
        mr = self.beta2 * self.margin_ranking_loss(logits, logit_prob, targets)
        sim = self.beta3 * self.similarity_loss(logits, logit_prob, targets)

        cl_loss = torch.tensor(0.0, device=logits.device)
        for emb, score in zip(embeddings, scores):
            score_value = float(score.item())
            if score_value not in self.score_means:
                continue
            positive, negative = self.get_contrastive_samples(score_value)
            cl_loss = cl_loss + self.contrastive_loss(emb, positive, negative)

        if len(embeddings) > 0:
            cl_loss = cl_loss / len(embeddings)
        cl_loss = cl_loss * self.beta4

        ce_loss = self.cross_entropy_loss(logits, targets)
        loss_cl = mse + mr + sim + cl_loss
        return self.alpha * ce_loss + (1 - self.alpha) * loss_cl
