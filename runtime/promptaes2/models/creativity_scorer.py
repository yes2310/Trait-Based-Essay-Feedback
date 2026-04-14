from __future__ import annotations

import os

import torch
import torch.nn as nn

from promptaes2.models.blocks import BaseNetwork, GroupMoE, RelationProcessor
from promptaes2.types import TraitGroup


class CreativityScorer(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 768,
        hidden_sizes: list[int] | None = None,
        num_classes: int = 4,
        dropout_rates: list[float] | None = None,
        use_skip1: bool = False,
        use_skip2: bool = False,
        use_skip3: bool = False,
        use_pre_homo_skip: bool = False,
        use_pre_hetero_skip: bool = False,
        trait_groups: TraitGroup | None = None,
        evolution_stage: str = "full",
        warmup_epochs: int = 20,
        enable_multitask: bool = False,
        trait_checkpoint_paths: dict[str, str] | None = None,
    ):
        super().__init__()

        if trait_groups is None:
            raise ValueError("trait_groups must be provided")
        if evolution_stage not in {"baseline", "cross_attention", "full"}:
            raise ValueError(
                f"Unsupported evolution_stage '{evolution_stage}'. "
                "Expected one of: baseline, cross_attention, full"
            )

        hidden_sizes = hidden_sizes or [512, 512, 512]
        dropout_rates = dropout_rates or [0.3, 0.3]

        self.num_classes = num_classes
        self.trait_groups_info = trait_groups
        self.evolution_stage = evolution_stage
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
        self.enable_multitask = enable_multitask
        self.use_pre_homo_skip = use_pre_homo_skip
        self.use_pre_hetero_skip = use_pre_hetero_skip

        self.trait_groups = nn.ModuleList(
            [
                GroupMoE(
                    embedding_dim,
                    hidden_sizes,
                    embedding_dim,
                    group_dim,
                    dropout_rates,
                    use_skip1,
                )
                for _, group_dim in trait_groups
            ]
        )

        self.pre_homo_skip_layers = nn.ModuleList(
            [nn.Linear(embedding_dim, embedding_dim) for _ in trait_groups]
        )
        self.pre_hetero_skip_layers = nn.ModuleList(
            [nn.Linear(embedding_dim, embedding_dim) for _ in trait_groups]
        )
        for layer in self.pre_homo_skip_layers:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        for layer in self.pre_hetero_skip_layers:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

        if self.evolution_stage in {"cross_attention", "full"}:
            self.cross_attention = nn.ModuleList(
                [
                    RelationProcessor(embedding_dim, embedding_dim, use_skip2)
                    for _ in range(len(trait_groups))
                ]
            )
            self.gate = nn.Linear(embedding_dim * len(trait_groups), len(trait_groups))
        else:
            self.cross_attention = None
            self.gate = None

        self.experts = nn.ModuleList(
            [
                BaseNetwork(embedding_dim, hidden_sizes, num_classes, dropout_rates, use_skip3)
                for _ in range(len(trait_groups))
            ]
        )

        self.trait_weights: dict[str, torch.Tensor | None] | None = None
        self.group_weights: torch.Tensor | None = None
        self.moe_aux_loss: torch.Tensor | None = None

        if enable_multitask:
            self.trait_heads = nn.ModuleDict()
            for trait_names, _ in trait_groups:
                for trait_name in trait_names:
                    self.trait_heads[trait_name] = nn.Sequential(
                        nn.Dropout(0.1),
                        nn.Linear(embedding_dim, num_classes),
                    )
            if trait_checkpoint_paths is not None:
                self.load_trait_heads(trait_checkpoint_paths)
        else:
            self.trait_heads = nn.ModuleDict()

    def load_trait_heads(self, checkpoint_paths: dict[str, str]) -> None:
        for trait_name, ckpt_path in checkpoint_paths.items():
            if trait_name not in self.trait_heads:
                continue
            if not os.path.exists(ckpt_path):
                continue

            checkpoint = torch.load(ckpt_path, map_location="cpu")
            state = checkpoint.get("model_state_dict", checkpoint)

            classifier_weight = None
            classifier_bias = None
            for key, value in state.items():
                if "classifier" in key and key.endswith("weight"):
                    classifier_weight = value
                elif "classifier" in key and key.endswith("bias"):
                    classifier_bias = value

            if classifier_weight is None or classifier_bias is None:
                continue

            num_classes = classifier_weight.size(0)
            self.trait_heads[trait_name] = nn.Sequential(nn.Dropout(0.1), nn.Linear(768, num_classes))
            self.trait_heads[trait_name][1].weight.data = classifier_weight
            self.trait_heads[trait_name][1].bias.data = classifier_bias

    def freeze_trait_heads(self) -> None:
        for param in self.trait_heads.parameters():
            param.requires_grad = False

    def unfreeze_trait_heads(self) -> None:
        for param in self.trait_heads.parameters():
            param.requires_grad = True

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = epoch

    @staticmethod
    def _compute_load_balancing_loss(
        gate_logits: torch.Tensor,
        topk_indices: torch.Tensor,
        top_k: int,
    ) -> torch.Tensor:
        probs = torch.softmax(gate_logits, dim=1)
        assignment = torch.zeros_like(gate_logits)
        assignment.scatter_(1, topk_indices, 1.0)

        load = assignment.mean(dim=0) / float(top_k)
        importance = probs.mean(dim=0)
        num_experts = gate_logits.size(1)
        return float(num_experts) * torch.sum(load * importance)

    def _compute_group_outputs(
        self,
        batch_embeddings: dict[str, torch.Tensor],
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        outputs: list[torch.Tensor] = []
        aux_loss: torch.Tensor | None = None
        for group_idx, (trait_names, _) in enumerate(self.trait_groups_info):
            group_inputs = [batch_embeddings[trait_name] for trait_name in trait_names]
            group_output = self.trait_groups[group_idx](*group_inputs)
            if self.use_pre_homo_skip:
                homo_anchor = torch.stack(group_inputs, dim=0).mean(dim=0)
                group_output = group_output + self.pre_homo_skip_layers[group_idx](homo_anchor)
            outputs.append(group_output)

            group_aux = self.trait_groups[group_idx].aux_loss
            if group_aux is None:
                group_aux = group_output.new_tensor(0.0)
            aux_loss = group_aux if aux_loss is None else aux_loss + group_aux

        if aux_loss is None:
            raise RuntimeError("trait_groups must contain at least one group.")
        return outputs, aux_loss

    def _baseline_logits(self, group_outputs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = [expert(group_out) for expert, group_out in zip(self.experts, group_outputs)]
        logits = torch.stack(outputs, dim=1).mean(dim=1)
        pooled_embedding = torch.stack(group_outputs, dim=1).mean(dim=1)
        return logits, pooled_embedding

    def _relation_logits(
        self,
        group_outputs: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.cross_attention is not None
        assert self.gate is not None

        relation_inputs = group_outputs
        if self.use_pre_hetero_skip:
            relation_inputs = [
                output + self.pre_hetero_skip_layers[group_idx](output)
                for group_idx, output in enumerate(group_outputs)
            ]

        relation_outputs = [
            self.cross_attention[group_idx](
                relation_inputs[group_idx],
                relation_inputs[(group_idx + 1) % len(relation_inputs)],
            )
            for group_idx in range(len(relation_inputs))
        ]

        combined = torch.cat(relation_outputs, dim=1)
        gate_logits = self.gate(combined)
        num_experts = len(self.experts)
        top_k = min(2, num_experts)

        topk_logits, topk_indices = torch.topk(gate_logits, k=top_k, dim=1)
        topk_weights = torch.softmax(topk_logits, dim=1)

        dense_weights = torch.zeros_like(gate_logits)
        dense_weights.scatter_(1, topk_indices, topk_weights)
        self.group_weights = dense_weights

        dispatch_mask = torch.zeros_like(gate_logits, dtype=torch.bool)
        dispatch_mask.scatter_(1, topk_indices, True)

        batch_size = combined.size(0)
        logits = combined.new_zeros((batch_size, self.num_classes))
        for expert_idx, (expert, relation) in enumerate(zip(self.experts, relation_outputs)):
            selected = dispatch_mask[:, expert_idx]
            if not torch.any(selected):
                continue
            expert_output = expert(relation[selected])
            weights = dense_weights[selected, expert_idx].unsqueeze(-1)
            logits[selected] = logits[selected] + (weights * expert_output)

        aux_loss = self._compute_load_balancing_loss(
            gate_logits=gate_logits,
            topk_indices=topk_indices,
            top_k=top_k,
        )
        pooled_embedding = torch.stack(relation_outputs, dim=1).mean(dim=1)
        return logits, aux_loss, pooled_embedding

    def forward(
        self,
        batch_embeddings: dict[str, torch.Tensor],
        return_trait_logits: bool = False,
        return_holistic_embedding: bool = False,
    ):
        trait_logits: dict[str, torch.Tensor] = {}
        if self.enable_multitask and return_trait_logits:
            for trait_name, head in self.trait_heads.items():
                if trait_name in batch_embeddings:
                    trait_logits[trait_name] = head(batch_embeddings[trait_name])

        group_outputs, group_aux_loss = self._compute_group_outputs(batch_embeddings)
        self.trait_weights = {
            f"group{i + 1}": group.expert_weights for i, group in enumerate(self.trait_groups)
        }
        self.group_weights = None

        use_baseline = self.evolution_stage == "baseline" or (
            self.evolution_stage == "full" and self.current_epoch < self.warmup_epochs
        )
        if use_baseline:
            holistic_logits, holistic_embedding = self._baseline_logits(group_outputs)
            relation_aux_loss = holistic_logits.new_tensor(0.0)
        else:
            holistic_logits, relation_aux_loss, holistic_embedding = self._relation_logits(group_outputs)

        self.moe_aux_loss = group_aux_loss + relation_aux_loss

        if return_trait_logits and return_holistic_embedding:
            return holistic_logits, trait_logits, holistic_embedding
        if return_trait_logits:
            return holistic_logits, trait_logits
        if return_holistic_embedding:
            return holistic_logits, holistic_embedding
        return holistic_logits
