from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int],
        output_size: int,
        dropout_rates: list[float] | None = None,
        use_skip: bool = False,
    ):
        super().__init__()
        if not hidden_sizes:
            raise ValueError("hidden_sizes must not be empty")

        self.use_skip = use_skip
        self.dropout_rates = dropout_rates or [0.1, 0.1]

        self.input_layer = nn.Linear(input_size, hidden_sizes[0])
        self.input_dropout = nn.Dropout(self.dropout_rates[0])

        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]) for i in range(len(hidden_sizes) - 1)]
        )
        self.hidden_dropouts = nn.ModuleList(
            [nn.Dropout(self.dropout_rates[1]) for _ in range(len(hidden_sizes) - 1)]
        )
        self.skip_projection = (
            nn.Identity()
            if hidden_sizes[0] == hidden_sizes[-1]
            else nn.Linear(hidden_sizes[0], hidden_sizes[-1])
        )

        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.input_layer(x))
        x = self.input_dropout(x)
        identity = x

        for hidden_layer, dropout in zip(self.hidden_layers, self.hidden_dropouts):
            x = F.relu(hidden_layer(x))
            x = dropout(x)

        if self.use_skip:
            x = x + self.skip_projection(identity)

        return self.output_layer(x)


class GroupMoE(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int],
        output_size: int,
        num_traits: int,
        dropout_rates: list[float] | None = None,
        use_skip: bool = False,
    ):
        super().__init__()
        self.output_size = output_size
        self.experts = nn.ModuleList(
            [
                BaseNetwork(input_size, hidden_sizes, output_size, dropout_rates, use_skip)
                for _ in range(num_traits)
            ]
        )
        self.gate = nn.Linear(input_size * num_traits, num_traits)
        self.expert_weights: torch.Tensor | None = None
        self.aux_loss: torch.Tensor | None = None

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

    def forward(self, *trait_embeddings: torch.Tensor) -> torch.Tensor:
        if len(trait_embeddings) != len(self.experts):
            raise ValueError(
                f"Expected {len(self.experts)} trait embeddings, got {len(trait_embeddings)}."
            )

        combined = torch.cat(trait_embeddings, dim=1)
        gate_logits = self.gate(combined)
        num_experts = len(self.experts)
        top_k = min(2, num_experts)

        topk_logits, topk_indices = torch.topk(gate_logits, k=top_k, dim=1)
        topk_weights = torch.softmax(topk_logits, dim=1)

        dense_weights = torch.zeros_like(gate_logits)
        dense_weights.scatter_(1, topk_indices, topk_weights)
        self.expert_weights = dense_weights

        dispatch_mask = torch.zeros_like(gate_logits, dtype=torch.bool)
        dispatch_mask.scatter_(1, topk_indices, True)

        batch_size = combined.size(0)
        moe_output = combined.new_zeros((batch_size, self.output_size))
        for expert_idx, (expert, emb) in enumerate(zip(self.experts, trait_embeddings)):
            selected = dispatch_mask[:, expert_idx]
            if not torch.any(selected):
                continue
            expert_output = expert(emb[selected])
            weights = dense_weights[selected, expert_idx].unsqueeze(-1)
            moe_output[selected] = moe_output[selected] + (weights * expert_output)

        self.aux_loss = self._compute_load_balancing_loss(
            gate_logits=gate_logits,
            topk_indices=topk_indices,
            top_k=top_k,
        )
        return moe_output


class CrossAttention(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.query_linear = nn.Linear(embedding_dim, embedding_dim)
        self.key_linear = nn.Linear(embedding_dim, embedding_dim)
        self.value_linear = nn.Linear(embedding_dim, embedding_dim)
        self.attention_weights: torch.Tensor | None = None

    def forward(self, cls1: torch.Tensor, cls2: torch.Tensor) -> torch.Tensor:
        query = self.query_linear(cls1).unsqueeze(1)
        key = self.key_linear(cls2).unsqueeze(1)
        value = self.value_linear(cls2).unsqueeze(1)

        scores = torch.matmul(query, key.transpose(-2, -1)) / (cls1.size(-1) ** 0.5)
        attention_weights = torch.softmax(scores, dim=-1)
        self.attention_weights = attention_weights

        return torch.matmul(attention_weights, value).squeeze(1)


class RelationProcessor(nn.Module):
    def __init__(self, embedding_dim: int, output_dim: int, use_skip: bool = False):
        super().__init__()
        self.attention = CrossAttention(embedding_dim)
        self.relation_projection = nn.Linear(embedding_dim, output_dim)
        self.use_skip = use_skip

    def forward(self, cls1: torch.Tensor, cls2: torch.Tensor) -> torch.Tensor:
        attn_output = self.attention(cls1, cls2)
        if self.use_skip:
            attn_output = attn_output + cls1 + cls2
        return self.relation_projection(attn_output)


class GroupInteractionEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_groups: int,
        *,
        num_heads: int = 4,
        dropout: float = 0.1,
        ff_multiplier: int = 2,
        use_post_skip: bool = False,
    ) -> None:
        super().__init__()
        if num_groups < 1:
            raise ValueError("num_groups must be >= 1")
        if num_heads < 1:
            raise ValueError("num_heads must be >= 1")

        effective_num_heads = min(num_heads, embedding_dim)
        while effective_num_heads > 1 and (embedding_dim % effective_num_heads) != 0:
            effective_num_heads -= 1

        self.num_groups = num_groups
        self.num_heads = effective_num_heads
        self.use_post_skip = use_post_skip
        self.group_type_embeddings = nn.Parameter(torch.zeros(1, num_groups, embedding_dim))
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=effective_num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(embedding_dim)
        hidden_dim = embedding_dim * ff_multiplier
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(embedding_dim)
        self.attention_weights: torch.Tensor | None = None

    def forward(self, group_tokens: torch.Tensor) -> torch.Tensor:
        if group_tokens.ndim != 3:
            raise ValueError(
                f"group_tokens must have shape [batch, groups, embedding_dim], got {group_tokens.shape}"
            )
        if group_tokens.size(1) != self.num_groups:
            raise ValueError(
                f"Expected {self.num_groups} group tokens, got {group_tokens.size(1)}"
            )

        residual_input = group_tokens
        x = group_tokens + self.group_type_embeddings[:, : group_tokens.size(1), :]
        attn_output, attention_weights = self.attention(
            x,
            x,
            x,
            need_weights=True,
            average_attn_weights=False,
        )
        self.attention_weights = attention_weights.mean(dim=1)
        x = self.attention_norm(x + attn_output)

        ffn_output = self.ffn(x)
        x = self.ffn_norm(x + ffn_output)
        if self.use_post_skip:
            x = x + residual_input
        return x
