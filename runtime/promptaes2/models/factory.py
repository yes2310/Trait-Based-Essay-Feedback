from __future__ import annotations

from promptaes2.models.canonical_moe_scorer import CanonicalMoECreativityScorer
from promptaes2.models.creativity_scorer import CreativityScorer
from promptaes2.types import TraitGroup


def build_scoring_model(
    *,
    model_variant: str,
    embedding_dim: int,
    hidden_sizes: list[int],
    num_classes: int,
    dropout_rates: list[float],
    use_skip1: bool,
    use_skip2: bool,
    use_skip3: bool,
    use_pre_homo_skip: bool,
    use_pre_hetero_skip: bool,
    trait_groups: TraitGroup,
    evolution_stage: str,
    warmup_epochs: int,
    enable_multitask: bool = False,
    trait_checkpoint_paths: dict[str, str] | None = None,
    group_num_experts: int = 8,
    classifier_num_experts: int = 8,
    router_top_k: int = 2,
):
    variant = model_variant.strip().lower()
    if variant == "legacy":
        return CreativityScorer(
            embedding_dim=embedding_dim,
            hidden_sizes=hidden_sizes,
            num_classes=num_classes,
            dropout_rates=dropout_rates,
            use_skip1=use_skip1,
            use_skip2=use_skip2,
            use_skip3=use_skip3,
            use_pre_homo_skip=use_pre_homo_skip,
            use_pre_hetero_skip=use_pre_hetero_skip,
            trait_groups=trait_groups,
            evolution_stage=evolution_stage,
            warmup_epochs=warmup_epochs,
            enable_multitask=enable_multitask,
            trait_checkpoint_paths=trait_checkpoint_paths,
        )
    if variant == "canonical_moe":
        return CanonicalMoECreativityScorer(
            embedding_dim=embedding_dim,
            hidden_sizes=hidden_sizes,
            num_classes=num_classes,
            dropout_rates=dropout_rates,
            use_skip1=use_skip1,
            use_skip2=use_skip2,
            use_skip3=use_skip3,
            use_pre_homo_skip=use_pre_homo_skip,
            use_pre_hetero_skip=use_pre_hetero_skip,
            trait_groups=trait_groups,
            evolution_stage=evolution_stage,
            warmup_epochs=warmup_epochs,
            enable_multitask=enable_multitask,
            trait_checkpoint_paths=trait_checkpoint_paths,
            group_num_experts=group_num_experts,
            classifier_num_experts=classifier_num_experts,
            router_top_k=router_top_k,
        )
    raise ValueError(
        f"Unsupported model_variant '{model_variant}'. "
        "Expected one of: legacy, canonical_moe"
    )
