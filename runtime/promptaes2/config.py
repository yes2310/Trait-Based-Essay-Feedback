from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from promptaes2.types import DatasetPreset, TraitGroup


class ConfigError(ValueError):
    """Raised when user-facing configuration is invalid."""


class DataAlignmentError(ValueError):
    """Raised when NPZ and CSV cannot be aligned by ID."""


class MissingColumnError(ValueError):
    """Raised when required columns are absent from input CSV."""


DATASET_PRESETS: dict[str, DatasetPreset] = {
    "asap": DatasetPreset(
        dataset="asap",
        default_traits=[
            "content",
            "word_choice",
            "sentence_fluency",
            "conventions",
            "organization",
            "prompt_adherence",
            "language",
            "narrativity",
        ],
        default_model_name="roberta-base",
        default_data_path="data/asap/train/split/prompt_all_essay_labed.csv",
    ),
    "aihub": DatasetPreset(
        dataset="aihub",
        default_traits=[
            "org1",
            "org2",
            "org3",
            "org4",
            "cont1",
            "cont2",
            "cont3",
            "exp1",
            "exp2",
            "exp3",
        ],
        default_model_name="klue/roberta-base",
        default_data_path="data/aihub/train/all/train_all.csv",
    ),
    "leaf": DatasetPreset(
        dataset="leaf",
        default_traits=[
            "grammar_accuracy",
            "appropriateness_of_word_use",
            "elasticity_of_sentence_expression",
            "appropriateness_of_structure_within_a_paragraph",
            "adequacy_of_inter_paragraph_structure",
            "consistency_of_structure",
            "appropriateness_of_portion_size",
            "clarity_of_topic",
            "specificity_of_explanation",
            "creativity_of_thought",
        ],
        default_model_name="roberta-base",
        default_data_path="data/leaf_merged.csv",
    ),
}


def get_dataset_preset(dataset: str) -> DatasetPreset:
    if dataset not in DATASET_PRESETS:
        raise ConfigError(
            f"Unsupported dataset '{dataset}'. Available: {sorted(DATASET_PRESETS.keys())}"
        )
    return DATASET_PRESETS[dataset]


def parse_hidden_sizes(value: str) -> list[int]:
    try:
        parsed = [int(size) for size in value.split("-") if size]
    except ValueError as exc:
        raise ConfigError(
            "Hidden sizes format should be dash-separated integers, e.g. '512-512'."
        ) from exc

    if not parsed:
        raise ConfigError("Hidden sizes must contain at least one integer.")
    return parsed


def parse_dropout_rates(value: str) -> list[float]:
    try:
        parsed = [float(rate) for rate in value.split("-") if rate]
    except ValueError as exc:
        raise ConfigError(
            "Dropout format should be 'rate1-rate2' with numeric values between 0 and 1."
        ) from exc

    if len(parsed) != 2 or any(rate < 0.0 or rate > 1.0 for rate in parsed):
        raise ConfigError("Dropout format should be 'rate1-rate2' with rates between 0 and 1.")
    return parsed


def parse_trait_groups(value: str) -> TraitGroup:
    groups: TraitGroup = []
    raw_groups = [chunk.strip() for chunk in value.split(";") if chunk.strip()]
    if not raw_groups:
        raise ConfigError(
            "Trait groups format should be 'name1,name2:dim1;name3,name4:dim2'."
        )

    for group in raw_groups:
        if ":" not in group:
            raise ConfigError(
                f"Invalid trait group '{group}'. Expected format 'trait1,trait2:2' or 'trait1,trait2,trait3:3'."
            )

        traits_raw, dim_raw = group.split(":", 1)
        trait_names = [trait.strip() for trait in traits_raw.split(",") if trait.strip()]
        if not trait_names:
            raise ConfigError(f"Trait group '{group}' has no trait names.")

        if not dim_raw.isdigit():
            raise ConfigError(
                f"Invalid dimension '{dim_raw}' in group '{group}'. It must be an integer."
            )

        groups.append((trait_names, int(dim_raw)))

    return groups


def validate_required_columns(columns: Iterable[str], required: Iterable[str]) -> None:
    column_set = set(columns)
    missing = [col for col in required if col not in column_set]
    if missing:
        available = ", ".join(str(col) for col in columns)
        missing_text = ", ".join(missing)
        raise MissingColumnError(
            f"Missing required columns: {missing_text}. Available columns: {available}"
        )
