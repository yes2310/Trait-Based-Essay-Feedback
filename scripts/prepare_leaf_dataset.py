#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

import pandas as pd


FEEDBACK_DIMENSION_COLUMNS = [
    "alignment_with_topic",
    "spelling_grammar_style",
    "clarity_of_view_point",
    "arguments_supporting_details",
]

TRAIT_MAPPINGS = [
    ("trait_1", "grammar_accuracy", "Grammar accuracy"),
    ("trait_2", "appropriateness_of_word_use", "Appropriateness of word use"),
    ("trait_3", "elasticity_of_sentence_expression", "Elasticity of sentence expression"),
    (
        "trait_4",
        "appropriateness_of_structure_within_a_paragraph",
        "Appropriateness of structure within a paragraph",
    ),
    (
        "trait_5",
        "adequacy_of_inter_paragraph_structure",
        "Adequacy of inter-paragraph structure",
    ),
    ("trait_6", "consistency_of_structure", "Consistency of structure"),
    ("trait_7", "appropriateness_of_portion_size", "Appropriateness of portion size"),
    ("trait_8", "clarity_of_topic", "Clarity of topic"),
    ("trait_9", "specificity_of_explanation", "Specificity of explanation"),
    ("trait_10", "creativity_of_thought", "Creativity of thought"),
]

NORMALIZED_RUBRICS = {
    "grammar_accuracy": {
        "raw_column": "trait_1",
        "display_name": "Grammar accuracy",
        "scores": {
            "5": "Command of grammar and usage with few or no errors",
            "4": "Minimum errors in grammar and usage",
            "3": "Some errors in grammar and usage",
            "2": "Many errors in grammar and usage",
            "1": "Errors in grammar and usage throughout",
        },
    },
    "appropriateness_of_word_use": {
        "raw_column": "trait_2",
        "display_name": "Appropriateness of word use",
        "scores": {
            "5": "Uses the most appropriate words for the context",
            "4": "Mostly appropriate word choices, with rare instances of slightly awkward phrasing",
            "3": "Some word choices feel out of place, vague, very complex or simple for the context",
            "2": "Several words are misused or inappropriate for the intended meaning",
            "1": "Many words incorrectly used, leading to serious misunderstanding",
        },
    },
    "elasticity_of_sentence_expression": {
        "raw_column": "trait_3",
        "display_name": "Elasticity of sentence expression",
        "scores": {
            "5": "Sentences flow smoothly with varied structures",
            "4": "Generally fluid and adaptable, but occasional repetition in structure",
            "3": "Some flexibility but noticeable repetition in sentence structure",
            "2": "Sentences feel rigid or overly structured",
            "1": "Extremely repetitive, awkward, or mechanically structured sentences",
        },
    },
    "appropriateness_of_structure_within_a_paragraph": {
        "raw_column": "trait_4",
        "display_name": "Appropriateness of structure within a paragraph",
        "scores": {
            "5": "Sentences flow smoothly with a clear and logical progression of ideas",
            "4": "Sentences are generally well-organized with only slight inconsistencies in flow",
            "3": "Paragraph structure is somewhat logical but may feel disjointed in parts",
            "2": "Sentences are placed in an order that disrupts readability for logical progression",
            "1": "Sentences appear randomly placed with no clear logical order",
        },
    },
    "adequacy_of_inter_paragraph_structure": {
        "raw_column": "trait_5",
        "display_name": "Adequacy of inter-paragraph structure",
        "scores": {
            "5": "Each paragraph is clearly linked to the previous and subsequent ones, contributing to a smooth and logical flow",
            "4": "The paragraphs are generally well-structured with a clear connection between them",
            "3": "Paragraphs are related, but the connections between them can feel weak",
            "2": "The relation between paragraphs is unclear, with incomplete transitions",
            "1": "The paragraphs feel disjoint or randomly ordered with no logical progression",
        },
    },
    "consistency_of_structure": {
        "raw_column": "trait_6",
        "display_name": "Consistency of structure",
        "scores": {
            "5": "The essay maintains a consistent structure throughout",
            "4": "Overall, the structure is consistent, but there may be a few minor instances of sentence or paragraph length variations or slight shifts in tone",
            "3": "Some variations in structure are noticeable, but they do not majorly affect comprehension",
            "2": "Significant variations in sentence length, structure, or formatting that detracts from the clarity and flow of the text",
            "1": "Frequent changes in formatting, tone or sentence structure disrupt the reader experience",
        },
    },
    "appropriateness_of_portion_size": {
        "raw_column": "trait_7",
        "display_name": "Appropriateness of portion size",
        "scores": {
            "5": "Each paragraph and section is appropriately sized, neither too long nor too short",
            "4": "The portion size is generally well-maintained, but there may be a few slightly long or short sections",
            "3": "Some paragraphs or sections feel either overly brief or overly detailed, which can slightly hinder comprehension flow",
            "2": "The text has noticeable inconsistencies in portion sizes, where some sections are too long and others too brief",
            "1": "Portions are severely unbalanced, with overly long sections that are hard to digest and very short sections that lack necessary information",
        },
    },
    "clarity_of_topic": {
        "raw_column": "trait_8",
        "display_name": "Clarity of topic",
        "scores": {
            "5": "The main topic is immediately clear, well-defined, and consistently addressed throughout the text",
            "4": "The topic is generally clear, though there may be minor ambiguities or moments where the focus drifts slightly",
            "3": "The topic is somewhat clear, but some sections may feel vague or lack clear focus",
            "2": "The topic is difficult to identify, either because it is underdeveloped or not clearly stated",
            "1": "The topic is completely unclear, making it difficult for the reader to understand the purpose of the text",
        },
    },
    "specificity_of_explanation": {
        "raw_column": "trait_9",
        "display_name": "Specificity of explanation",
        "scores": {
            "5": "The explanation is clear, focused, and provides all the relevant details to fully understand the topic",
            "4": "The explanation is generally clear and detailed, though there may be occasional generalizations or lack of examples",
            "3": "The explanation is somewhat clear but lacks sufficient detail in certain areas",
            "2": "The explanation is too vague or incomplete, lacking sufficient detail or examples",
            "1": "The explanation is general and completely lacks detail",
        },
    },
    "creativity_of_thought": {
        "raw_column": "trait_10",
        "display_name": "Creativity of thought",
        "scores": {
            "5": "The essay is original, imaginative, and thought provoking",
            "4": "The essay contains novel insights or unique approaches, but with a slight reliance on common approaches",
            "3": "The essay contains some creative ideas, but many of them are not highly original",
            "2": "The ideas presented are mostly conventional and lack originality",
            "1": "The essay lacks any creative or original thinking, offering ideas that are entirely common or uninspired",
        },
        "normalization_note": "Raw trait_10 values of 0 are treated as missing in the normalized creativity_of_thought column.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the merged LEAF dataset from the released zip bundle.")
    parser.add_argument("--zip-path", required=True, help="Path to leaf--main.zip")
    parser.add_argument(
        "--workspace-dir",
        default=str(Path(__file__).resolve().parents[1]),
        help="Repository root containing the data/ directory",
    )
    return parser.parse_args()


def _load_split_csvs(zf: zipfile.ZipFile, prefix: str) -> pd.DataFrame:
    frames = [
        pd.read_csv(zf.open(f"{prefix}/{split}.csv"))
        for split in ("train", "dev", "test")
    ]
    return pd.concat(frames, ignore_index=True)


def _normalize_text(value: object) -> str:
    return str(value).replace("\\r\\n", "\n").strip()


def _assert_alignment(root_df: pd.DataFrame, dim_df: pd.DataFrame) -> None:
    root_sorted = root_df.sort_values(["ID", "split"]).reset_index(drop=True)
    dim_sorted = dim_df.sort_values(["ID", "split"]).reset_index(drop=True)

    if len(root_sorted) != len(dim_sorted):
        raise ValueError(f"Row count mismatch: root={len(root_sorted)} vs dim={len(dim_sorted)}")

    if not root_sorted[["ID", "split"]].equals(dim_sorted[["ID", "split"]]):
        raise ValueError("Root and feedback-dimension splits do not align on (ID, split).")

    comparable_pairs = [
        ("source_url", "source_url"),
        ("essay_title", "prompt"),
        ("Type", "type"),
    ]
    for root_col, dim_col in comparable_pairs:
        if not (root_sorted[root_col].astype(str) == dim_sorted[dim_col].astype(str)).all():
            raise ValueError(f"Column mismatch between root.{root_col} and dim.{dim_col}")


def _build_compatibility_frame(root_df: pd.DataFrame, dim_df: pd.DataFrame) -> pd.DataFrame:
    merged = root_df.merge(dim_df, on=["ID", "split"], suffixes=("_root", "_dim"))
    total_score = merged[FEEDBACK_DIMENSION_COLUMNS].sum(axis=1).astype(int)

    compatibility = pd.DataFrame(
        {
            "ID": merged["ID"].astype(int),
            "total_score": total_score,
            "split": merged["split"],
            "type": merged["Type"],
            "essay_title": merged["essay_title"],
            "source_url": merged["source_url_root"],
            "text": merged["essay_text"],
            "human_feedback": merged["human_feedback_text"],
            "ai_augmented_feedback": merged["AI_augmented_feedback_text"],
            "alignment_with_topic": merged["alignment_with_topic"].astype("Int64"),
            "spelling_grammar_style": merged["spelling_grammar_style"].astype("Int64"),
            "clarity_of_view_point": merged["clarity_of_view_point"].astype("Int64"),
            "arguments_supporting_details": merged["arguments_supporting_details"].astype("Int64"),
            "feedback_dimension_overall": merged["overall"].astype("Int64"),
        }
    )
    return compatibility


def _build_multitrait_frame(
    compatibility_df: pd.DataFrame,
    root_df: pd.DataFrame,
) -> pd.DataFrame:
    merged = compatibility_df.merge(
        root_df[
            [
                "ID",
                "split",
                "trait_1",
                "trait_2",
                "trait_3",
                "trait_4",
                "trait_5",
                "trait_6",
                "trait_7",
                "trait_8",
                "trait_9",
                "trait_10",
            ]
        ],
        on=["ID", "split"],
        how="left",
    )

    for raw_column, _, _ in TRAIT_MAPPINGS:
        merged[raw_column] = merged[raw_column].astype("Int64")

    for raw_column, semantic_column, _ in TRAIT_MAPPINGS:
        semantic_values = merged[raw_column].copy()
        if semantic_column == "creativity_of_thought":
            semantic_values = semantic_values.mask(~semantic_values.isin([1, 2, 3, 4, 5]))
        merged[semantic_column] = semantic_values.astype("Int64")

    return merged


def _write_normalized_rubrics(destination: Path) -> None:
    destination.write_text(
        json.dumps(NORMALIZED_RUBRICS, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    zip_path = Path(args.zip_path).expanduser().resolve()
    workspace_dir = Path(args.workspace_dir).expanduser().resolve()

    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")

    data_dir = workspace_dir / "data"
    raw_dir = data_dir / "raw"
    extract_root = raw_dir / "leaf--main"
    raw_dir.mkdir(parents=True, exist_ok=True)

    copied_zip_path = raw_dir / zip_path.name
    shutil.copy2(zip_path, copied_zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(raw_dir)
        root_df = _load_split_csvs(zf, "leaf--main")
        dim_df = _load_split_csvs(zf, "leaf--main/LEAF-Feedback-dimension")

    _assert_alignment(root_df, dim_df)

    compatibility_df = _build_compatibility_frame(root_df, dim_df)
    multitrait_df = _build_multitrait_frame(compatibility_df, root_df)

    compatibility_path = data_dir / "leaf_merged.feedback4_only.csv"
    merged_path = data_dir / "leaf_merged.csv"
    rubrics_path = extract_root / "rubrics.normalized.json"

    compatibility_df.to_csv(compatibility_path, index=False)
    multitrait_df.to_csv(merged_path, index=False)
    _write_normalized_rubrics(rubrics_path)

    print(f"Copied zip: {copied_zip_path}")
    print(f"Extracted raw dataset: {extract_root}")
    print(f"Wrote compatibility dataset: {compatibility_path}")
    print(f"Wrote merged dataset: {merged_path}")
    print(f"Wrote normalized rubrics: {rubrics_path}")
    print(f"Rows: {len(multitrait_df)}")
    print(f"Split counts: {multitrait_df['split'].value_counts(sort=False).to_dict()}")
    creativity_missing = int(multitrait_df['creativity_of_thought'].isna().sum())
    print(f"creativity_of_thought missing rows after normalization: {creativity_missing}")


if __name__ == "__main__":
    main()
