from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from promptaes2.config import ConfigError, DataAlignmentError, MissingColumnError


def _extract_npz_id_key(data: Any) -> str:
    if "id" in data.keys():
        return "id"
    if "ID" in data.keys():
        return "ID"
    raise DataAlignmentError("NPZ file must contain either 'id' or 'ID' key")


def _normalize_npz_id(value: Any) -> int:
    text = str(value)
    if "_" in text:
        # Handles values like ESSAY_123 or essay_123
        return int(text.split("_")[-1])
    return int(text)


def align_npz_and_csv(
    npz_path: str | Path,
    csv_path: str | Path,
    csv_id_column: str = "ID",
) -> tuple[dict[str, np.ndarray], pd.DataFrame, str]:
    data = np.load(npz_path, allow_pickle=True)
    id_key = _extract_npz_id_key(data)

    essay = pd.read_csv(csv_path)
    if csv_id_column not in essay.columns:
        available = ", ".join(essay.columns.astype(str).tolist())
        raise MissingColumnError(
            f"CSV is missing id column '{csv_id_column}'. Available columns: {available}"
        )

    npz_numeric_ids = [_normalize_npz_id(npz_id) for npz_id in data[id_key]]
    essay_ids = essay[csv_id_column].astype(int).tolist()

    id_to_indices: dict[int, list[int]] = {}
    for idx, npz_id in enumerate(npz_numeric_ids):
        id_to_indices.setdefault(npz_id, []).append(idx)

    matched_indices: list[int] = []
    matched_csv_rows: list[int] = []

    for row_idx, row_id in enumerate(essay_ids):
        candidate = id_to_indices.get(row_id)
        if candidate:
            matched_indices.append(candidate.pop(0))
            matched_csv_rows.append(row_idx)

    if not matched_indices:
        raise DataAlignmentError(
            f"No overlapping IDs between NPZ '{npz_path}' and CSV '{csv_path}'."
        )

    aligned_npz = {key: value[matched_indices] for key, value in data.items()}
    aligned_essay = essay.iloc[matched_csv_rows].copy()
    aligned_essay = aligned_essay.sort_values(csv_id_column).reset_index(drop=True)

    sorted_indices = np.argsort([_normalize_npz_id(v) for v in aligned_npz[id_key]])
    aligned_npz = {key: value[sorted_indices] for key, value in aligned_npz.items()}

    if len(aligned_essay) != len(aligned_npz[id_key]):
        raise DataAlignmentError(
            "Aligned NPZ and CSV lengths differ after sorting: "
            f"{len(aligned_npz[id_key])} vs {len(aligned_essay)}"
        )

    return aligned_npz, aligned_essay, id_key


def extract_trait_embeddings(npz_data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    embeddings = {
        key.replace("emb_", ""): value.astype(np.float32)
        for key, value in npz_data.items()
        if key.startswith("emb_")
    }
    if not embeddings:
        raise ConfigError("No embedding keys found in NPZ (expected keys starting with 'emb_').")
    return embeddings
