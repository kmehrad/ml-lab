"""Data loading, the fixed column schema, and the held-out test split.

Dataset: Kaggle ULB credit-card fraud (``mlg-ulb/creditcardfraud``). A single
``creditcard.csv`` of 284,807 European card transactions over two days, of which
**492 (0.173%) are fraud** — an extreme class imbalance. Columns:

* ``Time``   — seconds elapsed since the first transaction in the file.
* ``V1``..``V28`` — anonymised principal components (PCA of the original, secret
  features); already centred/decorrelated, so they need no scaling for trees.
* ``Amount`` — transaction amount.
* ``Class``  — target: 1 = fraud, 0 = genuine.

There is **no Kaggle leaderboard or held-out test file**, so evaluation is fully
local. :func:`make_holdout_split` carves a fixed, **stratified** test set once and
persists the row indices to ``experiments/artifacts/holdout_idx.npz``; every model
trains/CV-selects on the remainder and is finally scored on that same test set,
so results are comparable across runs.

The metric is **AUPRC / average precision** (the dataset authors' recommendation
under this imbalance), so downstream code predicts and scores probabilities.
``load_raw`` calls ``_check_schema`` which raises on a column mismatch, so a bad
download fails loudly instead of silently mis-modelling.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET_COLUMN = "Class"

# Anonymised PCA components V1..V28.
PCA_FEATURES: tuple[str, ...] = tuple(f"V{i}" for i in range(1, 29))

# Continuous features alongside the PCA components.
NUMERIC_FEATURES: tuple[str, ...] = ("Time", "Amount") + PCA_FEATURES

RAW_FEATURES: tuple[str, ...] = NUMERIC_FEATURES

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "experiments" / "artifacts"
HOLDOUT_PATH = ARTIFACTS_DIR / "holdout_idx.npz"


def load_raw(data_dir: Path | str | None = None) -> pd.DataFrame:
    """Load ``creditcard.csv`` and verify the expected schema."""
    directory = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    frame = pd.read_csv(directory / "creditcard.csv")
    _check_schema(frame)
    return frame


def _check_schema(frame: pd.DataFrame) -> None:
    """Fail loudly if the data does not match the pinned schema."""
    missing = [c for c in RAW_FEATURES if c not in frame.columns]
    if missing:
        raise ValueError(
            f"creditcard.csv is missing expected feature columns {missing}. "
            f"Got columns: {list(frame.columns)}. Update src/data.py constants."
        )
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(
            f"creditcard.csv is missing target {TARGET_COLUMN!r}. "
            f"Got columns: {list(frame.columns)}."
        )


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a labelled frame into the feature matrix and the 0/1 target."""
    if TARGET_COLUMN not in frame.columns:
        raise KeyError(f"{TARGET_COLUMN!r} not present in frame")
    features = frame[list(RAW_FEATURES)].copy()
    target = frame[TARGET_COLUMN].astype("int8")
    return features, target


def make_holdout_split(
    test_size: float = 0.2,
    seed: int = 42,
    data_dir: Path | str | None = None,
    *,
    cache: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return fixed, stratified ``(train_idx, test_idx)`` row indices.

    The split is computed once and cached to ``holdout_idx.npz`` so the test set
    is identical across every training/evaluation run. Pass ``cache=False`` to
    recompute without reading or writing the cache (used in tests).
    """
    if cache and HOLDOUT_PATH.exists():
        cached = np.load(HOLDOUT_PATH)
        return cached["train_idx"], cached["test_idx"]

    frame = load_raw(data_dir)
    y = frame[TARGET_COLUMN].to_numpy()
    train_idx, test_idx = train_test_split(
        np.arange(len(y)), test_size=test_size, random_state=seed, stratify=y
    )
    train_idx.sort()
    test_idx.sort()
    if cache:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(HOLDOUT_PATH, train_idx=train_idx, test_idx=test_idx)
    return train_idx, test_idx
