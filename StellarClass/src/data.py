"""Schema constants and loaders for the Stellar Class competition (S6E6).

The pipeline communicates through saved prediction arrays; this module is the single source of truth
for the raw schema and the label<->int encoding so every model agrees on column names and class order.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

ID_COL = "id"
TARGET = "class"

NUMERIC_COLS = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]
CATEGORICAL_COLS = ["spectral_type", "galaxy_population"]
FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS

# Fixed class order so OOF/test probability columns are consistent across every model.
CLASSES = ["GALAXY", "QSO", "STAR"]
CLASS_TO_INT = {c: i for i, c in enumerate(CLASSES)}
INT_TO_CLASS = {i: c for c, i in CLASS_TO_INT.items()}

TRAIN_COLS = [ID_COL] + FEATURE_COLS + [TARGET]
TEST_COLS = [ID_COL] + FEATURE_COLS


def _check_columns(df: pd.DataFrame, expected: list[str], name: str) -> None:
    actual = list(df.columns)
    if actual != expected:
        raise ValueError(
            f"{name} schema mismatch.\n  expected: {expected}\n  actual:   {actual}"
        )


def load_train() -> pd.DataFrame:
    df = pd.read_csv(RAW / "train.csv")
    _check_columns(df, TRAIN_COLS, "train.csv")
    bad = set(df[TARGET].unique()) - set(CLASSES)
    if bad:
        raise ValueError(f"unexpected class labels: {bad}")
    return df


def load_test() -> pd.DataFrame:
    df = pd.read_csv(RAW / "test.csv")
    _check_columns(df, TEST_COLS, "test.csv")
    return df


def load_sample_submission() -> pd.DataFrame:
    df = pd.read_csv(RAW / "sample_submission.csv")
    _check_columns(df, [ID_COL, TARGET], "sample_submission.csv")
    return df


ORIGINAL_FILE = "star_classification.csv"


def load_original() -> pd.DataFrame:
    """Load the public SDSS17 dataset, aligned to the competition schema.

    The original shares alpha/delta/ugriz/redshift/class but lacks the two synthetic categoricals
    (`spectral_type`, `galaxy_population`), which are set to NaN. A known bad row has ugriz == -9999
    sentinels; those are dropped. Returns columns == TRAIN_COLS so add_features works unchanged.
    """
    df = pd.read_csv(RAW / ORIGINAL_FILE)
    df.columns = [c.strip() for c in df.columns]
    keep = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift", "class"]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(f"original dataset missing columns: {missing}; got {list(df.columns)}")
    df = df[keep].copy()

    # drop sentinel/garbage photometry rows (the dataset has a -9999 outlier row)
    for b in ["u", "g", "r", "i", "z"]:
        df = df[(df[b] > 0) & (df[b] < 40)]
    bad = set(df["class"].unique()) - set(CLASSES)
    if bad:
        raise ValueError(f"unexpected original class labels: {bad}")

    df[CATEGORICAL_COLS[0]] = np.nan
    df[CATEGORICAL_COLS[1]] = np.nan
    df[ID_COL] = -1  # not used; original rows only ever go into training folds
    return df[TRAIN_COLS].reset_index(drop=True)


def encode_target(y: pd.Series) -> np.ndarray:
    """Map class strings -> ints using the fixed CLASSES order."""
    return y.map(CLASS_TO_INT).to_numpy()


def decode_target(y_int: np.ndarray) -> np.ndarray:
    """Map ints -> class strings."""
    return np.array([INT_TO_CLASS[int(i)] for i in y_int])
