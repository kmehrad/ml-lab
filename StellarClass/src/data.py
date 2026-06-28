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


def encode_target(y: pd.Series) -> np.ndarray:
    """Map class strings -> ints using the fixed CLASSES order."""
    return y.map(CLASS_TO_INT).to_numpy()


def decode_target(y_int: np.ndarray) -> np.ndarray:
    """Map ints -> class strings."""
    return np.array([INT_TO_CLASS[int(i)] for i in y_int])
