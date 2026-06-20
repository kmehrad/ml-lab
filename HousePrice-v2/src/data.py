"""Data loading utilities for the House Prices dataset.

The raw Kaggle files (``train.csv``, ``test.csv``, ``sample_submission.csv``)
are expected to live in ``data/raw``. Paths are resolved relative to the
project root so the functions work regardless of the current working
directory.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# Project root = parent of the ``src`` directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SUBMISSIONS_DIR = PROJECT_ROOT / "data" / "submissions"

TARGET = "SalePrice"
ID_COL = "Id"


def load_train(path: Path | str | None = None) -> pd.DataFrame:
    """Load the training set (features + ``SalePrice``)."""
    path = Path(path) if path is not None else RAW_DIR / "train.csv"
    return pd.read_csv(path)


def load_test(path: Path | str | None = None) -> pd.DataFrame:
    """Load the test set (features only, no target)."""
    path = Path(path) if path is not None else RAW_DIR / "test.csv"
    return pd.read_csv(path)


def load_sample_submission(path: Path | str | None = None) -> pd.DataFrame:
    """Load Kaggle's sample submission file (for the expected format)."""
    path = Path(path) if path is not None else RAW_DIR / "sample_submission.csv"
    return pd.read_csv(path)


def split_features_target(df: pd.DataFrame):
    """Split a training frame into ``(X, y)``, dropping ``Id``."""
    y = df[TARGET]
    X = df.drop(columns=[c for c in (TARGET, ID_COL) if c in df.columns])
    return X, y


def make_submission(
    ids: pd.Series,
    predictions,
    filename: str = "submission.csv",
) -> Path:
    """Write a Kaggle-ready submission to ``data/submissions``.

    Returns the path of the written file.
    """
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SUBMISSIONS_DIR / filename
    sub = pd.DataFrame({ID_COL: ids.values, TARGET: predictions})
    sub.to_csv(out_path, index=False)
    return out_path
