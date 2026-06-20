"""Data loading utilities for the Spaceship Titanic dataset.

The raw Kaggle files (``train.csv``, ``test.csv``, ``sample_submission.csv``)
are expected to live in ``data/raw``. Paths are resolved relative to the project
root so the functions work regardless of the current working directory.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# Project root = parent of the ``src`` directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SUBMISSIONS_DIR = PROJECT_ROOT / "data" / "submissions"

TARGET = "Transported"   # boolean target (True/False)
ID_COL = "PassengerId"   # unique passenger identifier


def load_train(path: Path | str | None = None) -> pd.DataFrame:
    """Load the training set (features + ``Transported``)."""
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
