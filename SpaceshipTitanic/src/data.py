"""Data loading utilities for the Spaceship Titanic dataset.

The raw Kaggle files (``train.csv``, ``test.csv``, ``sample_submission.csv``)
are expected to live in ``data/raw``. Paths are resolved relative to the project
root so the functions work regardless of the current working directory.

Nothing is implemented yet — this module only defines the layout and the
functions to be filled in later.
"""
from __future__ import annotations

from pathlib import Path

# Project root = parent of the ``src`` directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SUBMISSIONS_DIR = PROJECT_ROOT / "data" / "submissions"

TARGET = "Transported"   # boolean target (True/False)
ID_COL = "PassengerId"   # unique passenger identifier


def load_train():
    """Load the training set (features + ``Transported``)."""
    # TODO: read ``RAW_DIR / "train.csv"`` and return a DataFrame.
    raise NotImplementedError


def load_test():
    """Load the test set (features only, no target)."""
    # TODO: read ``RAW_DIR / "test.csv"`` and return a DataFrame.
    raise NotImplementedError


def load_sample_submission():
    """Load Kaggle's sample submission file (for the expected format)."""
    # TODO: read ``RAW_DIR / "sample_submission.csv"`` and return a DataFrame.
    raise NotImplementedError
