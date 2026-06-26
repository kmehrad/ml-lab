"""Data loading and schema for the Kaggle ``nlp-getting-started`` competition.

Binary text classification: predict ``target`` (1 = real disaster, 0 = not) from a
short tweet. The pipeline is prediction-file based, so this module's only job is to
load the raw CSVs and pin the schema so column drift fails loudly.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

ID_COL = "id"
KEYWORD_COL = "keyword"
LOCATION_COL = "location"
TEXT_COL = "text"
TARGET = "target"

FEATURE_COLS = [ID_COL, KEYWORD_COL, LOCATION_COL, TEXT_COL]
TRAIN_COLS = FEATURE_COLS + [TARGET]

# Expected row counts (sanity only — warn, do not hard-fail, in case Kaggle reships).
N_TRAIN = 7613
N_TEST = 3263


def _check_schema(df: pd.DataFrame, expected: list[str], name: str) -> None:
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(
            f"{name}: missing expected column(s) {missing}; got {list(df.columns)}"
        )


def _raw_path(filename: str) -> Path:
    path = RAW_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the data first:\n"
            '  export KAGGLE_API_TOKEN="$(cat ~/.kaggle/access_token)"\n'
            "  uv run kaggle competitions download -c nlp-getting-started -p data/raw\n"
            "  unzip -o data/raw/nlp-getting-started.zip -d data/raw"
        )
    return path


def load_train() -> pd.DataFrame:
    """Load ``train.csv`` (7,613 rows) with the target column."""
    df = pd.read_csv(_raw_path("train.csv"))
    _check_schema(df, TRAIN_COLS, "train.csv")
    return df


def load_test() -> pd.DataFrame:
    """Load ``test.csv`` (3,263 rows), no target."""
    df = pd.read_csv(_raw_path("test.csv"))
    _check_schema(df, FEATURE_COLS, "test.csv")
    return df


def load_sample_submission() -> pd.DataFrame:
    """Load ``sample_submission.csv`` (id, target)."""
    df = pd.read_csv(_raw_path("sample_submission.csv"))
    _check_schema(df, [ID_COL, TARGET], "sample_submission.csv")
    return df


if __name__ == "__main__":
    tr, te = load_train(), load_test()
    print(f"train: {tr.shape}  test: {te.shape}")
    print(f"target balance: {tr[TARGET].value_counts(normalize=True).round(4).to_dict()}")
    print(f"missing (train): {tr.isna().sum().to_dict()}")
