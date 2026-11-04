"""Load and validate the Kaggle S6E11 CSV contract."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
ID = "id"
TARGET = "is_fraud"


def _read(name: str) -> pd.DataFrame:
    path = RAW / name
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}; download playground-series-s6e11")
    return pd.read_csv(path)


def load_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train, test, sample = (_read(n) for n in ("train.csv", "test.csv", "sample_submission.csv"))
    if train.columns.tolist()[:-1] != test.columns.tolist() or train.columns[-1] != TARGET:
        raise ValueError("unexpected train/test schema or target position")
    if sample.columns.tolist() != [ID, TARGET]:
        raise ValueError("unexpected sample submission schema")
    if any(frame[ID].duplicated().any() for frame in (train, test, sample)):
        raise ValueError("duplicate IDs")
    if len(test) != len(sample) or test[ID].tolist() != sample[ID].tolist():
        raise ValueError("test/sample ID order mismatch")
    labels = set(train[TARGET].dropna().unique())
    if labels != {0.0, 1.0} or train[TARGET].isna().any():
        raise ValueError(f"target must be complete binary 0/1; got {labels}")
    return train, test, sample


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame if c not in (ID, TARGET)]
