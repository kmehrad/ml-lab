"""Data loading + schema for the Predicting Heart Disease competition (PS S6E2).

Clean synthetic data on the classic UCI Statlog Heart schema: 630k train / 270k test,
**no missing values, no duplicate rows, no train/test drift**. The task is **binary
classification** — predict `Heart Disease` (`Presence`/`Absence`), encoded to
``y = 1`` for `Presence`. Positive rate ≈ 44.8% (near-balanced).

Loaders assert the expected schema and cache to Parquet in ``data/processed/`` for speed.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

ID = "id"
TARGET = "Heart Disease"
POSITIVE_LABEL = "Presence"
NEGATIVE_LABEL = "Absence"

NUMERIC = ["Age", "BP", "Cholesterol", "Max HR", "ST depression"]
CATEGORICAL = [
    "Sex",
    "Chest pain type",
    "FBS over 120",
    "EKG results",
    "Exercise angina",
    "Slope of ST",
    "Number of vessels fluro",
    "Thallium",
]
FEATURES = NUMERIC + CATEGORICAL


def _assert_schema(df: pd.DataFrame, *, train: bool) -> None:
    expected = [ID] + NUMERIC + CATEGORICAL + ([TARGET] if train else [])
    missing = [c for c in expected if c not in df.columns]
    assert not missing, f"missing columns: {missing} (have {list(df.columns)})"
    assert df[NUMERIC].isna().sum().sum() == 0, "unexpected NaNs in numeric features"
    assert df[CATEGORICAL].isna().sum().sum() == 0, "unexpected NaNs in categorical features"
    if train:
        assert df[TARGET].notna().all(), "NaNs in target"
        labels = set(df[TARGET].unique())
        assert labels <= {POSITIVE_LABEL, NEGATIVE_LABEL}, f"unexpected target labels: {labels}"


def _cached(name: str, loader) -> pd.DataFrame:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    cache = PROCESSED / f"{name}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    df = loader()
    df.to_parquet(cache, index=False)
    return df


def load_train() -> pd.DataFrame:
    def _load():
        df = pd.read_csv(RAW / "train.csv")
        _assert_schema(df, train=True)
        return df
    return _cached("train", _load)


def load_test() -> pd.DataFrame:
    def _load():
        df = pd.read_csv(RAW / "test.csv")
        _assert_schema(df, train=False)
        return df
    return _cached("test", _load)


def load_sample_submission() -> pd.DataFrame:
    return pd.read_csv(RAW / "sample_submission.csv")


def encode_target(df: pd.DataFrame) -> pd.Series:
    """Map the string target to ``y = 1`` for Presence, ``0`` for Absence."""
    return (df[TARGET] == POSITIVE_LABEL).astype(int)
