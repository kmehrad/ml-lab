"""Data loading + schema for the Predicting Optimal Fertilizers competition (PS S5E6).

Clean synthetic data: 750k train / 250k test, no missing values, all-integer numeric
features. NB the temperature column is misspelled ``Temparature`` in the raw CSVs; we keep
the literal name. The original UCI seed dataset (``data/original/``) additionally has a
trailing space in ``Humidity `` — stripped on load.

Loaders assert the expected schema and cache to Parquet in ``data/processed/`` for speed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
ORIGINAL = ROOT / "data" / "original" / "Fertilizer Prediction.csv"

ID = "id"
TARGET = "Fertilizer Name"
CATEGORICAL = ["Soil Type", "Crop Type"]
# 'Temparature' misspelling is in the raw data — do not "fix" it.
NUMERIC = ["Temparature", "Humidity", "Moisture", "Nitrogen", "Potassium", "Phosphorous"]
FEATURES = NUMERIC + CATEGORICAL

# 7 fertilizer classes, in canonical (sorted) order — the column order of every proba matrix.
CLASSES = ["10-26-26", "14-35-14", "17-17-17", "20-20", "28-28", "DAP", "Urea"]
N_CLASSES = len(CLASSES)
_CODE = {c: i for i, c in enumerate(CLASSES)}


def encode_target(labels) -> np.ndarray:
    """Map fertilizer name strings to canonical integer class codes (0..6)."""
    return np.asarray(pd.Series(labels).map(_CODE), dtype=np.int64)


def _assert_schema(df: pd.DataFrame, *, train: bool) -> None:
    expected = [ID] + NUMERIC + CATEGORICAL + ([TARGET] if train else [])
    missing = [c for c in expected if c not in df.columns]
    assert not missing, f"missing columns: {missing} (have {list(df.columns)})"
    assert df[NUMERIC].isna().sum().sum() == 0, "unexpected NaNs in numeric features"
    if train:
        assert set(df[TARGET].unique()) == set(CLASSES), \
            f"unexpected target classes: {sorted(set(df[TARGET].unique()))}"


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


def load_original() -> pd.DataFrame:
    """The 99-row UCI seed dataset (schema-matched; for the Step 4 augmentation test)."""
    df = pd.read_csv(ORIGINAL)
    df.columns = [c.strip() for c in df.columns]   # 'Humidity ' has a trailing space
    _assert_schema(df.assign(**{ID: range(len(df))}), train=True)
    return df
