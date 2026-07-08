"""Data loading + schema for the Predicting Student Test Scores competition (PS S6E1).

Clean synthetic data: 630k train / 270k test, **no missing values**. The task is
**regression** — predict the continuous ``exam_score`` (range ~19.6–100). Four numeric
features and seven low-cardinality categoricals; native GBDT categorical handling is fine.

Loaders assert the expected schema and cache to Parquet in ``data/processed/`` for speed.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

ID = "id"
TARGET = "exam_score"
NUMERIC = ["age", "study_hours", "class_attendance", "sleep_hours"]
CATEGORICAL = ["gender", "course", "internet_access", "sleep_quality",
               "study_method", "facility_rating", "exam_difficulty"]
FEATURES = NUMERIC + CATEGORICAL

# Observed train-target range — used to clip predictions at submission time.
TARGET_MIN, TARGET_MAX = 19.6, 100.0


def _assert_schema(df: pd.DataFrame, *, train: bool) -> None:
    expected = [ID] + NUMERIC + CATEGORICAL + ([TARGET] if train else [])
    missing = [c for c in expected if c not in df.columns]
    assert not missing, f"missing columns: {missing} (have {list(df.columns)})"
    assert df[NUMERIC].isna().sum().sum() == 0, "unexpected NaNs in numeric features"
    assert df[CATEGORICAL].isna().sum().sum() == 0, "unexpected NaNs in categorical features"
    if train:
        assert df[TARGET].notna().all(), "NaNs in target"
        lo, hi = df[TARGET].min(), df[TARGET].max()
        assert 0.0 < lo and hi <= 100.0 + 1e-6, f"target out of expected range: [{lo}, {hi}]"


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
