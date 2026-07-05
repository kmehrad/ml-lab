"""Data loading + schema for the Predicting Student Health Risk competition (PS S6E7).

690,088 train / 295,753 test rows. Target ``health_condition`` has 3 classes, heavily imbalanced
(``at-risk`` ~86%, ``unhealthy`` ~8%, ``fit`` ~6%). 7 numeric behavioural features + 6 low-cardinality
(3-level) categoricals. **Missing values in nearly every column** (~449k cells) — we keep NaNs and let
the GBDTs handle them natively; categoricals are ``category`` dtype so NaN is simply an extra level.

Loaders assert the expected schema and cache to Parquet in ``data/processed/`` for speed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
ORIGINAL = ROOT / "data" / "original" / "student_health_dataset_50k.csv"

ID = "id"
TARGET = "health_condition"
CATEGORICAL = [
    "diet_type",
    "stress_level",
    "sleep_quality",
    "physical_activity_level",
    "smoking_alcohol",
    "gender",
]
NUMERIC = [
    "sleep_duration",
    "heart_rate",
    "bmi",
    "calorie_expenditure",
    "step_count",
    "exercise_duration",
    "water_intake",
]
FEATURES = NUMERIC + CATEGORICAL

# 3 target classes, in canonical (sorted) order — the column order of every proba matrix.
CLASSES = ["at-risk", "fit", "unhealthy"]
N_CLASSES = len(CLASSES)
_CODE = {c: i for i, c in enumerate(CLASSES)}


def encode_target(labels) -> np.ndarray:
    """Map health_condition strings to canonical integer class codes (0..2)."""
    codes = pd.Series(labels).map(_CODE)
    assert codes.notna().all(), f"unknown target labels: {set(pd.Series(labels).unique()) - set(CLASSES)}"
    return np.asarray(codes, dtype=np.int64)


def decode_target(codes) -> np.ndarray:
    """Map integer class codes (0..2) back to health_condition strings."""
    return np.asarray([CLASSES[int(c)] for c in codes], dtype=object)


def _assert_schema(df: pd.DataFrame, *, train: bool) -> None:
    expected = [ID] + NUMERIC + CATEGORICAL + ([TARGET] if train else [])
    missing = [c for c in expected if c not in df.columns]
    assert not missing, f"missing columns: {missing} (have {list(df.columns)})"
    if train:
        assert set(df[TARGET].dropna().unique()) == set(CLASSES), \
            f"unexpected target classes: {sorted(set(df[TARGET].dropna().unique()))}"
        assert df[TARGET].notna().all(), "unexpected NaN in target"


def _typed(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric features as float; categoricals as pandas ``category`` (NaN kept as a level)."""
    for c in NUMERIC:
        df[c] = df[c].astype("float32")
    for c in CATEGORICAL:
        df[c] = df[c].astype("category")
    return df


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
        return _typed(df)
    return _cached("train", _load)


def load_test() -> pd.DataFrame:
    def _load():
        df = pd.read_csv(RAW / "test.csv")
        _assert_schema(df, train=False)
        return _typed(df)
    return _cached("test", _load)


def load_sample_submission() -> pd.DataFrame:
    return pd.read_csv(RAW / "sample_submission.csv")


def load_original() -> pd.DataFrame:
    """The ~50k-row real seed dataset the competition was synthesized from (schema-matched).

    Same 13 feature columns + ``health_condition`` target as the competition; drops the extra
    ``student_id``/``timestamp`` columns. Used to augment each *training* fold only (never
    validation/test) via ``train.py --augment``. No caching — it is small and loaded once per run.
    """
    df = pd.read_csv(ORIGINAL)
    df = df.drop(columns=[c for c in ("student_id", "timestamp") if c in df.columns])
    df[ID] = range(len(df))                       # synthetic id (original rows never hit submission)
    _assert_schema(df, train=True)
    return _typed(df)
