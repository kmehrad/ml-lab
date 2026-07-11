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


ORIGINAL_RAW = ROOT / "data" / "original" / "Heart_disease_statlog.csv"


def load_original() -> pd.DataFrame:
    """Load the 270-row UCI Statlog Heart source dataset, remapped onto this competition's
    schema/encodings for use as training-only augmentation (see ``run_cv``'s ``augment`` arg).

    The Kaggle mirror (`ritwikb3/heart-disease-statlog`) uses 0-indexed Cleveland-style codes
    for a few columns that this competition encodes 1-indexed/Statlog-style; confirmed by
    comparing marginal value sets and target-correlation signs against ``data/raw/train.csv``:
    - ``cp`` 0-3 -> ``Chest pain type`` 1-4 (offset +1)
    - ``slope`` 0-2 -> ``Slope of ST`` 1-3 (offset +1)
    - ``thal`` 1/2/3 -> ``Thallium`` 3/6/7 (rank-preserving remap; both have exactly 3 levels)
    - ``target`` 1 -> `Presence` (positive correlation with cp/oldpeak/ca, negative with
      thalach, matches the competition's known feature-target signs)
    """
    def _load():
        df = pd.read_csv(ORIGINAL_RAW)
        df.columns = [c.strip().lstrip("﻿") for c in df.columns]
        out = pd.DataFrame({
            "Age": df["age"],
            "Sex": df["sex"],
            "Chest pain type": df["cp"] + 1,
            "BP": df["trestbps"],
            "Cholesterol": df["chol"],
            "FBS over 120": df["fbs"],
            "EKG results": df["restecg"],
            "Max HR": df["thalach"],
            "Exercise angina": df["exang"],
            "ST depression": df["oldpeak"],
            "Slope of ST": df["slope"] + 1,
            "Number of vessels fluro": df["ca"],
            "Thallium": df["thal"].map({1: 3, 2: 6, 3: 7}),
            TARGET: df["target"].map({1: POSITIVE_LABEL, 0: NEGATIVE_LABEL}),
        })
        assert out["Thallium"].isna().sum() == 0, "unmapped thal level"
        _assert_schema(out.assign(**{ID: range(len(out))}), train=True)
        return out
    return _cached("original", _load)


def encode_target(df: pd.DataFrame) -> pd.Series:
    """Map the string target to ``y = 1`` for Presence, ``0`` for Absence."""
    return (df[TARGET] == POSITIVE_LABEL).astype(int)
