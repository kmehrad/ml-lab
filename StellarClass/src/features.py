"""Leakage-safe, stateless feature engineering applied identically to train and test.

The raw data is clean (no missing values, no sentinels), so feature work is about giving the models
informative transforms: photometric **color indices** (differences of adjacent bands) and
**redshift transforms** (redshift is the dominant separator of STAR vs GALAXY vs QSO). Categorical
columns are returned as pandas `category` dtype so LightGBM/CatBoost can use them natively.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as D

# Adjacent-band color indices (standard astronomy features).
COLOR_PAIRS = [("u", "g"), ("g", "r"), ("r", "i"), ("i", "z")]
COLOR_COLS = [f"{a}_{b}" for a, b in COLOR_PAIRS]
# A couple of wider baselines that help separate the SED shape.
WIDE_PAIRS = [("u", "r"), ("g", "i"), ("u", "z")]
WIDE_COLS = [f"{a}_{b}" for a, b in WIDE_PAIRS]

REDSHIFT_COLS = ["redshift_log1p", "redshift_is_neg"]

NUMERIC_FEATURES = D.NUMERIC_COLS + COLOR_COLS + WIDE_COLS + REDSHIFT_COLS
CATEGORICAL_FEATURES = D.CATEGORICAL_COLS
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a feature matrix (ALL_FEATURES columns) from a raw train/test frame."""
    out = df[D.NUMERIC_COLS].copy()

    for a, b in COLOR_PAIRS + WIDE_PAIRS:
        out[f"{a}_{b}"] = df[a] - df[b]

    # redshift can be marginally negative (~ -0.01); log1p is well-defined for > -1.
    out["redshift_log1p"] = np.log1p(df["redshift"].clip(lower=-0.999))
    out["redshift_is_neg"] = (df["redshift"] < 0).astype("int8")

    for c in D.CATEGORICAL_COLS:
        out[c] = df[c].astype("category")

    return out[ALL_FEATURES]
