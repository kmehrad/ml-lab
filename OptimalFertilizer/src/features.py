"""Feature engineering.

All features are derived from INPUTS only (never the target), so they are computed on the
full frame before the CV split without leakage.

Step 2 is the **baseline passthrough**: the 8 raw features, with categoricals cast to the
`category` dtype (native GBDT handling). Engineered features (NPK ratios/sums, env bins,
Soil×Crop combo) arrive in Step 4, each gated on OOF MAP@3 — EDA showed `Soil×Crop` is the
strongest single signal, so it is the first candidate.
"""
from __future__ import annotations

import pandas as pd

from .data import CATEGORICAL, NUMERIC


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with model-ready columns (baseline: raw features, categoricals as category)."""
    df = df.copy()
    for c in CATEGORICAL:
        df[c] = df[c].astype("category")
    return df


def feature_columns() -> list[str]:
    return NUMERIC + CATEGORICAL
