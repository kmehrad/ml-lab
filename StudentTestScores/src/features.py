"""Feature engineering.

All features are derived from INPUTS only (never the target), so they are computed on the
full frame before the CV split without leakage. Features are organized into named **groups**
selected at train time (`--features base ratios`); each group is adopted only if it clears
OOF RMSE above fold noise (see experiments/README.md).

Groups
------
- ``base``    : the 11 raw features, categoricals as `category` dtype (always on).
- ``ratios``  : simple interactions/ratios among the numeric features (study intensity,
                study-per-sleep, attendance×study). Axis-aligned tree splits reach these only
                indirectly. Off by default — enable only if it beats OOF RMSE.
"""
from __future__ import annotations

import pandas as pd

from .data import CATEGORICAL, NUMERIC

RATIO_COLS = ["study_x_attendance", "study_per_sleep", "attend_per_age"]


def add_features(df: pd.DataFrame, groups=("base",)) -> pd.DataFrame:
    """Return df with the requested feature groups added. `base` is always applied."""
    groups = set(groups)
    df = df.copy()
    for c in CATEGORICAL:
        df[c] = df[c].astype("category")

    if "ratios" in groups:
        df["study_x_attendance"] = df["study_hours"] * df["class_attendance"]
        df["study_per_sleep"] = df["study_hours"] / (df["sleep_hours"] + 1.0)
        df["attend_per_age"] = df["class_attendance"] / (df["age"] + 1.0)

    return df


def feature_columns(groups=("base",)) -> list[str]:
    groups = set(groups)
    cols = list(NUMERIC) + list(CATEGORICAL)
    if "ratios" in groups:
        cols += RATIO_COLS
    return cols


def categorical_columns(groups=("base",)) -> list[str]:
    # No engineered categoricals yet; the raw categoricals are always the categorical set.
    return list(CATEGORICAL)
