"""Feature engineering.

All features are derived from INPUTS only (never the target), so they are computed on the
full frame before the CV split without leakage. Features are organized into named **groups**
selected at train time (`--features base interactions`); each group is adopted only if it
clears OOF AUC above fold noise (see experiments/README.md).

Groups
------
- ``base``         : the 13 raw features, categoricals as `category` dtype (always on).
- ``interactions`` : EDA-flagged interaction terms among the strongest single features
                     (Thallium x Chest pain type, ST depression x Slope of ST). Axis-aligned
                     tree splits reach these only indirectly. Off by default — enable only if
                     it beats OOF AUC.
"""
from __future__ import annotations

import pandas as pd

from .data import CATEGORICAL, NUMERIC

INTERACTION_COLS = ["thallium_x_cp", "stdep_x_slope"]


def add_features(df: pd.DataFrame, groups=("base",)) -> pd.DataFrame:
    """Return df with the requested feature groups added. `base` is always applied."""
    groups = set(groups)
    df = df.copy()
    for c in CATEGORICAL:
        df[c] = df[c].astype("category")

    if "interactions" in groups:
        df["thallium_x_cp"] = (
            df["Thallium"].astype(int) * 10 + df["Chest pain type"].astype(int)
        ).astype("category")
        df["stdep_x_slope"] = df["ST depression"] * df["Slope of ST"].astype(int)

    return df


def feature_columns(groups=("base",)) -> list[str]:
    groups = set(groups)
    cols = list(NUMERIC) + list(CATEGORICAL)
    if "interactions" in groups:
        cols += INTERACTION_COLS
    return cols


def categorical_columns(groups=("base",)) -> list[str]:
    groups = set(groups)
    cols = list(CATEGORICAL)
    if "interactions" in groups:
        cols += ["thallium_x_cp"]
    return cols
