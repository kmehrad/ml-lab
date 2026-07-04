"""Feature engineering.

All features are derived from INPUTS only (never the target), so they are computed on the
full frame before the CV split without leakage. Features are organized into named **groups**
selected at train time (`--features base soilcross npk ...`); each group is adopted only if it
clears OOF MAP@3 above fold noise (see experiments/README.md).

Groups
------
- ``base``      : the 8 raw features, categoricals as `category` dtype (always on).
- ``soilcross`` : ``Soil×Crop`` combined categorical (55 levels) — the strongest single signal
                  measured in EDA.
- ``npk``       : NPK sum, pairwise differences, and ratios (expose combinations that axis-aligned
                  tree splits reach only indirectly).
- ``env``       : environment×NPK and env×env products (Moisture had the highest MI in EDA).
"""
from __future__ import annotations

import pandas as pd

from .data import CATEGORICAL, NUMERIC

SOILCROSS_COL = "Soil_Crop"
NPK_COLS = ["NPK_sum", "N_minus_P", "N_minus_K", "P_minus_K", "N_over_P", "N_over_K", "K_over_P"]
ENV_COLS = ["Temp_x_Moisture", "Humidity_x_Moisture", "Moisture_x_N", "Temp_x_Humidity"]


def add_features(df: pd.DataFrame, groups=("base",)) -> pd.DataFrame:
    """Return df with the requested feature groups added. `base` is always applied."""
    groups = set(groups)
    df = df.copy()
    for c in CATEGORICAL:
        df[c] = df[c].astype("category")

    if "soilcross" in groups:
        df[SOILCROSS_COL] = (df["Soil Type"].astype(str) + "_" + df["Crop Type"].astype(str)) \
            .astype("category")

    if "npk" in groups:
        n, k, p = df["Nitrogen"], df["Potassium"], df["Phosphorous"]
        df["NPK_sum"] = n + k + p
        df["N_minus_P"] = n - p
        df["N_minus_K"] = n - k
        df["P_minus_K"] = p - k
        df["N_over_P"] = n / (p + 1.0)
        df["N_over_K"] = n / (k + 1.0)
        df["K_over_P"] = k / (p + 1.0)

    if "env" in groups:
        df["Temp_x_Moisture"] = df["Temparature"] * df["Moisture"]
        df["Humidity_x_Moisture"] = df["Humidity"] * df["Moisture"]
        df["Moisture_x_N"] = df["Moisture"] * df["Nitrogen"]
        df["Temp_x_Humidity"] = df["Temparature"] * df["Humidity"]

    return df


def feature_columns(groups=("base",)) -> list[str]:
    groups = set(groups)
    cols = list(NUMERIC) + list(CATEGORICAL)
    if "soilcross" in groups:
        cols.append(SOILCROSS_COL)
    if "npk" in groups:
        cols += NPK_COLS
    if "env" in groups:
        cols += ENV_COLS
    return cols


def categorical_columns(groups=("base",)) -> list[str]:
    groups = set(groups)
    cols = list(CATEGORICAL)
    if "soilcross" in groups:
        cols.append(SOILCROSS_COL)
    return cols
