"""Feature engineering.

All features derive from INPUTS only (never the target), so they are computed on the full frame
before the CV split without leakage. Features are organized into named **groups** selected at train
time (``--features base ratio ...``); each group is adopted only if it clears OOF balanced-accuracy
above fold noise (see experiments/README.md).

Missing values are kept as-is: the GBDTs split on NaN natively, and categoricals are ``category``
dtype so NaN is simply an extra level. Missingness may itself be informative here (~65% of rows have
at least one missing field), which the ``missflag`` group exposes explicitly.

Groups
------
- ``base``     : the 7 numeric + 6 categorical raw features (always on).
- ``ratio``    : physiologically meaningful ratios/interactions axis-aligned splits reach only
                 indirectly (steps-per-calorie, exercise-per-step, bmi×heart_rate, ...).
- ``missflag`` : per-column ``_isna`` indicator columns (missingness as signal).
"""
from __future__ import annotations

import pandas as pd

from .data import CATEGORICAL, NUMERIC

RATIO_COLS = [
    "steps_per_calorie",
    "calorie_per_step",
    "exercise_per_step",
    "water_per_bmi",
    "bmi_x_heart_rate",
    "sleep_x_exercise",
]
MISSFLAG_COLS = [f"{c}_isna" for c in NUMERIC + CATEGORICAL]


def add_features(df: pd.DataFrame, groups=("base",)) -> pd.DataFrame:
    """Return df with the requested feature groups added. `base` is always applied."""
    groups = set(groups)
    df = df.copy()
    for c in CATEGORICAL:
        df[c] = df[c].astype("category")

    if "ratio" in groups:
        eps = 1.0
        df["steps_per_calorie"] = df["step_count"] / (df["calorie_expenditure"] + eps)
        df["calorie_per_step"] = df["calorie_expenditure"] / (df["step_count"] + eps)
        df["exercise_per_step"] = df["exercise_duration"] / (df["step_count"] + eps)
        df["water_per_bmi"] = df["water_intake"] / (df["bmi"] + eps)
        df["bmi_x_heart_rate"] = df["bmi"] * df["heart_rate"]
        df["sleep_x_exercise"] = df["sleep_duration"] * df["exercise_duration"]

    if "missflag" in groups:
        for c in NUMERIC + CATEGORICAL:
            df[f"{c}_isna"] = df[c].isna().astype("int8")

    return df


def feature_columns(groups=("base",)) -> list[str]:
    groups = set(groups)
    cols = list(NUMERIC) + list(CATEGORICAL)
    if "ratio" in groups:
        cols += RATIO_COLS
    if "missflag" in groups:
        cols += MISSFLAG_COLS
    return cols


def categorical_columns(groups=("base",)) -> list[str]:
    # engineered groups add only numeric columns; categoricals are the raw six.
    return list(CATEGORICAL)
