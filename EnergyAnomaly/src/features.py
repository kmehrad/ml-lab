"""Feature engineering.

All features here are derived from INPUTS only (meter_reading, weather, calendar,
metadata) — never the target — so they are computed on the full frame before the CV
split without leakage. Per-building temporal features are computed within each
building, ordered by timestamp.

EDA-driven additions (see reports/EDA_FINDINGS.md):
- meter_reading == 1.0 is a stuck/flatlined meter (99.97% anomaly, ~46% of all
  anomalies). Captured as a flag + flatline run-length so it generalizes to unseen
  test buildings rather than as a hard ==1.0 override.
- meter_reading missing (~6.2%) is never an anomaly -> flagged.
- cloud_coverage == 255 is a sentinel for missing -> restored to NaN + flag.
- wind_direction is circular -> sin/cos encoding (raw degrees are meaningless to splits).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import (
    CALENDAR_COLS, CATEGORICAL, META_COLS, WEATHER_COLS, WEATHER_LAG_COLS,
)

LAGS = [1, 24, 168]
_CLOUD_SENTINEL = 255


def _run_length(s: pd.Series) -> pd.Series:
    """Length-so-far of the consecutive constant-value run (within an ordered series)."""
    filled = s.fillna(np.inf)  # treat consecutive NaNs as a constant run
    new_run = filled.ne(filled.shift())
    run_id = new_run.cumsum()
    return filled.groupby(run_id).cumcount() + 1


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with engineered columns added (expects sorted by building_id, timestamp)."""
    df = df.copy()
    r = df["meter_reading"]
    g = df.groupby("building_id", sort=False)["meter_reading"]

    # --- meter-reading signal & known artifacts ---
    df["reading_log"] = np.log1p(r.clip(lower=0))
    df["is_reading_missing"] = r.isna().astype("int8")
    df["is_reading_one"] = (r == 1.0).astype("int8")
    df["is_reading_zero"] = (r == 0.0).astype("int8")
    df["flatline_run"] = df.groupby("building_id", sort=False)["meter_reading"].transform(_run_length)

    # --- per-building lag / value-change features ---
    for lag in LAGS:
        df[f"lag_{lag}"] = g.shift(lag)
        df[f"diff_{lag}"] = r - df[f"lag_{lag}"]
        df[f"absdiff_{lag}"] = df[f"diff_{lag}"].abs()
    roll = g.transform(lambda x: x.shift(1).rolling(24, min_periods=4).mean())
    rstd = g.transform(lambda x: x.shift(1).rolling(24, min_periods=4).std())
    df["roll24_mean"] = roll
    df["roll24_std"] = rstd
    df["dev_roll24"] = (r - roll) / (rstd + 1e-6)  # z-score vs recent local behaviour

    # --- weather fixes ---
    df["cloud_missing"] = (df["cloud_coverage"] == _CLOUD_SENTINEL).astype("int8")
    df.loc[df["cloud_coverage"] == _CLOUD_SENTINEL, "cloud_coverage"] = np.nan
    rad = np.deg2rad(df["wind_direction"].fillna(0))
    df["wind_dir_sin"] = np.sin(rad)
    df["wind_dir_cos"] = np.cos(rad)

    for c in CATEGORICAL:
        df[c] = df[c].astype("category")
    return df


# Final model input columns (everything except ids/target and the raw circular wind_direction).
ENGINEERED = (
    ["reading_log", "is_reading_missing", "is_reading_one", "is_reading_zero", "flatline_run"]
    + [f"lag_{l}" for l in LAGS] + [f"diff_{l}" for l in LAGS] + [f"absdiff_{l}" for l in LAGS]
    + ["roll24_mean", "roll24_std", "dev_roll24", "cloud_missing", "wind_dir_sin", "wind_dir_cos"]
)


def feature_columns() -> list[str]:
    base_weather = [c for c in WEATHER_COLS if c != "wind_direction"]
    return (
        ["meter_reading"] + META_COLS + base_weather + WEATHER_LAG_COLS
        + CALENDAR_COLS + ENGINEERED
    )
