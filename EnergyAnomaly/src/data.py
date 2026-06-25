"""Data loading for the LEAD energy-anomaly competition.

The competition ships pre-engineered feature files (`*_features.csv`) alongside the
minimal files. We build on the feature files. The `gte_*` columns are target-derived
and leakage-prone, so they are dropped here for honest validation.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"

ID_COLS = ["building_id", "timestamp"]
TARGET = "anomaly"
GROUP = "building_id"

# Raw provided columns we keep as model inputs (gte_* and id/string composites dropped).
META_COLS = ["site_id", "primary_use", "square_feet", "year_built", "floor_count"]
WEATHER_COLS = [
    "air_temperature", "cloud_coverage", "dew_temperature", "precip_depth_1_hr",
    "sea_level_pressure", "wind_direction", "wind_speed",
]
WEATHER_LAG_COLS = [
    "air_temperature_mean_lag7", "air_temperature_max_lag7", "air_temperature_min_lag7",
    "air_temperature_std_lag7", "air_temperature_mean_lag73", "air_temperature_max_lag73",
    "air_temperature_min_lag73", "air_temperature_std_lag73",
]
CALENDAR_COLS = [
    "hour", "weekday", "month", "hour_x", "hour_y", "month_x", "month_y",
    "weekday_x", "weekday_y", "is_holiday",
]
CATEGORICAL = ["site_id", "primary_use"]

# Columns loaded from the feature CSVs (everything we use except engineered-on-the-fly).
_BASE = ID_COLS + ["meter_reading"] + META_COLS + WEATHER_COLS + WEATHER_LAG_COLS + CALENDAR_COLS


def load_train() -> pd.DataFrame:
    cols = _BASE + [TARGET]
    df = pd.read_csv(RAW / "train_features.csv", usecols=cols, parse_dates=["timestamp"])
    return df.sort_values(ID_COLS).reset_index(drop=True)


def load_test() -> pd.DataFrame:
    cols = ["row_id"] + _BASE
    df = pd.read_csv(RAW / "test_features.csv", usecols=cols, parse_dates=["timestamp"])
    return df.sort_values(ID_COLS).reset_index(drop=True)


def load_sample_submission() -> pd.DataFrame:
    return pd.read_csv(RAW / "sample_submission.csv")
