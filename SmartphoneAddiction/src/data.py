from __future__ import annotations

from pathlib import Path
import pandas as pd

TARGET = "addicted_label"
ID = "id"
FEATURES = [
    "age", "daily_screen_time_hours", "social_media_hours", "gaming_hours",
    "work_study_hours", "sleep_hours", "notifications_per_day",
    "app_opens_per_day", "weekend_screen_time", "gender", "stress_level",
    "academic_work_impact",
]
CATEGORICAL = ["gender", "stress_level", "academic_work_impact"]
NUMERIC = [c for c in FEATURES if c not in CATEGORICAL]


def _validate(df: pd.DataFrame, train: bool) -> None:
    expected = {ID, *FEATURES, *([TARGET] if train else [])}
    missing, extra = expected - set(df), set(df) - expected
    if missing or extra:
        raise ValueError(f"schema mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    if df[ID].isna().any() or not df[ID].is_unique:
        raise ValueError("id must be non-null and unique")
    if train and (df[TARGET].isna().any() or not set(df[TARGET].unique()) <= {0, 1}):
        raise ValueError("target must contain only 0/1 and no missing values")


def load_data(data_dir: str | Path = "data/raw", cache: bool = True):
    data_dir = Path(data_dir)
    processed = data_dir.parent / "processed"
    train_pq, test_pq = processed / "train.parquet", processed / "test.parquet"
    if cache and train_pq.exists() and test_pq.exists():
        train, test = pd.read_parquet(train_pq), pd.read_parquet(test_pq)
    else:
        train = pd.read_csv(data_dir / "train.csv")
        test = pd.read_csv(data_dir / "test.csv")
        if cache:
            processed.mkdir(parents=True, exist_ok=True)
            train.to_parquet(train_pq, index=False)
            test.to_parquet(test_pq, index=False)
    _validate(train, True)
    _validate(test, False)
    return train, test


ORIGINAL_ALIASES = {
    "Age": "age", "Daily_Screen_Time_Hours": "daily_screen_time_hours",
    "Social_Media_Hours": "social_media_hours", "Gaming_Hours": "gaming_hours",
    "Work_Study_Hours": "work_study_hours", "Sleep_Hours": "sleep_hours",
    "Notifications_Per_Day": "notifications_per_day", "App_Opens_Per_Day": "app_opens_per_day",
    "Weekend_Screen_Time": "weekend_screen_time", "Gender": "gender",
    "Stress_Level": "stress_level", "Academic_Work_Impact": "academic_work_impact",
    "Addicted_Label": TARGET,
}


def map_original(df: pd.DataFrame) -> pd.DataFrame:
    """Map source data and intentionally discard IDs and derived addiction level."""
    out = df.rename(columns=ORIGINAL_ALIASES)
    required = FEATURES + [TARGET]
    missing = set(required) - set(out)
    if missing:
        raise ValueError(f"original data missing mapped columns: {sorted(missing)}")
    out = out[required].copy()
    if not set(out[TARGET].dropna().unique()) <= {0, 1}:
        raise ValueError("original target must be binary")
    return out
