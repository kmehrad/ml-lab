"""Data loading and the fixed column schema for the Diabetes dataset.

Competition: Kaggle Playground Series S5E12 — Diabetes Prediction Challenge.
The data is synthetic (700k train / 300k test rows, no missing values). Features
are a mix of continuous clinical/lifestyle measurements, three binary history
flags, three ordered categoricals, and three nominal categoricals. The target
``diagnosed_diabetes`` is 0/1.

The metric is **ROC-AUC** on the positive-class probability, so downstream code
predicts and scores probabilities, never hard labels.

``load_raw`` calls ``_check_schema`` which raises if the downloaded CSV columns
differ, so a schema mismatch fails loudly instead of silently mis-modelling.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ID_COLUMN = "id"
TARGET_COLUMN = "diagnosed_diabetes"

# Continuous / count measurements.
NUMERIC_FEATURES: tuple[str, ...] = (
    "age",
    "alcohol_consumption_per_week",
    "physical_activity_minutes_per_week",
    "diet_score",
    "sleep_hours_per_day",
    "screen_time_hours_per_day",
    "bmi",
    "waist_to_hip_ratio",
    "systolic_bp",
    "diastolic_bp",
    "heart_rate",
    "cholesterol_total",
    "hdl_cholesterol",
    "ldl_cholesterol",
    "triglycerides",
)

# Binary 0/1 history flags.
BINARY_FEATURES: tuple[str, ...] = (
    "family_history_diabetes",
    "hypertension_history",
    "cardiovascular_history",
)

# Ordered categoricals: mapped to integer codes (low -> high) in the feature step.
ORDINAL_LEVELS: dict[str, tuple[str, ...]] = {
    "education_level": ("No formal", "Highschool", "Graduate", "Postgraduate"),
    "income_level": ("Low", "Lower-Middle", "Middle", "Upper-Middle", "High"),
    "smoking_status": ("Never", "Former", "Current"),
}
ORDINAL_FEATURES: tuple[str, ...] = tuple(ORDINAL_LEVELS)

# Nominal categoricals: one-hot encoded in preprocessing.
NOMINAL_FEATURES: tuple[str, ...] = ("gender", "ethnicity", "employment_status")

CATEGORICAL_FEATURES: tuple[str, ...] = ORDINAL_FEATURES + NOMINAL_FEATURES
RAW_FEATURES: tuple[str, ...] = (
    NUMERIC_FEATURES + BINARY_FEATURES + ORDINAL_FEATURES + NOMINAL_FEATURES
)

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def load_raw(split: str = "train", data_dir: Path | str | None = None) -> pd.DataFrame:
    """Load ``train.csv`` or ``test.csv`` and verify the expected schema."""
    if split not in {"train", "test"}:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    directory = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    frame = pd.read_csv(directory / f"{split}.csv")
    _check_schema(frame, split)
    return frame


def _check_schema(frame: pd.DataFrame, split: str) -> None:
    """Fail loudly if the downloaded data does not match the pinned schema."""
    missing = [c for c in RAW_FEATURES if c not in frame.columns]
    if missing:
        raise ValueError(
            f"{split}.csv is missing expected feature columns {missing}. "
            f"Got columns: {list(frame.columns)}. Update src/data.py constants."
        )
    if split == "train" and TARGET_COLUMN not in frame.columns:
        raise ValueError(
            f"train.csv is missing target {TARGET_COLUMN!r}. "
            f"Got columns: {list(frame.columns)}. Update TARGET_COLUMN in src/data.py."
        )


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a labelled frame into the feature matrix and the 0/1 target."""
    if TARGET_COLUMN not in frame.columns:
        raise KeyError(f"{TARGET_COLUMN!r} not present; this looks like a test split")
    features = frame[list(RAW_FEATURES)].copy()
    target = frame[TARGET_COLUMN].astype("int8")
    return features, target


def load_original(path: Path | str | None = None) -> pd.DataFrame:
    """Load an optional external dataset for training augmentation.

    The CSV must already match this competition's schema (same feature columns
    and a ``diagnosed_diabetes`` target). It is loaded only when ``--use-original``
    is passed; an ``id`` column is synthesised with negative ids to avoid
    collisions with the competition rows.
    """
    file = Path(path) if path is not None else DEFAULT_DATA_DIR / "original.csv"
    frame = pd.read_csv(file)
    if TARGET_COLUMN not in frame.columns:
        for alt in ("Outcome", "diabetes", "Diabetes_binary"):
            if alt in frame.columns:
                frame = frame.rename(columns={alt: TARGET_COLUMN})
                break
    if ID_COLUMN not in frame.columns:
        frame[ID_COLUMN] = np.arange(-len(frame), 0)
    _check_schema(frame, "train")
    return frame
