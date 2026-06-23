"""Data loading and the fixed column schema for the Irrigation Need dataset.

The competition data is clean (no missing values, no outliers, train/test drawn
from the same distribution — see ``reports/EDA_FINDINGS.md``), so this module is
deliberately thin: it pins the schema, loads raw CSVs, and encodes the ordinal
target. Category levels are hard-coded from the EDA so that encoding is identical
across train, test, and any future split.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ID_COLUMN = "id"
TARGET_COLUMN = "Irrigation_Need"

# The target is ordinal: Low < Medium < High. Codes preserve that order.
CLASS_ORDER: tuple[str, ...] = ("Low", "Medium", "High")
CLASS_TO_CODE: dict[str, int] = {label: code for code, label in enumerate(CLASS_ORDER)}
CODE_TO_CLASS: dict[int, str] = {code: label for label, code in CLASS_TO_CODE.items()}

NUMERIC_FEATURES: tuple[str, ...] = (
    "Soil_pH",
    "Soil_Moisture",
    "Organic_Carbon",
    "Electrical_Conductivity",
    "Temperature_C",
    "Humidity",
    "Rainfall_mm",
    "Sunlight_Hours",
    "Wind_Speed_kmh",
    "Field_Area_hectare",
    "Previous_Irrigation_mm",
)

CATEGORICAL_FEATURES: tuple[str, ...] = (
    "Soil_Type",
    "Crop_Type",
    "Crop_Growth_Stage",
    "Season",
    "Irrigation_Type",
    "Water_Source",
    "Mulching_Used",
    "Region",
)

RAW_FEATURES: tuple[str, ...] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Hard-coded from the EDA. Fixing the level set (and its order) keeps categorical
# encoding stable across splits and guards against an unseen level silently
# becoming NaN.
CATEGORY_LEVELS: dict[str, tuple[str, ...]] = {
    "Soil_Type": ("Clay", "Loamy", "Sandy", "Silt"),
    "Crop_Type": ("Cotton", "Maize", "Potato", "Rice", "Sugarcane", "Wheat"),
    "Crop_Growth_Stage": ("Flowering", "Harvest", "Sowing", "Vegetative"),
    "Season": ("Kharif", "Rabi", "Zaid"),
    "Irrigation_Type": ("Canal", "Drip", "Rainfed", "Sprinkler"),
    "Water_Source": ("Groundwater", "Rainwater", "Reservoir", "River"),
    "Mulching_Used": ("No", "Yes"),
    "Region": ("Central", "East", "North", "South", "West"),
}

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def load_raw(split: str = "train", data_dir: Path | str | None = None) -> pd.DataFrame:
    """Load ``train.csv`` or ``test.csv`` with categoricals typed and ordered.

    Categorical columns are cast to ``pandas.Categorical`` with the fixed
    ``CATEGORY_LEVELS`` so downstream encoders see a consistent level set.
    """
    if split not in {"train", "test"}:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    directory = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    frame = pd.read_csv(directory / f"{split}.csv")
    return _apply_category_dtypes(frame)


def _apply_category_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    """Cast known categorical columns to ordered-level ``category`` dtype."""
    frame = frame.copy()
    for column, levels in CATEGORY_LEVELS.items():
        if column in frame.columns:
            frame[column] = pd.Categorical(frame[column], categories=levels)
            if frame[column].isna().any():
                unseen = set(pd.unique(frame[column].astype("object").dropna()))
                raise ValueError(f"{column} contains levels outside {levels}: {unseen}")
    return frame


def split_features_target(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a labelled frame into the feature matrix and the raw-label target."""
    if TARGET_COLUMN not in frame.columns:
        raise KeyError(f"{TARGET_COLUMN!r} not present; this looks like a test split")
    features = frame[list(RAW_FEATURES)].copy()
    target = frame[TARGET_COLUMN]
    return features, target


def encode_target(target: pd.Series) -> pd.Series:
    """Map ``Low``/``Medium``/``High`` to ordinal codes ``0``/``1``/``2``."""
    codes = target.map(CLASS_TO_CODE)
    if codes.isna().any():
        bad = sorted(set(target[codes.isna()]))
        raise ValueError(f"Unknown target labels: {bad}")
    return codes.astype("int8")


def decode_target(codes: "np.ndarray | pd.Series") -> np.ndarray:
    """Map integer class codes back to their string labels."""
    codes = np.asarray(codes).astype(int)
    return np.array([CODE_TO_CLASS[c] for c in codes], dtype=object)
