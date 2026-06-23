"""Domain feature engineering for the Diabetes dataset.

The features encode well-established type-2 diabetes / metabolic-syndrome
relationships from the raw measurements:

* Lipid ratios — ``triglycerides / HDL`` is a strong surrogate for insulin
  resistance; ``total/HDL`` and ``LDL/HDL`` are atherogenic indices; non-HDL
  cholesterol captures total atherogenic burden.
* Blood pressure — pulse pressure and mean arterial pressure summarise the
  systolic/diastolic pair.
* Adiposity — obesity (BMI) and central obesity (waist-to-hip ratio) flags, and a
  BMI x age interaction.
* Accumulated risk — count of history flags plus an overall metabolic-risk score.
* Lifestyle — a physical-activity-to-screen-time balance.

Ordered categoricals (education, income, smoking) are mapped to integer codes
here. Nominal categoricals (gender, ethnicity, employment) are left as strings
for the one-hot encoder in :mod:`src.preprocessing`.

Every engineered feature is a deterministic function of a single row, so there
is no fitted state and no leakage risk; ``fit`` only records the input columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .data import ORDINAL_LEVELS

# Listed in the exact order ``_add_features`` appends them, so this matches the
# transformed frame's column order and ``get_feature_names_out``.
ENGINEERED_FLAGS: tuple[str, ...] = ("is_obese", "high_waist")
ENGINEERED_FEATURES: tuple[str, ...] = (
    "tg_hdl_ratio",
    "chol_hdl_ratio",
    "ldl_hdl_ratio",
    "non_hdl_cholesterol",
    "pulse_pressure",
    "mean_arterial_pressure",
    "bmi_age",
    "is_obese",
    "high_waist",
    "comorbidity_count",
    "metabolic_risk",
    "activity_screen_ratio",
)
ENGINEERED_NUMERIC: tuple[str, ...] = tuple(
    f for f in ENGINEERED_FEATURES if f not in ENGINEERED_FLAGS
)


class DiabetesFeatureEngineer(BaseEstimator, TransformerMixin):
    """Encode ordinal categoricals and append clinical composite features.

    Parameters
    ----------
    add_engineered:
        When ``False`` only the ordinal encoding is applied (no composites),
        convenient for ablation experiments.
    """

    def __init__(self, add_engineered: bool = True) -> None:
        self.add_engineered = add_engineered

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "DiabetesFeatureEngineer":
        self.feature_names_in_ = np.asarray(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        result = X.copy()
        self._encode_ordinals(result)
        if self.add_engineered:
            result = self._add_features(result)
        return result

    def _encode_ordinals(self, frame: pd.DataFrame) -> None:
        """Map ordered categoricals to integer codes in place (-1 if unseen)."""
        for column, levels in ORDINAL_LEVELS.items():
            mapping = {level: code for code, level in enumerate(levels)}
            frame[column] = frame[column].map(mapping).fillna(-1).astype("int8")

    def _add_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        hdl = frame["hdl_cholesterol"].astype("float64")
        systolic = frame["systolic_bp"].astype("float64")
        diastolic = frame["diastolic_bp"].astype("float64")
        bmi = frame["bmi"].astype("float64")

        # Lipid ratios (+1 denominators keep them finite).
        frame["tg_hdl_ratio"] = frame["triglycerides"] / (hdl + 1.0)
        frame["chol_hdl_ratio"] = frame["cholesterol_total"] / (hdl + 1.0)
        frame["ldl_hdl_ratio"] = frame["ldl_cholesterol"] / (hdl + 1.0)
        frame["non_hdl_cholesterol"] = frame["cholesterol_total"] - hdl

        # Blood-pressure summaries.
        frame["pulse_pressure"] = systolic - diastolic
        frame["mean_arterial_pressure"] = diastolic + (systolic - diastolic) / 3.0

        # Adiposity.
        frame["bmi_age"] = bmi * frame["age"].astype("float64")
        frame["is_obese"] = (bmi >= 30.0).astype("int8")
        frame["high_waist"] = (frame["waist_to_hip_ratio"] >= 0.90).astype("int8")

        # Accumulated risk.
        comorbidity = (
            frame["family_history_diabetes"]
            + frame["hypertension_history"]
            + frame["cardiovascular_history"]
        )
        frame["comorbidity_count"] = comorbidity.astype("int8")
        frame["metabolic_risk"] = (
            comorbidity + frame["is_obese"] + frame["high_waist"]
        ).astype("int8")

        # Lifestyle balance: weekly active minutes vs weekly screen minutes.
        screen_minutes = frame["screen_time_hours_per_day"].astype("float64") * 7.0 * 60.0
        frame["activity_screen_ratio"] = (
            frame["physical_activity_minutes_per_week"] / (screen_minutes + 1.0)
        )
        return frame

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        names = list(self.feature_names_in_)
        if self.add_engineered:
            names = names + list(ENGINEERED_FEATURES)
        return np.asarray(names, dtype=object)


def build_features(frame: pd.DataFrame, *, add_engineered: bool = True) -> pd.DataFrame:
    """Convenience wrapper that fits and transforms ``frame`` in one call."""
    return DiabetesFeatureEngineer(add_engineered=add_engineered).fit_transform(frame)
