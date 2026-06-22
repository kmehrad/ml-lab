"""Domain feature engineering for the Irrigation Need dataset.

The EDA (``reports/EDA_FINDINGS.md``) showed irrigation need rises with drier
soil, less rainfall, hotter temperatures, and stronger wind, while most
categoricals carry little marginal signal. These transforms encode that
agronomy explicitly: a reference evapotranspiration (water *demand*) proxy,
demand-vs-supply ratios, a standardized aridity composite, and the two semantic
"no practice applied" flags identified in the EDA.

All engineered features are deterministic functions of a single row except the
``aridity_index`` composite, whose standardization statistics are learned from
the training split in ``fit`` to avoid leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .data import CATEGORICAL_FEATURES

# Columns whose z-scores compose the aridity index. POS pushes need up (drier),
# NEG pushes need down (wetter / more humid).
_ARIDITY_POSITIVE = ("Temperature_C", "Wind_Speed_kmh")
_ARIDITY_NEGATIVE = ("Soil_Moisture", "Rainfall_mm", "Humidity")
_ARIDITY_COLUMNS = _ARIDITY_POSITIVE + _ARIDITY_NEGATIVE

ENGINEERED_NUMERIC: tuple[str, ...] = (
    "et_demand",
    "temp_to_moisture",
    "moisture_rain_ratio",
    "wind_dryness",
    "aridity_index",
)
ENGINEERED_FLAGS: tuple[str, ...] = (
    "is_rainfed",
    "no_mulch",
)
ENGINEERED_FEATURES: tuple[str, ...] = ENGINEERED_NUMERIC + ENGINEERED_FLAGS


class IrrigationFeatureEngineer(BaseEstimator, TransformerMixin):
    """Append agronomic features derived from the raw columns.

    Parameters
    ----------
    add_engineered:
        When ``False`` the transformer is a pass-through, which is convenient for
        ablation experiments that isolate the engineered features' contribution.
    categorical_as_category:
        Cast the original categorical columns to ``category`` dtype on output so
        gradient-boosted tree libraries can consume them natively.
    """

    def __init__(
        self,
        add_engineered: bool = True,
        categorical_as_category: bool = True,
    ) -> None:
        self.add_engineered = add_engineered
        self.categorical_as_category = categorical_as_category

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "IrrigationFeatureEngineer":
        if self.add_engineered:
            stats = X[list(_ARIDITY_COLUMNS)].astype("float64")
            self.aridity_means_ = stats.mean()
            # Guard against a zero-variance column producing inf z-scores.
            self.aridity_stds_ = stats.std(ddof=0).replace(0.0, 1.0)
        self.feature_names_in_ = np.asarray(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        result = X.copy()
        if self.add_engineered:
            self._check_fitted()
            result = self._add_features(result)
        if self.categorical_as_category:
            for column in CATEGORICAL_FEATURES:
                if column in result.columns:
                    result[column] = result[column].astype("category")
        return result

    def _add_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        temp = frame["Temperature_C"]
        humidity = frame["Humidity"]
        wind = frame["Wind_Speed_kmh"]
        moisture = frame["Soil_Moisture"]
        rainfall = frame["Rainfall_mm"]

        dryness = (100.0 - humidity) / 100.0  # 0 at saturation, 1 when bone dry

        # Reference evapotranspiration proxy: water demand grows with heat, dry
        # air, and wind. Scaled so a calm, saturated day contributes ~0.
        frame["et_demand"] = temp * dryness * (1.0 + wind / 20.0)
        # Demand-vs-supply ratios (+1 denominators keep them finite and bounded).
        frame["temp_to_moisture"] = temp / (moisture + 1.0)
        frame["moisture_rain_ratio"] = moisture / (rainfall + 1.0)
        frame["wind_dryness"] = wind * dryness

        # Standardized composite of the four strongest EDA signals.
        z = (frame[list(_ARIDITY_COLUMNS)].astype("float64") - self.aridity_means_) / self.aridity_stds_
        frame["aridity_index"] = (
            z[list(_ARIDITY_POSITIVE)].sum(axis=1) - z[list(_ARIDITY_NEGATIVE)].sum(axis=1)
        )

        # Semantic "none applied" flags surfaced in the EDA (Q3).
        frame["is_rainfed"] = (frame["Irrigation_Type"].astype("object") == "Rainfed").astype("int8")
        frame["no_mulch"] = (frame["Mulching_Used"].astype("object") == "No").astype("int8")
        return frame

    def _check_fitted(self) -> None:
        if not hasattr(self, "aridity_means_"):
            raise RuntimeError("IrrigationFeatureEngineer must be fitted before transform")

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        names = list(self.feature_names_in_)
        if self.add_engineered:
            names = names + list(ENGINEERED_FEATURES)
        return np.asarray(names, dtype=object)


def build_features(
    frame: pd.DataFrame,
    *,
    add_engineered: bool = True,
    categorical_as_category: bool = True,
) -> pd.DataFrame:
    """Convenience wrapper that fits and transforms ``frame`` in one call.

    Use this for single-frame exploration only. Inside cross-validation, fit a
    fresh :class:`IrrigationFeatureEngineer` on the training fold and transform
    the validation fold so the ``aridity_index`` statistics never leak.
    """
    engineer = IrrigationFeatureEngineer(
        add_engineered=add_engineered,
        categorical_as_category=categorical_as_category,
    )
    return engineer.fit_transform(frame)
