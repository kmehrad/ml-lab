"""Tests for the agronomic feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import RAW_FEATURES
from src.features import (
    ENGINEERED_FEATURES,
    IrrigationFeatureEngineer,
    build_features,
)


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Soil_pH": [6.5, 7.1, 5.2],
            "Soil_Moisture": [40.0, 20.0, 10.0],
            "Organic_Carbon": [0.9, 1.1, 0.5],
            "Electrical_Conductivity": [1.5, 2.0, 1.0],
            "Temperature_C": [22.0, 30.0, 38.0],
            "Humidity": [70.0, 50.0, 30.0],
            "Rainfall_mm": [1500.0, 1000.0, 500.0],
            "Sunlight_Hours": [6.0, 8.0, 9.0],
            "Wind_Speed_kmh": [5.0, 12.0, 18.0],
            "Field_Area_hectare": [3.0, 7.0, 11.0],
            "Previous_Irrigation_mm": [60.0, 62.0, 63.0],
            "Soil_Type": ["Clay", "Loamy", "Sandy"],
            "Crop_Type": ["Wheat", "Maize", "Cotton"],
            "Crop_Growth_Stage": ["Sowing", "Vegetative", "Flowering"],
            "Season": ["Rabi", "Kharif", "Zaid"],
            "Irrigation_Type": ["Drip", "Rainfed", "Canal"],
            "Water_Source": ["River", "Rainwater", "Groundwater"],
            "Mulching_Used": ["Yes", "No", "No"],
            "Region": ["East", "South", "West"],
        }
    )


def test_engineered_columns_added_and_finite():
    out = build_features(_sample_frame())
    for column in ENGINEERED_FEATURES:
        assert column in out.columns
    engineered = out[list(ENGINEERED_FEATURES)].to_numpy(dtype="float64")
    assert np.isfinite(engineered).all()


def test_flags_match_semantics():
    out = build_features(_sample_frame())
    assert out["is_rainfed"].tolist() == [0, 1, 0]
    assert out["no_mulch"].tolist() == [0, 1, 1]


def test_aridity_index_increases_with_dryness():
    # Rows are ordered wet -> dry, so the aridity composite should increase.
    out = build_features(_sample_frame())
    aridity = out["aridity_index"].to_numpy()
    assert aridity[0] < aridity[1] < aridity[2]


def test_transform_uses_train_statistics_no_leakage():
    train = _sample_frame()
    engineer = IrrigationFeatureEngineer().fit(train)
    # A held-out row transformed alone must reuse the train mean/std, so the
    # value matches the same row transformed within the training batch.
    single = train.iloc[[1]].reset_index(drop=True)
    expected = engineer.transform(train).iloc[1]["aridity_index"]
    got = engineer.transform(single).iloc[0]["aridity_index"]
    assert got == pytest.approx(expected)


def test_passthrough_when_disabled():
    out = build_features(_sample_frame(), add_engineered=False)
    assert list(out.columns) == list(RAW_FEATURES)


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        IrrigationFeatureEngineer().transform(_sample_frame())
