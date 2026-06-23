"""Tests for the clinical feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import ORDINAL_FEATURES
from src.features import (
    ENGINEERED_FEATURES,
    DiabetesFeatureEngineer,
    build_features,
)


def _sample_frame() -> pd.DataFrame:
    # Three rows ordered low -> high diabetes risk.
    return pd.DataFrame(
        {
            "age": [30, 50, 70],
            "alcohol_consumption_per_week": [1, 3, 7],
            "physical_activity_minutes_per_week": [300, 120, 0],
            "diet_score": [8.0, 5.0, 2.0],
            "sleep_hours_per_day": [8.0, 6.5, 5.0],
            "screen_time_hours_per_day": [1.0, 4.0, 8.0],
            "bmi": [22.0, 27.0, 34.0],
            "waist_to_hip_ratio": [0.80, 0.88, 0.97],
            "systolic_bp": [110, 130, 160],
            "diastolic_bp": [70, 85, 100],
            "heart_rate": [60, 75, 90],
            "cholesterol_total": [160, 200, 260],
            "hdl_cholesterol": [60, 50, 30],
            "ldl_cholesterol": [80, 120, 180],
            "triglycerides": [80, 150, 300],
            "gender": ["Female", "Male", "Other"],
            "ethnicity": ["Asian", "White", "Black"],
            "education_level": ["Postgraduate", "Highschool", "No formal"],
            "income_level": ["High", "Middle", "Low"],
            "smoking_status": ["Never", "Former", "Current"],
            "employment_status": ["Employed", "Retired", "Unemployed"],
            "family_history_diabetes": [0, 1, 1],
            "hypertension_history": [0, 0, 1],
            "cardiovascular_history": [0, 0, 1],
        }
    )


def test_engineered_columns_added_and_finite():
    out = build_features(_sample_frame())
    for column in ENGINEERED_FEATURES:
        assert column in out.columns
    engineered = out[list(ENGINEERED_FEATURES)].to_numpy(dtype="float64")
    assert np.isfinite(engineered).all()


def test_ordinals_encoded_to_ordered_codes():
    out = build_features(_sample_frame())
    # Rows go high -> low education/income, never -> current smoking.
    assert out["education_level"].tolist() == [3, 1, 0]
    assert out["income_level"].tolist() == [4, 2, 0]
    assert out["smoking_status"].tolist() == [0, 1, 2]
    for col in ORDINAL_FEATURES:
        assert out[col].dtype.kind in "iu"


def test_nominals_left_as_strings_for_encoder():
    out = build_features(_sample_frame())
    assert out["gender"].tolist() == ["Female", "Male", "Other"]
    # Left non-numeric (object or pandas StringDtype) for the one-hot encoder.
    assert out["ethnicity"].dtype.kind in "OU" or isinstance(out["ethnicity"].dtype, pd.StringDtype)


def test_composites_match_semantics():
    out = build_features(_sample_frame())
    assert out["is_obese"].tolist() == [0, 0, 1]
    assert out["high_waist"].tolist() == [0, 0, 1]
    assert out["comorbidity_count"].tolist() == [0, 1, 3]
    assert out["metabolic_risk"].tolist() == [0, 1, 5]
    assert out["pulse_pressure"].tolist() == [40.0, 45.0, 60.0]
    # tg/hdl rises sharply with risk (insulin-resistance surrogate).
    tg_hdl = out["tg_hdl_ratio"].to_numpy()
    assert tg_hdl[0] < tg_hdl[1] < tg_hdl[2]


def test_transform_is_stateless_no_leakage():
    # Engineered features are row-wise; a single row transformed alone must equal
    # the same row transformed within a batch (no fitted statistics).
    frame = _sample_frame()
    engineer = DiabetesFeatureEngineer().fit(frame)
    single = frame.iloc[[2]].reset_index(drop=True)
    cols = list(ENGINEERED_FEATURES)
    expected = engineer.transform(frame).iloc[2][cols]
    got = engineer.transform(single).iloc[0][cols]
    pd.testing.assert_series_equal(expected, got, check_names=False)
