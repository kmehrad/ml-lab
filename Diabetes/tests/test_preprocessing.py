"""Tests for the preprocessing pipelines."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import NOMINAL_FEATURES, RAW_FEATURES
from src.features import ENGINEERED_FEATURES
from src.preprocessing import build_preprocessor


def _sample_frame(n: int = 64, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "age": rng.integers(18, 90, n),
            "alcohol_consumption_per_week": rng.integers(0, 10, n),
            "physical_activity_minutes_per_week": rng.integers(0, 600, n),
            "diet_score": rng.uniform(0, 10, n),
            "sleep_hours_per_day": rng.uniform(4, 10, n),
            "screen_time_hours_per_day": rng.uniform(0, 12, n),
            "bmi": rng.uniform(15, 50, n),
            "waist_to_hip_ratio": rng.uniform(0.7, 1.1, n),
            "systolic_bp": rng.integers(90, 180, n),
            "diastolic_bp": rng.integers(60, 110, n),
            "heart_rate": rng.integers(50, 110, n),
            "cholesterol_total": rng.integers(120, 300, n),
            "hdl_cholesterol": rng.integers(20, 90, n),
            "ldl_cholesterol": rng.integers(50, 200, n),
            "triglycerides": rng.integers(50, 400, n),
            "gender": rng.choice(["Female", "Male", "Other"], n),
            "ethnicity": rng.choice(["Asian", "Black", "Hispanic", "Other", "White"], n),
            "education_level": rng.choice(
                ["No formal", "Highschool", "Graduate", "Postgraduate"], n),
            "income_level": rng.choice(
                ["Low", "Lower-Middle", "Middle", "Upper-Middle", "High"], n),
            "smoking_status": rng.choice(["Never", "Former", "Current"], n),
            "employment_status": rng.choice(
                ["Employed", "Retired", "Student", "Unemployed"], n),
            "family_history_diabetes": rng.integers(0, 2, n),
            "hypertension_history": rng.integers(0, 2, n),
            "cardiovascular_history": rng.integers(0, 2, n),
        }
    )[list(RAW_FEATURES)]


def _expected_width() -> int:
    # one-hot widths for the nominal categoricals in the sample frame
    onehot = {"gender": 3, "ethnicity": 5, "employment_status": 4}
    n_onehot = sum(onehot[c] for c in NOMINAL_FEATURES)
    n_other = len(RAW_FEATURES) - len(NOMINAL_FEATURES) + len(ENGINEERED_FEATURES)
    return n_onehot + n_other


def test_tree_pipeline_is_all_numeric_no_nan():
    frame = _sample_frame()
    out = build_preprocessor("tree").fit_transform(frame)
    assert out.shape[1] == _expected_width()
    assert out.select_dtypes(include="object").shape[1] == 0
    assert out.notna().all().all()


def test_linear_pipeline_scales_non_onehot_columns():
    frame = _sample_frame()
    out = build_preprocessor("linear").fit_transform(frame)
    assert out.shape[1] == _expected_width()
    # The continuous columns should be standardized (~zero mean, unit variance).
    for col in ("bmi", "age", "tg_hdl_ratio"):
        assert abs(out[col].mean()) < 1e-6
        assert abs(out[col].std(ddof=0) - 1.0) < 1e-2


def test_pipeline_fit_on_train_only_transforms_holdout():
    frame = _sample_frame(n=100)
    train, holdout = frame.iloc[:50], frame.iloc[50:]
    pre = build_preprocessor("linear").fit(train)
    out = pre.transform(holdout)
    assert len(out) == 50
    assert np.isfinite(out.to_numpy()).all()
