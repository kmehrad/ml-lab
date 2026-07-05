"""Light tests for schema helpers and the baseline feature transform (no heavy data load)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import data as D
from src.features import add_features, feature_columns


def test_encode_decode_target_roundtrip():
    labels = ["at-risk", "fit", "unhealthy", "at-risk"]
    codes = D.encode_target(labels)
    assert codes.dtype == np.int64
    assert list(D.decode_target(codes)) == labels
    # canonical ordering: CLASSES sorted, length 3
    assert D.CLASSES == sorted(D.CLASSES) and D.N_CLASSES == 3


def _toy_frame():
    return pd.DataFrame({
        "sleep_duration": [7.0, np.nan], "heart_rate": [70.0, 80.0], "bmi": [22.0, 27.0],
        "calorie_expenditure": [2000.0, 2500.0], "step_count": [8000.0, 12000.0],
        "exercise_duration": [30.0, 45.0], "water_intake": [2.0, 3.0],
        "diet_type": ["veg", None], "stress_level": ["low", "high"],
        "sleep_quality": ["good", "poor"], "physical_activity_level": ["active", "sedentary"],
        "smoking_alcohol": ["no", "yes"], "gender": ["male", "female"],
    })


def test_base_feature_columns_and_dtypes():
    out = add_features(_toy_frame())
    feats = feature_columns()
    assert set(feats) == set(D.NUMERIC + D.CATEGORICAL)
    for c in D.CATEGORICAL:
        assert isinstance(out[c].dtype, pd.CategoricalDtype)   # NaN kept as a level, not filled
    for c in D.NUMERIC:
        assert out[c].fillna(-1).tolist() == _toy_frame()[c].fillna(-1).tolist()


def test_engineered_groups_add_only_numeric_columns():
    out = add_features(_toy_frame(), groups=("base", "ratio", "missflag"))
    feats = feature_columns(("base", "ratio", "missflag"))
    assert set(feats).issubset(set(out.columns))
    # every engineered column is numeric (categoricals stay the raw six)
    from src.features import categorical_columns
    assert categorical_columns(("base", "ratio", "missflag")) == list(D.CATEGORICAL)
    assert "steps_per_calorie" in out.columns and "diet_type_isna" in out.columns
