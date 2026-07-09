"""Light tests for schema helpers and the baseline feature transform (no heavy data load)."""
from __future__ import annotations

import pandas as pd

from src import data as D
from src.features import add_features, categorical_columns, feature_columns

_SAMPLE = pd.DataFrame({
    "age": [20, 22], "study_hours": [4.0, 2.5], "class_attendance": [80.0, 60.0],
    "sleep_hours": [7.0, 6.5], "gender": ["male", "female"], "course": ["b.tech", "bca"],
    "internet_access": ["yes", "no"], "sleep_quality": ["good", "poor"],
    "study_method": ["self-study", "coaching"], "facility_rating": ["high", "low"],
    "exam_difficulty": ["easy", "hard"],
})


def test_schema_constants():
    assert D.TARGET == "exam_score" and D.ID == "id"
    assert len(D.NUMERIC) == 4 and len(D.CATEGORICAL) == 7
    assert D.FEATURES == D.NUMERIC + D.CATEGORICAL


def test_feature_columns_and_transform():
    out = add_features(_SAMPLE)
    feats = feature_columns()
    assert set(feats) == set(D.NUMERIC + D.CATEGORICAL)
    assert categorical_columns() == D.CATEGORICAL
    for c in D.CATEGORICAL:
        assert isinstance(out[c].dtype, pd.CategoricalDtype)
    # numeric columns untouched
    for c in D.NUMERIC:
        assert out[c].tolist() == _SAMPLE[c].tolist()


def test_ratios_group_adds_columns():
    from src.features import RATIO_COLS
    out = add_features(_SAMPLE, groups=("base", "ratios"))
    for c in RATIO_COLS:
        assert c in out.columns
    assert set(feature_columns(("base", "ratios"))) == set(D.NUMERIC + D.CATEGORICAL + RATIO_COLS)
