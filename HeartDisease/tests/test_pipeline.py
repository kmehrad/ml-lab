"""Light tests for schema helpers and the baseline feature transform (no heavy data load)."""
from __future__ import annotations

import pandas as pd

from src import data as D
from src.features import add_features, categorical_columns, feature_columns

_SAMPLE = pd.DataFrame({
    "Age": [58, 52], "BP": [130, 120], "Cholesterol": [230, 200], "Max HR": [150, 160],
    "ST depression": [2.0, 0.0], "Sex": [1, 0], "Chest pain type": [4, 2],
    "FBS over 120": [0, 1], "EKG results": [2, 0], "Exercise angina": [1, 0],
    "Slope of ST": [2, 1], "Number of vessels fluro": [2, 0], "Thallium": [7, 3],
    "Heart Disease": ["Presence", "Absence"],
})


def test_schema_constants():
    assert D.TARGET == "Heart Disease" and D.ID == "id"
    assert len(D.NUMERIC) == 5 and len(D.CATEGORICAL) == 8
    assert D.FEATURES == D.NUMERIC + D.CATEGORICAL


def test_encode_target():
    y = D.encode_target(_SAMPLE)
    assert y.tolist() == [1, 0]


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


def test_interactions_group_adds_columns():
    from src.features import INTERACTION_COLS
    out = add_features(_SAMPLE, groups=("base", "interactions"))
    for c in INTERACTION_COLS:
        assert c in out.columns
    assert set(feature_columns(("base", "interactions"))) == (
        set(D.NUMERIC + D.CATEGORICAL + INTERACTION_COLS)
    )
