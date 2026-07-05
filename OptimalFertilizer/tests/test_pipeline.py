"""Light tests for schema helpers and the baseline feature transform (no heavy data load)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import data as D
from src.features import add_features, feature_columns


def test_encode_target_roundtrip():
    labels = ["Urea", "DAP", "10-26-26", "Urea"]
    codes = D.encode_target(labels)
    assert codes.dtype == np.int64
    # decoding via CLASSES must recover the original labels
    assert [D.CLASSES[c] for c in codes] == labels
    # canonical ordering: CLASSES is sorted and length 7
    assert D.CLASSES == sorted(D.CLASSES) and D.N_CLASSES == 7


def test_feature_columns_and_transform():
    df = pd.DataFrame({
        "Temparature": [30, 32], "Humidity": [60, 55], "Moisture": [40, 50],
        "Nitrogen": [20, 10], "Potassium": [5, 9], "Phosphorous": [15, 30],
        "Soil Type": ["Sandy", "Loamy"], "Crop Type": ["Maize", "Wheat"],
    })
    out = add_features(df)
    feats = feature_columns()
    assert set(feats) == set(D.NUMERIC + D.CATEGORICAL)
    for c in D.CATEGORICAL:
        assert isinstance(out[c].dtype, pd.CategoricalDtype)
    # numeric columns untouched
    for c in D.NUMERIC:
        assert out[c].tolist() == df[c].tolist()
