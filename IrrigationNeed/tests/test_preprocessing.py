"""Tests for data loading, target encoding, and the preprocessing pipelines."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import (
    CATEGORICAL_FEATURES,
    RAW_FEATURES,
    decode_target,
    encode_target,
)
from src.preprocessing import build_preprocessor, prepare_tree_features
from tests.test_features import _sample_frame


def test_encode_decode_round_trip_is_ordinal():
    labels = pd.Series(["Low", "Medium", "High", "Low"])
    codes = encode_target(labels)
    assert codes.tolist() == [0, 1, 2, 0]  # ordering preserved
    assert decode_target(codes).tolist() == labels.tolist()


def test_encode_unknown_label_raises():
    with pytest.raises(ValueError):
        encode_target(pd.Series(["Low", "VeryHigh"]))


def test_tree_pipeline_keeps_categoricals_as_category():
    out = prepare_tree_features(_sample_frame())
    for column in CATEGORICAL_FEATURES:
        assert isinstance(out[column].dtype, pd.CategoricalDtype)
    # Raw + engineered columns present, no NaNs introduced.
    assert out.shape[1] > len(RAW_FEATURES)
    assert not out.isna().any().any()


def test_linear_pipeline_returns_finite_numeric_matrix():
    pipe = build_preprocessor("linear")
    out = pipe.fit_transform(_sample_frame())
    # One-hot expands categoricals, so width exceeds the raw column count.
    assert out.shape[0] == 3
    values = np.asarray(out, dtype="float64")
    assert np.isfinite(values).all()


def test_linear_pipeline_handles_unseen_category():
    pipe = build_preprocessor("linear").fit(_sample_frame())
    novel = _sample_frame().iloc[[0]].copy()
    novel["Soil_Type"] = pd.Categorical(["Silt"], categories=["Silt"])
    # handle_unknown="ignore" must not raise on a level absent from this batch.
    transformed = pipe.transform(novel)
    assert np.isfinite(np.asarray(transformed, dtype="float64")).all()


def test_invalid_kind_raises():
    with pytest.raises(ValueError):
        build_preprocessor("svm")
