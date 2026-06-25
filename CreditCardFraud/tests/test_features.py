"""Tests for the stateless feature engineer and preprocessing pipelines."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import data
from src.features import ENGINEERED_FEATURES, FraudFeatureEngineer
from src.preprocessing import build_preprocessor


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            **{c: np.linspace(-1, 1, 6) for c in data.RAW_FEATURES},
            "Time": [0.0, 3600.0, 7200.0, 86400.0, 90000.0, 172792.0],
            "Amount": [0.0, 1.0, 10.0, 100.0, 0.0, 25000.0],
        }
    )


def test_engineered_semantics():
    out = FraudFeatureEngineer().transform(_frame())
    assert np.allclose(out["amount_log"], np.log1p(out["Amount"]))
    assert out["amount_zero"].tolist() == [1, 0, 0, 0, 1, 0]
    # 3600s -> hour 1, 86400s -> hour 0, 90000s -> hour 1.
    assert np.allclose(out["hour"], [0, 1, 2, 0, 1, (172792 / 3600) % 24])


def test_engineer_is_stateless_and_preserves_pca():
    frame = _frame()
    eng = FraudFeatureEngineer()
    a = eng.fit(frame).transform(frame)
    b = eng.transform(frame)  # no fit in between
    pd.testing.assert_frame_equal(a, b)
    for col in (f"V{i}" for i in range(1, 29)):
        assert np.array_equal(a[col].to_numpy(), frame[col].to_numpy())


def test_add_engineered_false_passes_through():
    frame = _frame()
    out = FraudFeatureEngineer(add_engineered=False).transform(frame)
    assert list(out.columns) == list(frame.columns)


def test_tree_vs_linear_columns():
    frame = _frame()
    tree = build_preprocessor("tree").fit_transform(frame)
    linear = build_preprocessor("linear").fit_transform(frame)
    expected = list(data.RAW_FEATURES) + list(ENGINEERED_FEATURES)
    assert set(tree.columns) == set(expected)
    assert set(linear.columns) == set(expected)
    # tree passes Amount through unchanged; linear scales it (median/IQR).
    assert np.array_equal(tree["Amount"].to_numpy(), frame["Amount"].to_numpy())
    assert not np.array_equal(linear["Amount"].to_numpy(), frame["Amount"].to_numpy())
