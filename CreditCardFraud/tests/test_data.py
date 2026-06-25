"""Tests for schema validation and the fixed held-out split."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import data


def _toy_frame(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = {c: rng.normal(size=n) for c in data.RAW_FEATURES}
    cols["Time"] = np.sort(rng.uniform(0, 172792, size=n))
    cols["Amount"] = rng.uniform(0, 1000, size=n)
    # ~5% positives so stratification has something to preserve.
    cols[data.TARGET_COLUMN] = (rng.uniform(size=n) < 0.05).astype(int)
    return pd.DataFrame(cols)


def test_check_schema_raises_on_missing_feature():
    frame = _toy_frame().drop(columns=["V1"])
    with pytest.raises(ValueError, match="missing expected feature"):
        data._check_schema(frame)


def test_check_schema_raises_on_missing_target():
    frame = _toy_frame().drop(columns=[data.TARGET_COLUMN])
    with pytest.raises(ValueError, match="missing target"):
        data._check_schema(frame)


def test_split_features_target_shapes():
    frame = _toy_frame()
    X, y = data.split_features_target(frame)
    assert list(X.columns) == list(data.RAW_FEATURES)
    assert data.TARGET_COLUMN not in X.columns
    assert set(y.unique()) <= {0, 1}
    assert len(X) == len(y) == len(frame)


def test_holdout_split_is_stratified_and_deterministic(tmp_path, monkeypatch):
    frame = _toy_frame(n=1000)
    csv = tmp_path / "creditcard.csv"
    frame.to_csv(csv, index=False)
    monkeypatch.setattr(data, "DEFAULT_DATA_DIR", tmp_path)

    tr1, te1 = data.make_holdout_split(test_size=0.2, seed=42, cache=False)
    tr2, te2 = data.make_holdout_split(test_size=0.2, seed=42, cache=False)

    # Deterministic and a clean partition.
    assert np.array_equal(tr1, tr2) and np.array_equal(te1, te2)
    assert set(tr1).isdisjoint(te1)
    assert len(tr1) + len(te1) == len(frame)
    assert len(te1) == pytest.approx(0.2 * len(frame), abs=1)

    # Positive rate preserved across the split.
    y = frame[data.TARGET_COLUMN].to_numpy()
    assert y[te1].mean() == pytest.approx(y.mean(), abs=0.01)
