import numpy as np
import pandas as pd
import pytest

from src import data as D


def test_load_train_schema():
    df = D.load_train()
    assert list(df.columns) == D.TRAIN_COLS
    assert set(df[D.TARGET].unique()) <= set(D.CLASSES)
    assert len(df) > 0


def test_load_test_schema():
    df = D.load_test()
    assert list(df.columns) == D.TEST_COLS


def test_sample_submission_schema():
    df = D.load_sample_submission()
    assert list(df.columns) == [D.ID_COL, D.TARGET]


def test_schema_mismatch_raises():
    bad = pd.DataFrame({"id": [1], "foo": [2]})
    with pytest.raises(ValueError):
        D._check_columns(bad, D.TRAIN_COLS, "train.csv")


def test_label_round_trip():
    y = pd.Series(["GALAXY", "QSO", "STAR", "GALAXY"])
    enc = D.encode_target(y)
    assert enc.tolist() == [0, 1, 2, 0]
    assert D.decode_target(enc).tolist() == y.tolist()
    assert np.array_equal(D.decode_target(D.encode_target(y)), y.to_numpy())
