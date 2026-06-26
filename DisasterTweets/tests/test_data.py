"""Schema and sanity tests for the raw competition data."""
from __future__ import annotations

import pytest

from src import data


@pytest.fixture(scope="module")
def train():
    return data.load_train()


@pytest.fixture(scope="module")
def test():
    return data.load_test()


def test_train_columns(train):
    for col in data.TRAIN_COLS:
        assert col in train.columns


def test_test_columns(test):
    for col in data.FEATURE_COLS:
        assert col in test.columns
    assert data.TARGET not in test.columns


def test_target_is_binary(train):
    assert set(train[data.TARGET].unique()) <= {0, 1}


def test_text_never_missing(train, test):
    assert train[data.TEXT_COL].isna().sum() == 0
    assert test[data.TEXT_COL].isna().sum() == 0


def test_row_counts(train, test):
    assert len(train) == data.N_TRAIN
    assert len(test) == data.N_TEST


def test_ids_unique(train, test):
    assert train[data.ID_COL].is_unique
    assert test[data.ID_COL].is_unique
