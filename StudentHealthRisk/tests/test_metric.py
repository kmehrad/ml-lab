"""Unit tests for the balanced-accuracy metric, decision tuning, and submission builder."""
from __future__ import annotations

import numpy as np
import pytest

from src.metric import (
    balanced_accuracy,
    predict_codes,
    score_proba,
    to_submission_labels,
    tune_weights,
)


def test_balanced_accuracy_matches_mean_recall():
    # 3 classes. class 0: 2/2 correct, class 1: 1/2, class 2: 0/2 -> (1 + .5 + 0)/3
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 0, 0, 0])
    assert balanced_accuracy(y_true, y_pred) == pytest.approx((1.0 + 0.5 + 0.0) / 3.0)


def test_predict_codes_weighted_argmax():
    proba = np.array([[0.6, 0.3, 0.1]])
    # raw argmax -> class 0; upweighting class 2 by 10x flips the decision to class 2
    assert predict_codes(proba)[0] == 0
    assert predict_codes(proba, weights=[1.0, 1.0, 10.0])[0] == 2


def test_tune_weights_beats_raw_on_imbalanced():
    # Heavily imbalanced (like the real target): raw argmax ignores the minority class and scores
    # poorly on balanced accuracy; tuned per-class weights must do at least as well.
    rng = np.random.default_rng(0)
    n = 3000
    y = np.array([0] * 2600 + [1] * 250 + [2] * 150)
    proba = np.full((n, 3), 0.02)
    proba[np.arange(n), y] += rng.uniform(0.1, 0.5, n)  # weak but real signal on the true class
    proba /= proba.sum(axis=1, keepdims=True)
    w = tune_weights(y, proba)
    assert score_proba(y, proba, w) >= score_proba(y, proba) - 1e-9
    assert w.shape == (3,)


def test_submission_labels():
    classes = np.array(["at-risk", "fit", "unhealthy"])
    proba = np.array([[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]])
    out = to_submission_labels(proba, classes)
    assert list(out) == ["at-risk", "unhealthy"]
    assert set(out).issubset(set(classes))
