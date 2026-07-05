"""Unit tests for the MAP@3 metric and the top-3 submission-string builder."""
from __future__ import annotations

import numpy as np
import pytest

from src.metric import mapk3, to_submission_strings, top3_indices


def test_mapk3_rank_positions():
    # 3 rows, 4 classes. Probabilities chosen so the true class lands at rank 1, 2, 3.
    proba = np.array([
        [0.7, 0.1, 0.1, 0.1],  # argsort -> [0,1,2,3]; true=0 at rank 1 -> 1.0
        [0.1, 0.7, 0.1, 0.1],  # argsort -> [1,0,2,3]; true=0 at rank 2 -> 1/2
        [0.1, 0.6, 0.2, 0.1],  # argsort -> [1,2,0,3]; true=0 at rank 3 -> 1/3
    ])
    y = np.array([0, 0, 0])
    assert mapk3(y, proba) == pytest.approx((1.0 + 0.5 + 1.0 / 3.0) / 3.0)


def test_mapk3_miss_scores_zero():
    # true class ranked 4th (outside top-3) -> 0 for that row.
    proba = np.array([[0.4, 0.3, 0.2, 0.1]])
    assert mapk3(np.array([3]), proba) == pytest.approx(0.0)


def test_mapk3_all_correct_and_all_missed():
    proba = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05]])
    assert mapk3(np.array([0, 1]), proba) == pytest.approx(1.0)     # both rank 1
    assert mapk3(np.array([2, 2]), proba) == pytest.approx(1.0 / 3) # class 2 is rank 3 in both


def test_stable_tiebreak_by_class_code():
    # all-equal probabilities -> top-3 must be the lowest class codes, in order.
    proba = np.full((1, 5), 0.2)
    np.testing.assert_array_equal(top3_indices(proba), np.array([[0, 1, 2]]))


def test_submission_strings_format():
    classes = np.array(["A", "B", "C", "D"])
    proba = np.array([
        [0.7, 0.1, 0.1, 0.1],  # -> "A B C" (best-first)
        [0.1, 0.2, 0.3, 0.4],  # -> "D C B"
    ])
    out = to_submission_strings(proba, classes)
    assert out == ["A B C", "D C B"]
    for s in out:
        parts = s.split(" ")
        assert len(parts) == 3                       # exactly 3 names
        assert len(set(parts)) == 3                  # distinct
        assert all(p in set(classes) for p in parts) # valid class names
