"""Tests for the RMSE metric — the single source of truth for scoring."""
from __future__ import annotations

import numpy as np

from src.metric import rmse


def test_rmse_perfect_is_zero():
    y = [10.0, 20.0, 30.0]
    assert rmse(y, y) == 0.0


def test_rmse_known_value():
    # errors [1, -1, 2, -2] -> mean square = (1+1+4+4)/4 = 2.5 -> sqrt = 1.5811...
    y_true = np.array([0.0, 0.0, 0.0, 0.0])
    y_pred = np.array([1.0, -1.0, 2.0, -2.0])
    assert rmse(y_true, y_pred) == np.sqrt(2.5)


def test_rmse_shape_agnostic():
    # accepts lists / column vectors / row vectors identically
    a = rmse([1, 2, 3], [1, 2, 4])
    b = rmse(np.array([[1], [2], [3]]), np.array([1, 2, 4]))
    assert np.isclose(a, b) and np.isclose(a, np.sqrt(1 / 3))
