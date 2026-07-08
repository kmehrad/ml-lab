"""RMSE metric — the single source of truth for scoring.

The competition metric is **Root Mean Squared Error** on the predicted ``exam_score``
(lower is better). Both OOF scoring and blend comparison go through ``rmse`` here so the
number we optimise is exactly the number Kaggle grades.
"""
from __future__ import annotations

import numpy as np


def rmse(y_true, y_pred) -> float:
    """Root mean squared error between two 1-D arrays."""
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(y_pred, dtype=float).ravel()
    return float(np.sqrt(np.mean((y - p) ** 2)))
