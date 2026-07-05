"""MAP@3 metric + top-3 submission-string builder — the single source of truth.

The competition metric is **Mean Average Precision @ 3**. Each row has exactly one true
label, so per-row Average Precision @ 3 reduces to `1/rank` if the true class is ranked
1st/2nd/3rd (→ 1, 1/2, 1/3), else 0. MAP@3 is the mean over rows.

Convention: `proba` is an `(n_rows, n_classes)` matrix whose column `j` is the predicted
probability of **class code `j`**, and `y_true` holds those integer class codes. Ranking
uses a *stable* argsort so ties break by class code (deterministic, reproducible).
"""
from __future__ import annotations

import numpy as np

_WEIGHTS = np.array([1.0, 1.0 / 2.0, 1.0 / 3.0])  # reciprocal rank for positions 1,2,3


def top3_indices(proba: np.ndarray) -> np.ndarray:
    """Return the `(n_rows, 3)` class-code indices, best-first (stable tie-break)."""
    return np.argsort(-np.asarray(proba), axis=1, kind="stable")[:, :3]


def mapk3(y_true, proba: np.ndarray) -> float:
    """MAP@3 for single-label rows. `y_true`: int class codes; `proba`: (n, n_classes)."""
    proba = np.asarray(proba)
    y = np.asarray(y_true).reshape(-1, 1)
    hits = top3_indices(proba) == y                # (n, 3) boolean
    return float((hits * _WEIGHTS).sum(axis=1).mean())


def to_submission_strings(proba: np.ndarray, classes) -> list[str]:
    """Turn a proba matrix into `'C1 C2 C3'` strings (top-3 class *names*, best-first)."""
    classes = np.asarray(classes)
    labels = classes[top3_indices(proba)]          # (n, 3) of class names
    return [" ".join(row) for row in labels]
