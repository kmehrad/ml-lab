"""Balanced-accuracy helpers, including the per-class decision correction.

Balanced accuracy weights every class equally, so for an imbalanced target the argmax of raw
probabilities is rarely optimal. We search per-class multipliers `m` (applied as `proba * m` before
argmax) that maximize balanced accuracy on OOF. Only relative scale matters, so GALAXY is fixed at 1.0
and we search the other two classes on a geometric grid.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score

from . import data as D


def predict_labels(proba: np.ndarray, mult: np.ndarray | None = None) -> np.ndarray:
    if mult is not None:
        proba = proba * mult
    return proba.argmax(axis=1)


def balanced_accuracy(y_true: np.ndarray, proba: np.ndarray, mult: np.ndarray | None = None) -> float:
    return balanced_accuracy_score(y_true, predict_labels(proba, mult))


def tune_multipliers(
    y_true: np.ndarray, proba: np.ndarray, grid: np.ndarray | None = None
) -> tuple[np.ndarray, float]:
    """Grid-search per-class multipliers maximizing balanced accuracy on (y_true, proba).

    GALAXY (index 0) is fixed at 1.0; QSO (1) and STAR (2) are searched. Returns (best_mult, best_bacc).
    """
    if grid is None:
        grid = np.geomspace(0.25, 4.0, 25)
    galaxy = D.CLASS_TO_INT["GALAXY"]
    best_mult = np.ones(len(D.CLASSES))
    best = balanced_accuracy(y_true, proba)
    for mq in grid:
        for ms in grid:
            mult = np.ones(len(D.CLASSES))
            mult[D.CLASS_TO_INT["QSO"]] = mq
            mult[D.CLASS_TO_INT["STAR"]] = ms
            mult[galaxy] = 1.0
            score = balanced_accuracy(y_true, proba, mult)
            if score > best:
                best, best_mult = score, mult
    return best_mult, best
