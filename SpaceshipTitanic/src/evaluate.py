"""Metrics and cross-validation utilities for Spaceship Titanic.

The competition is scored on classification **accuracy**, so that is the focus
here. :func:`cross_validate` uses stratified k-fold (the classes are ~balanced
but stratification keeps folds comparable) and returns per-fold accuracy.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

RANDOM_STATE = 42


def accuracy(y_true, y_pred) -> float:
    """Return classification accuracy for predicted vs. true labels."""
    return float(accuracy_score(y_true, y_pred))


def cross_validate(model, X, y, n_splits: int = 5, random_state: int = RANDOM_STATE):
    """Run stratified k-fold CV and return the per-fold accuracy scores.

    ``model`` is any estimator/pipeline (e.g. from :func:`models.get_model`);
    ``X`` should be the engineered feature frame and ``y`` the binary target.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    return scores


def summarize_scores(scores) -> str:
    """Format CV scores as ``mean ± std (min–max over k folds)``."""
    scores = np.asarray(scores)
    return (
        f"{scores.mean():.4f} +/- {scores.std():.4f} "
        f"(min {scores.min():.4f}, max {scores.max():.4f}, {len(scores)} folds)"
    )
