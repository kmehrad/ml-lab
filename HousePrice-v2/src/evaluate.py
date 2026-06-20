"""Evaluation: the competition metric (RMSLE) and cross-validation helpers.

Kaggle scores this competition with the RMSE between the log of the predicted
price and the log of the observed price -- i.e. RMSLE. A common strategy is to
train on ``log1p(SalePrice)`` and measure plain RMSE in that log space, which
is mathematically equivalent to RMSLE in the original space.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import KFold, cross_val_score


def rmsle(y_true, y_pred) -> float:
    """Root Mean Squared Log Error on original-scale prices."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    # Guard against negative predictions that would break log1p.
    y_pred = np.clip(y_pred, a_min=0, a_max=None)
    return float(np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred))))


def rmse(y_true, y_pred) -> float:
    """Plain RMSE; use on log-space targets for equivalence to RMSLE."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# Scorer for use with sklearn search/CV when the target is in log space.
# Higher is better, so we negate RMSE.
neg_rmse_scorer = make_scorer(rmse, greater_is_better=False)


def make_kfold(n_splits: int = 5, seed: int = 42) -> KFold:
    return KFold(n_splits=n_splits, shuffle=True, random_state=seed)


def cross_validate_rmse(estimator, X, y_log, n_splits: int = 5, seed: int = 42):
    """Cross-validate an estimator on a log-space target.

    ``y_log`` should already be ``log1p(SalePrice)``. Returns the array of
    per-fold RMSE scores (in log space == RMSLE).
    """
    cv = make_kfold(n_splits=n_splits, seed=seed)
    scores = cross_val_score(estimator, X, y_log, scoring=neg_rmse_scorer, cv=cv)
    return -scores


def summarize_scores(scores) -> dict:
    """Return mean/std summary for a set of CV scores."""
    scores = np.asarray(scores)
    return {"mean": float(scores.mean()), "std": float(scores.std()), "folds": scores.tolist()}


def oof_predictions(estimator_factory, X, y_log, n_splits: int = 5, seed: int = 42):
    """Return out-of-fold predictions for a fresh estimator per fold.

    ``estimator_factory`` is a zero-arg callable returning an unfitted model,
    so each fold trains a clean instance. Predictions are returned in the
    original row order, which lets several models' OOF vectors be averaged
    into an ensemble and scored under the *same* CV split (no leakage).
    """
    X = np.asarray(X)
    y_log = np.asarray(y_log)
    cv = make_kfold(n_splits=n_splits, seed=seed)
    oof = np.zeros(len(X), dtype=float)
    for train_idx, val_idx in cv.split(X):
        model = estimator_factory()
        model.fit(X[train_idx], y_log[train_idx])
        oof[val_idx] = model.predict(X[val_idx])
    return oof
