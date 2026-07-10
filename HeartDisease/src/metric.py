"""ROC AUC metric — the single source of truth for scoring.

The competition metric is **ROC AUC** on the predicted probability of `Presence`
(higher is better; confirmed via the Kaggle API `evaluationMetric` field). Both OOF
scoring and blend comparison go through ``roc_auc`` here so the number we optimise is
exactly the number Kaggle grades.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def roc_auc(y_true, y_prob) -> float:
    """ROC AUC between true binary labels and predicted probabilities."""
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(y_prob, dtype=float).ravel()
    return float(roc_auc_score(y, p))
