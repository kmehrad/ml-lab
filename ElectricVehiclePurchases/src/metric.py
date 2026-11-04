"""Competition scoring: positive-class probability ROC-AUC."""
import numpy as np
from sklearn.metrics import roc_auc_score


def auc(y, probability) -> float:
    p = np.asarray(probability, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all():
        raise ValueError("probabilities must be a finite 1-D vector")
    return float(roc_auc_score(y, p))
