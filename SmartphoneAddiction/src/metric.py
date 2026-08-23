import numpy as np
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score


def metrics(y, pred):
    y, pred = np.asarray(y), np.asarray(pred, dtype=float)
    if y.shape != pred.shape or not np.isfinite(pred).all():
        raise ValueError("targets and finite predictions must align")
    return {"roc_auc": float(roc_auc_score(y, pred)),
            "pr_auc": float(average_precision_score(y, pred)),
            "log_loss": float(log_loss(y, np.clip(pred, 1e-7, 1 - 1e-7)))}
