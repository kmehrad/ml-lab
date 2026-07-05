"""Balanced-accuracy metric + decision tuning + submission builder — the single source of truth.

The competition metric is **Balanced Accuracy Score**: the macro-average of per-class recall,
``mean_j recall_j``. With the target ~86/8/6 imbalanced, this weights every class equally — a naive
"always at-risk" scores only ``1/3``. So the metric is *decision-sensitive*: the argmax of predicted
class probabilities is **not** optimal (it chases the majority class). We instead pick a per-class
multiplicative decision weight and take ``argmax(proba * weights)``, tuning those weights to maximise
**OOF** balanced accuracy, then apply the same weights to the test proba (post-hoc decision correction —
compare/tune everything on the corrected score, never raw argmax).

Convention: ``proba`` is an ``(n_rows, n_classes)`` matrix whose column ``j`` is the predicted
probability of **class code ``j``**; ``y_true`` holds those integer class codes.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score


def balanced_accuracy(y_true, y_pred) -> float:
    """Macro-average of per-class recall (the competition metric), on integer class codes."""
    return float(balanced_accuracy_score(np.asarray(y_true), np.asarray(y_pred)))


def predict_codes(proba: np.ndarray, weights=None) -> np.ndarray:
    """Decision rule: ``argmax(proba * weights)`` → integer class codes (stable tie-break)."""
    proba = np.asarray(proba, dtype=np.float64)
    if weights is not None:
        proba = proba * np.asarray(weights, dtype=np.float64).reshape(1, -1)
    # stable argmax: argmax of the reversed-then-flip gives first-max, but np.argmax already returns
    # the first occurrence on ties, which is deterministic for a fixed class order.
    return np.argmax(proba, axis=1)


def score_proba(y_true, proba: np.ndarray, weights=None) -> float:
    """Balanced accuracy of the weighted-argmax decision over a proba matrix."""
    return balanced_accuracy(y_true, predict_codes(proba, weights))


def tune_weights(
    y_true,
    proba: np.ndarray,
    *,
    rounds: int = 6,
    grid: int = 21,
    span: float = 2.0,
) -> np.ndarray:
    """Coordinate-ascent search for per-class decision weights maximising balanced accuracy.

    Starts from inverse-prior weights (the theory-optimal prior correction for balanced accuracy) and
    refines each class weight in turn on a shrinking multiplicative grid. Deterministic; fast because
    each evaluation is a single vectorised argmax + recall. Returns weights normalised to mean 1.
    """
    y = np.asarray(y_true)
    proba = np.asarray(proba, dtype=np.float64)
    n_classes = proba.shape[1]

    prior = np.bincount(y, minlength=n_classes).astype(np.float64)
    prior = np.clip(prior / prior.sum(), 1e-9, None)
    w = 1.0 / prior  # inverse-prior start

    def norm(v):
        return v / v.mean()

    w = norm(w)
    best_w = w.copy()
    best_score = score_proba(y, proba, best_w)

    log_span = np.log(span)
    for r in range(rounds):
        factors = np.exp(np.linspace(-log_span, log_span, grid))
        for j in range(n_classes):
            base = best_w.copy()
            local_best_w, local_best = best_w.copy(), best_score
            for f in factors:
                cand = base.copy()
                cand[j] = base[j] * f
                cand = norm(cand)
                s = score_proba(y, proba, cand)
                if s > local_best:
                    local_best, local_best_w = s, cand
            best_w, best_score = local_best_w, local_best
        log_span *= 0.5  # shrink the search window each round
    return best_w


def to_submission_labels(proba: np.ndarray, classes, weights=None) -> np.ndarray:
    """Turn a proba matrix into an array of predicted class-*name* strings (weighted argmax)."""
    classes = np.asarray(classes)
    return classes[predict_codes(proba, weights)]
