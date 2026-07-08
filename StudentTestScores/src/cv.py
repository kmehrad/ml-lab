"""Cross-validation scheme.

Rows are i.i.d. synthetic and train/test share the same distribution (see
reports/EDA_FINDINGS.md), so a plain random KFold is appropriate and CV should track the
leaderboard. (If folds ever look noisy, stratifying on a binned ``exam_score`` is a cheap
later refinement — not needed for the baseline.)
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import KFold

N_SPLITS = 5
RANDOM_STATE = 42


def folds(y, n_splits: int = N_SPLITS, seed: int = RANDOM_STATE):
    """Return list of (train_idx, valid_idx) from a shuffled KFold.

    `seed` varies the split for optional seed-averaging; defaults to the canonical 42 so
    single-seed runs are comparable across experiments.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(kf.split(np.zeros(len(y))))
