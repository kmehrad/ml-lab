"""Cross-validation scheme.

Rows are i.i.d. synthetic with no train/test drift (see reports/EDA_FINDINGS.md), and the
target is near-balanced (44.8% positive). ``StratifiedKFold`` keeps the class ratio stable
per fold for cheap variance reduction on OOF AUC.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold

N_SPLITS = 5
RANDOM_STATE = 42


def folds(y, n_splits: int = N_SPLITS, seed: int = RANDOM_STATE):
    """Return list of (train_idx, valid_idx) from a shuffled StratifiedKFold.

    `seed` varies the split for optional seed-averaging; defaults to the canonical 42 so
    single-seed runs are comparable across experiments.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))
