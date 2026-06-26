"""Shared cross-validation folds.

All models (classic + transformer) iterate the SAME StratifiedKFold splits so their
OOF/test prediction arrays line up row-for-row and can be blended.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold

N_SPLITS = 5
SEED = 42


def make_folds(y: np.ndarray, n_splits: int = N_SPLITS, seed: int = SEED):
    """Return a list of ``(train_idx, val_idx)`` arrays — deterministic given y."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    dummy = np.zeros(len(y))
    return list(skf.split(dummy, y))
