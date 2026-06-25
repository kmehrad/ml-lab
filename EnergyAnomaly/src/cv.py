"""Cross-validation scheme.

Train and test buildings are DISJOINT, so validation must hold out whole buildings.
A random/stratified KFold would leak per-building behaviour and overstate OOF.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold

N_SPLITS = 5


def folds(groups, n_splits: int = N_SPLITS):
    """Yield (train_idx, valid_idx) splitting on `groups` (building_id)."""
    gkf = GroupKFold(n_splits=n_splits)
    dummy = np.zeros(len(groups))
    return list(gkf.split(dummy, groups=groups))
