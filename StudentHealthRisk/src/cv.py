"""Cross-validation scheme.

Stratified random KFold on the target: keeps class ratios stable across folds, which is the
right default for a synthetic Playground tabular set where train/test share the same
distribution. The train/test drift check in reports/EDA_FINDINGS.md confirms (or overrides)
this choice; switch to GroupKFold only if a grouping/leakage signal shows up there.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold

N_SPLITS = 5
RANDOM_STATE = 42


def folds(y, n_splits: int = N_SPLITS, seed: int = RANDOM_STATE):
    """Return list of (train_idx, valid_idx), stratified on target `y`.

    `seed` varies the split for seed-averaging (RepeatedKFold); defaults to the canonical 42 so
    single-seed runs are unchanged and comparable across experiments.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))
