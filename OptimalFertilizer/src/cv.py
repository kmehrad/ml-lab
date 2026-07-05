"""Cross-validation scheme.

Rows are i.i.d. synthetic and train/test share the same distribution (no drift; see
reports/EDA_FINDINGS.md), so a stratified random KFold on the target is appropriate and
CV should track the leaderboard. Stratifying keeps the 7 class ratios stable across folds.
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
