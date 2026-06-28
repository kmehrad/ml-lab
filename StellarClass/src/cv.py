"""Shared CV folds so every model produces blend-compatible OOF arrays."""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold

N_SPLITS = 5
SEED = 42


def get_folds(y: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return a list of (train_idx, val_idx) for StratifiedKFold on the encoded target.

    The same (n_splits, seed) is used by every model (GBDT + NN) so OOF rows align and
    predictions can be blended after the fact.
    """
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    return list(skf.split(np.zeros(len(y)), y))
