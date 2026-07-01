"""Shared cross-validation folds for BirdSpecies (multi-label).

Every model uses the **same** 5-fold split so OOF prediction arrays line up
row-for-row and can be blended. Splits are multilabel-stratified over the 87
class columns (``MultilabelStratifiedKFold``), which keeps rare-class prevalence
roughly balanced across folds — important given the strong class imbalance.
"""
from __future__ import annotations

import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

N_SPLITS = 5
SEED = 42


def make_folds(Y: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return ``[(train_idx, val_idx), ...]`` for the rows of ``Y`` (n×87 binary).

    Deterministic given ``Y`` (shuffle seeded with ``SEED``).
    """
    Y = np.asarray(Y)
    mskf = MultilabelStratifiedKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=SEED
    )
    X = np.zeros((len(Y), 1))
    return list(mskf.split(X, Y))
