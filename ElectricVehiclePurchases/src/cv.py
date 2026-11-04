"""Reproducible stratified folds."""
from sklearn.model_selection import StratifiedKFold


def folds(y, n_splits: int = 5, seed: int = 42):
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(
        range(len(y)), y
    )
