import numpy as np
from sklearn.model_selection import StratifiedKFold


def make_folds(y, n_splits=5, seed=42):
    y = np.asarray(y)
    folds = np.full(len(y), -1, dtype=np.int8)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, va) in enumerate(splitter.split(np.zeros(len(y)), y)):
        folds[va] = fold
    if (folds < 0).any():
        raise RuntimeError("fold assignment incomplete")
    return folds
