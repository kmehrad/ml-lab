import numpy as np
import pandas as pd
import pytest

from src.cv import folds
from src.data import encode_target
from src.metric import auc
from src.train import logreg_pipeline


def test_target_and_auc():
    y = encode_target(pd.Series(["No", "Yes", "No", "Yes"]))
    assert y.tolist() == [0, 1, 0, 1]
    assert auc(y, [0.1, 0.9, 0.2, 0.8]) == 1.0
    with pytest.raises(ValueError):
        auc(y, [0.1, np.nan, 0.2, 0.8])


def test_stratified_folds_disjoint_and_cover_all():
    y = np.array([0, 1] * 25)
    validation = []
    for tr, va in folds(y):
        assert not set(tr) & set(va)
        assert sum(y[va]) == len(va) // 2
        validation.extend(va)
    assert sorted(validation) == list(range(len(y)))


def test_logreg_preprocessing_fits_only_on_training_fold():
    train = pd.DataFrame({"num": [1, 2, np.nan, 4], "cat": ["a", "b", "a", "b"]})
    y = [0, 1, 0, 1]
    model = logreg_pipeline(train)
    model.fit(train, y)
    assert model.predict_proba(pd.DataFrame({"num": [3], "cat": ["unseen"]})).shape == (1, 2)
