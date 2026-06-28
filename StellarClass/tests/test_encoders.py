import numpy as np
import pandas as pd

from src import data as D
from src import encoders as E


def _frame(spectral, population):
    return pd.DataFrame(
        {"spectral_type": spectral, "galaxy_population": population}
    )


def test_transform_columns():
    X = _frame(["M", "A/F", "G/K", "O/B"], ["Red_Sequence", "Blue_Cloud", "Red_Sequence", "Blue_Cloud"])
    y = np.array([0, 1, 2, 1])
    enc = E.CategoryEncoders().fit(X, y)
    out = enc.transform(X)
    # freq + target-encoding cols for each of the 3 source columns (2 cats + interaction)
    assert any(c.startswith("freq_spectral_type") for c in out.columns)
    assert f"te_spectral_type_{D.CLASSES[0]}" in out.columns
    assert len(out) == len(X)
    assert out.notna().all().all()


def test_target_encoding_uses_only_train_fold():
    # category 'M' is GALAXY(0) only in TRAIN; a val 'M' row must encode toward GALAXY using
    # train statistics, never peeking at the val label.
    Xtr = _frame(["M", "M", "A/F", "A/F"], ["Red_Sequence"] * 4)
    ytr = np.array([0, 0, 1, 1])  # M->GALAXY, A/F->QSO
    enc = E.CategoryEncoders(smoothing=1.0).fit(Xtr, ytr)

    Xval = _frame(["M"], ["Red_Sequence"])
    out = enc.transform(Xval)
    te = out.iloc[0]
    # GALAXY encoding for 'M' should dominate the other classes
    assert te[f"te_spectral_type_{D.CLASSES[0]}"] > te[f"te_spectral_type_{D.CLASSES[1]}"]


def test_unseen_category_falls_back_to_prior():
    Xtr = _frame(["M", "A/F"], ["Red_Sequence", "Blue_Cloud"])
    ytr = np.array([0, 1])
    enc = E.CategoryEncoders().fit(Xtr, ytr)
    Xnew = _frame(["UNSEEN"], ["Blue_Cloud"])
    out = enc.transform(Xnew)
    assert out["freq_spectral_type"].iloc[0] == 0.0
    # target-encoding of unseen category equals the global prior
    assert np.isclose(out[f"te_spectral_type_{D.CLASSES[0]}"].iloc[0], enc.priors[0])
