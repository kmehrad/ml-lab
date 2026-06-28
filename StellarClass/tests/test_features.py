import numpy as np
import pandas as pd

from src import data as D
from src import features as F
from src import metrics as M


def _raw_frame(n=20):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({c: rng.normal(20, 1, n) for c in D.NUMERIC_COLS})
    df["redshift"] = rng.uniform(-0.01, 3, n)
    df["spectral_type"] = rng.choice(["M", "A/F", "G/K", "O/B"], n)
    df["galaxy_population"] = rng.choice(["Red_Sequence", "Blue_Cloud"], n)
    df[D.ID_COL] = np.arange(n)
    return df


def test_add_features_columns_and_no_nan():
    X = F.add_features(_raw_frame())
    assert list(X.columns) == F.ALL_FEATURES
    assert X[F.NUMERIC_FEATURES].notna().all().all()


def test_color_indices_are_differences():
    df = _raw_frame()
    X = F.add_features(df)
    assert np.allclose(X["u_g"], df["u"] - df["g"])
    assert np.allclose(X["i_z"], df["i"] - df["z"])


def test_all_pairwise_colors_present():
    assert len(F.COLOR_COLS) == 10  # C(5,2)
    X = F.add_features(_raw_frame())
    for c in F.COLOR_COLS:
        assert c in X.columns


def test_redshift_color_interactions():
    df = _raw_frame()
    X = F.add_features(df)
    assert np.allclose(X["rz_u_g"], df["redshift"] * (df["u"] - df["g"]))


def test_redshift_bin_is_valid_index():
    X = F.add_features(_raw_frame())
    assert X["redshift_bin"].notna().all()
    assert X["redshift_bin"].between(0, len(F.REDSHIFT_BIN_EDGES) - 2).all()


def test_magnitude_aggregates():
    df = _raw_frame()
    X = F.add_features(df)
    assert np.allclose(X["mag_mean"], df[F.BANDS].mean(axis=1))
    assert np.allclose(X["mag_range"], df[F.BANDS].max(axis=1) - df[F.BANDS].min(axis=1))


def test_categoricals_are_category_dtype():
    X = F.add_features(_raw_frame())
    for c in F.CATEGORICAL_FEATURES:
        assert str(X[c].dtype) == "category"


def test_train_test_feature_parity():
    a = F.add_features(_raw_frame(10))
    b = F.add_features(_raw_frame(5))
    assert list(a.columns) == list(b.columns)


def test_tune_multipliers_improves_or_keeps_balanced_accuracy():
    rng = np.random.default_rng(1)
    # imbalanced 3-class toy where argmax under-predicts minority classes
    y = np.array([0] * 60 + [1] * 25 + [2] * 15)
    proba = rng.dirichlet(np.ones(3), size=len(y))
    proba[:, 0] += 0.3  # bias toward majority
    proba = proba / proba.sum(1, keepdims=True)
    base = M.balanced_accuracy(y, proba)
    mult, tuned = M.tune_multipliers(y, proba)
    assert tuned >= base
    assert mult[D.CLASS_TO_INT["GALAXY"]] == 1.0
