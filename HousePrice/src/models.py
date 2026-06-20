"""Model factories, hyperparameter search spaces, and CV utilities.

Each ``*_model`` builder returns ``(estimator, param_distribution)`` for use with
:func:`tune`. A single shared :data:`KFOLD` is used for tuning *and* out-of-fold
prediction so that every model's OOF vector is row-aligned — a prerequisite for an
honest blend.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import nnls
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from . import config

MODEL_ORDER = ["enet", "lgbm", "xgb"]

# Shared fold scheme — reused everywhere so OOF predictions align across models.
KFOLD = KFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)


def rmse(y_true, y_pred) -> float:
    """Root mean squared error on the log target == the competition metric."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# --- Estimator builders -----------------------------------------------------

def enet_model() -> tuple[Pipeline, dict]:
    """ElasticNet with scaling inside the pipeline (fit per CV fold)."""
    est = Pipeline([
        ("scale", StandardScaler()),
        ("model", ElasticNet(max_iter=10000, random_state=config.RANDOM_STATE)),
    ])
    space = {
        "model__alpha": loguniform(1e-4, 1.0),
        "model__l1_ratio": uniform(0.05, 0.90),  # in [0.05, 0.95]
    }
    return est, space


def lgbm_model() -> tuple[LGBMRegressor, dict]:
    est = LGBMRegressor(
        random_state=config.RANDOM_STATE, n_jobs=-1, verbose=-1,
    )
    space = {
        "n_estimators": randint(500, 2500),
        "learning_rate": loguniform(0.01, 0.1),
        "num_leaves": randint(15, 64),
        "min_child_samples": randint(5, 30),
        "subsample": uniform(0.7, 0.3),         # [0.7, 1.0]
        "subsample_freq": randint(1, 4),
        "colsample_bytree": uniform(0.6, 0.4),  # [0.6, 1.0]
        "reg_lambda": loguniform(1e-3, 10.0),
        "reg_alpha": loguniform(1e-3, 10.0),
    }
    return est, space


def xgb_model() -> tuple[XGBRegressor, dict]:
    est = XGBRegressor(
        tree_method="hist", enable_categorical=True,
        random_state=config.RANDOM_STATE, n_jobs=-1,
    )
    space = {
        "n_estimators": randint(500, 2500),
        "learning_rate": loguniform(0.01, 0.1),
        "max_depth": randint(3, 7),
        "min_child_weight": randint(1, 8),
        "subsample": uniform(0.7, 0.3),
        "colsample_bytree": uniform(0.6, 0.4),
        "reg_lambda": loguniform(1e-3, 10.0),
        "reg_alpha": loguniform(1e-3, 10.0),
    }
    return est, space


# --- Tuning & out-of-fold prediction ---------------------------------------

def tune(estimator, space, X, y, n_iter: int = 40):
    """RandomizedSearchCV on the shared folds. Returns (best_estimator, params, cv_rmse)."""
    search = RandomizedSearchCV(
        estimator, space, n_iter=n_iter, cv=KFOLD,
        scoring="neg_root_mean_squared_error",
        random_state=config.RANDOM_STATE, n_jobs=-1, refit=True,
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_, -search.best_score_


def oof_predict(estimator, X, y) -> np.ndarray:
    """Out-of-fold predictions on the shared folds (honest, blend-ready)."""
    return cross_val_predict(clone(estimator), X, y, cv=KFOLD, n_jobs=-1)


# --- Ensemble blend ---------------------------------------------------------

def nnls_blend(oof_matrix: np.ndarray, y) -> tuple[np.ndarray, np.ndarray, float]:
    """Non-negative weights (summing to 1) over OOF predictions.

    Returns (weights, blended_oof, blend_rmse).
    """
    weights, _ = nnls(oof_matrix, np.asarray(y))
    if weights.sum() == 0:                     # degenerate guard
        weights = np.ones(oof_matrix.shape[1])
    weights = weights / weights.sum()
    blended = oof_matrix @ weights
    return weights, blended, rmse(y, blended)
