"""Model definitions.

Each factory returns a fresh estimator. XGBoost and LightGBM are imported
lazily so the project still works when they are not installed; call
:func:`available_models` to see what can actually be built in the current
environment.
"""
from __future__ import annotations

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge

try:  # optional dependency
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:  # pragma: no cover - depends on environment
    HAS_XGBOOST = False

try:  # optional dependency
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:  # pragma: no cover - depends on environment
    HAS_LIGHTGBM = False

try:  # optional dependency
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:  # pragma: no cover - depends on environment
    HAS_CATBOOST = False

RANDOM_STATE = 42


def ridge(**kwargs):
    return Ridge(alpha=kwargs.pop("alpha", 10.0), random_state=RANDOM_STATE, **kwargs)


def lasso(**kwargs):
    return Lasso(alpha=kwargs.pop("alpha", 0.001), random_state=RANDOM_STATE, max_iter=10000, **kwargs)


def elasticnet(**kwargs):
    return ElasticNet(
        alpha=kwargs.pop("alpha", 0.001),
        l1_ratio=kwargs.pop("l1_ratio", 0.5),
        random_state=RANDOM_STATE,
        max_iter=10000,
        **kwargs,
    )


def random_forest(**kwargs):
    params = dict(n_estimators=500, n_jobs=-1, random_state=RANDOM_STATE)
    params.update(kwargs)
    return RandomForestRegressor(**params)


def gradient_boosting(**kwargs):
    params = dict(n_estimators=500, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE)
    params.update(kwargs)
    return GradientBoostingRegressor(**params)


def xgboost(**kwargs):
    if not HAS_XGBOOST:
        raise ImportError("xgboost is not installed; `pip install xgboost`.")
    params = dict(
        n_estimators=1000, learning_rate=0.05, max_depth=3,
        subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE, n_jobs=-1,
    )
    params.update(kwargs)
    return xgb.XGBRegressor(**params)


def lightgbm(**kwargs):
    if not HAS_LIGHTGBM:
        raise ImportError("lightgbm is not installed; `pip install lightgbm`.")
    params = dict(
        n_estimators=1000, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE,
        n_jobs=-1, verbose=-1,
    )
    params.update(kwargs)
    return lgb.LGBMRegressor(**params)


def catboost(**kwargs):
    if not HAS_CATBOOST:
        raise ImportError("catboost is not installed; `pip install catboost`.")
    params = dict(
        iterations=1000, learning_rate=0.05, depth=6,
        random_seed=RANDOM_STATE, verbose=False,
    )
    params.update(kwargs)
    return CatBoostRegressor(**params)


# Registry of always-available model factories.
MODEL_REGISTRY = {
    "ridge": ridge,
    "lasso": lasso,
    "elasticnet": elasticnet,
    "random_forest": random_forest,
    "gradient_boosting": gradient_boosting,
}
if HAS_XGBOOST:
    MODEL_REGISTRY["xgboost"] = xgboost
if HAS_LIGHTGBM:
    MODEL_REGISTRY["lightgbm"] = lightgbm
if HAS_CATBOOST:
    MODEL_REGISTRY["catboost"] = catboost

# Linear models benefit from scaled numeric features.
NEEDS_SCALING = {"ridge", "lasso", "elasticnet"}


def available_models() -> list[str]:
    """Names of models that can be built in this environment."""
    return list(MODEL_REGISTRY)


def get_model(name: str, **kwargs):
    """Build a model by name from :data:`MODEL_REGISTRY`."""
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model {name!r}. Available: {available_models()}")
    return MODEL_REGISTRY[name](**kwargs)
