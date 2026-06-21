"""Model definitions for Spaceship Titanic (binary classification).

Each factory returns a complete scikit-learn :class:`~sklearn.pipeline.Pipeline`
that bundles the feature preprocessor (:func:`features.build_preprocessor`) with
a classifier, so callers can fit straight on the *engineered* DataFrame returned
by :func:`features.engineer_features` and let the pipeline handle encoding.

Linear models get ``scale_numeric=True``; tree ensembles do not need scaling.

Optional gradient-boosting libraries (XGBoost / LightGBM / CatBoost) are
imported lazily so the project still works when they are not installed — call
:func:`available_models` to see what can be built in the current environment.
"""
from __future__ import annotations

import json

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .data import PROJECT_ROOT
from .features import build_preprocessor

RANDOM_STATE = 42

try:  # optional dependency
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:  # pragma: no cover - depends on environment
    HAS_XGBOOST = False

try:  # optional dependency
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:  # pragma: no cover - depends on environment
    HAS_LIGHTGBM = False

try:  # optional dependency
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:  # pragma: no cover - depends on environment
    HAS_CATBOOST = False


def _build_classifier(name: str):
    """Return a bare (unpipelined) estimator and whether it wants scaled inputs."""
    if name == "logreg":
        return LogisticRegression(max_iter=1000, random_state=RANDOM_STATE), True
    if name in ("rf", "randomforest"):
        return RandomForestClassifier(
            n_estimators=400, n_jobs=-1, random_state=RANDOM_STATE,
        ), False
    if name in ("hgb", "gb", "histgb"):
        return HistGradientBoostingClassifier(random_state=RANDOM_STATE), False
    if name == "xgb":
        if not HAS_XGBOOST:
            raise ValueError("xgboost is not installed")
        return XGBClassifier(
            n_estimators=600, learning_rate=0.05, max_depth=5,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=-1,
        ), False
    if name in ("lgbm", "lightgbm"):
        if not HAS_LIGHTGBM:
            raise ValueError("lightgbm is not installed")
        return LGBMClassifier(
            n_estimators=600, learning_rate=0.05, random_state=RANDOM_STATE,
            n_jobs=-1, verbose=-1,
        ), False
    if name in ("cat", "catboost"):
        if not HAS_CATBOOST:
            raise ValueError("catboost is not installed")
        return CatBoostClassifier(
            iterations=700, depth=6, learning_rate=0.05,
            loss_function="Logloss", random_seed=RANDOM_STATE,
            verbose=False, thread_count=-1,
        ), False
    raise ValueError(f"Unknown model name: {name!r}. See available_models().")


def get_model(name: str = "hgb", **preprocessor_kwargs) -> Pipeline:
    """Return a fresh ``preprocessor + classifier`` pipeline by name.

    Parameters
    ----------
    name:
        One of :func:`available_models` (e.g. ``"logreg"``, ``"hgb"``,
        ``"catboost"``, or the ``"blend"`` ensemble when installed).
    **preprocessor_kwargs:
        Forwarded to :func:`features.build_preprocessor` (e.g. to override
        feature lists). ``scale_numeric`` is set automatically per model but can
        be overridden here.
    """
    if name == "blend":
        if not HAS_CATBOOST:
            raise ValueError("The hgb+catboost blend requires catboost")
        return get_ensemble(
            kind="voting", members=("hgb", "catboost"), **preprocessor_kwargs
        )

    clf, wants_scaling = _build_classifier(name)
    preprocessor_kwargs.setdefault("scale_numeric", wants_scaling)
    pre = build_preprocessor(**preprocessor_kwargs)
    return Pipeline([("pre", pre), ("clf", clf)])


def build_tuned_model(name: str, params: dict, **preprocessor_kwargs) -> Pipeline:
    """Build a ``preprocessor + classifier`` pipeline with explicit classifier params.

    Used by :mod:`tune` (Optuna) and by :mod:`train` when loading saved best
    params. ``params`` overrides the classifier's defaults.
    """
    clf, wants_scaling = _build_classifier(name)
    clf.set_params(**params)
    preprocessor_kwargs.setdefault("scale_numeric", wants_scaling)
    pre = build_preprocessor(**preprocessor_kwargs)
    return Pipeline([("pre", pre), ("clf", clf)])


def _load_tuned_params(name: str) -> dict:
    """Load saved Optuna best params for ``name`` (empty dict if none saved)."""
    path = PROJECT_ROOT / "reports" / f"best_params_{name}.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


def get_ensemble(
    kind: str = "voting",
    members: tuple[str, ...] = ("hgb", "lgbm", "xgb"),
    use_tuned: bool = False,
    **preprocessor_kwargs,
) -> Pipeline:
    """Build an ensemble of base classifiers over a single shared preprocessor.

    Parameters
    ----------
    kind:
        ``"voting"`` (soft-voting average of predicted probabilities) or
        ``"stack"`` (out-of-fold stacking with a logistic-regression meta-model).
    members:
        Base model names (must be tree-based / unscaled-friendly).
    use_tuned:
        Apply saved Optuna params (``reports/best_params_<name>.json``) when set.
    """
    estimators = []
    for name in members:
        clf, _ = _build_classifier(name)
        if use_tuned:
            params = _load_tuned_params(name)
            if params:
                clf.set_params(**params)
        estimators.append((name, clf))

    if kind == "voting":
        ensemble = VotingClassifier(estimators, voting="soft", n_jobs=-1)
    elif kind in ("stack", "stacking"):
        ensemble = StackingClassifier(
            estimators,
            final_estimator=LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            cv=5, n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown ensemble kind: {kind!r} (use 'voting' or 'stack').")

    # Tree-based members don't need scaling; share one unscaled preprocessor.
    preprocessor_kwargs.setdefault("scale_numeric", False)
    pre = build_preprocessor(**preprocessor_kwargs)
    return Pipeline([("pre", pre), ("clf", ensemble)])


def available_models() -> list[str]:
    """List the model names that can be built in the current environment."""
    names = ["logreg", "rf", "hgb"]
    if HAS_XGBOOST:
        names.append("xgb")
    if HAS_LIGHTGBM:
        names.append("lgbm")
    if HAS_CATBOOST:
        names.extend(["catboost", "blend"])
    return names


def default_model() -> str:
    """Return the strongest available default model."""
    return "blend" if HAS_CATBOOST else "hgb"
