"""Hyperparameter tuning with Optuna for Spaceship Titanic.

Optimizes a gradient-boosting classifier's hyperparameters to maximize mean
stratified k-fold **accuracy**, reusing the same feature pipeline as the rest of
the project (:func:`features.build_preprocessor`).

Usage::

    python -m src.tune --model lgbm --n-trials 60
    python -m src.tune --model hgb  --n-trials 40

The best parameters are printed and saved to
``reports/best_params_<model>.json`` so :mod:`train` can rebuild the model.
"""
from __future__ import annotations

import argparse
import json

import optuna

from . import data
from .evaluate import cross_validate, summarize_scores
from .features import build_preprocessor, engineer_features, split_X_y
from .models import RANDOM_STATE, build_tuned_model

# Models that don't need scaled inputs use the preprocessor as-is.
SUPPORTED = {"lgbm", "hgb", "xgb"}


def _suggest_params(trial: optuna.Trial, model_name: str) -> dict:
    """Return a hyperparameter dict sampled from the model's search space."""
    if model_name == "lgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
    if model_name == "hgb":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_iter": trial.suggest_int("max_iter", 200, 1000, step=100),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 255),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 100),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-8, 10.0, log=True),
        }
    if model_name == "xgb":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
    raise ValueError(f"Tuning not supported for model {model_name!r}; choose from {sorted(SUPPORTED)}")


def tune(model_name: str = "lgbm", n_trials: int = 60, n_splits: int = 5):
    """Run an Optuna study and return ``(best_params, best_score, study)``."""
    if model_name not in SUPPORTED:
        raise ValueError(f"Tuning not supported for {model_name!r}; choose from {sorted(SUPPORTED)}")

    X, y = split_X_y(engineer_features(data.load_train()))

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial, model_name)
        model = build_tuned_model(model_name, params)
        scores = cross_validate(model, X, y, n_splits=n_splits)
        return scores.mean()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"Best CV accuracy: {study.best_value:.4f}")
    print(f"Best params: {json.dumps(study.best_params, indent=2)}")

    # Persist for reuse by train.py.
    out_dir = data.PROJECT_ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"best_params_{model_name}.json"
    out_path.write_text(json.dumps(study.best_params, indent=2))
    print(f"Saved best params to {out_path}")

    return study.best_params, study.best_value, study


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune a Spaceship Titanic model with Optuna.")
    parser.add_argument("--model", default="lgbm", choices=sorted(SUPPORTED))
    parser.add_argument("--n-trials", type=int, default=60)
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()
    tune(model_name=args.model, n_trials=args.n_trials, n_splits=args.n_splits)


if __name__ == "__main__":
    main()
