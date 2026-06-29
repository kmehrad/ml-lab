"""Optuna hyperparameter search for the GBDTs, optimizing balanced accuracy.

For speed the objective uses a stratified **subsample** with 3-fold CV and early stopping; the best
params are saved to experiments/best_params_{model}.json and consumed automatically by
`src/train.py::_load_params` for the full 5-fold retrain.

Usage (run on the GPU box for speed):
    uv run python -m src.tune --model lgbm --trials 50 --subsample 200000
    uv run python -m src.tune --model xgb  --trials 40 --subsample 150000 --gpu
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold

from . import cv
from . import data as D
from . import features as F
from . import metrics as M

PARAMS_DIR = Path(__file__).resolve().parent.parent / "experiments"
N_CLASS = len(D.CLASSES)
TUNE_FOLDS = 3


def _subsample(X, y, n, seed=cv.SEED):
    if n is None or n >= len(y):
        return X, y
    idx, _ = next(StratifiedKFold(n_splits=max(2, round(len(y) / n)), shuffle=True,
                                  random_state=seed).split(X, y))
    return X.iloc[idx].reset_index(drop=True), y[idx]


def _cv_bacc(make_fit, X, y) -> float:
    skf = StratifiedKFold(n_splits=TUNE_FOLDS, shuffle=True, random_state=cv.SEED)
    oof = np.zeros((len(y), N_CLASS))
    for tr_idx, va_idx in skf.split(X, y):
        proba = make_fit(X.iloc[tr_idx], y[tr_idx], X.iloc[va_idx], y[va_idx])
        oof[va_idx] = proba
    return M.balanced_accuracy(y, oof)


def _lgbm_objective(X, y, cat_cols):
    from lightgbm import LGBMClassifier, early_stopping

    def objective(trial):
        params = dict(
            objective="multiclass", num_class=N_CLASS, class_weight="balanced",
            n_estimators=2000, n_jobs=-1, random_state=cv.SEED, verbosity=-1,
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
            num_leaves=trial.suggest_int("num_leaves", 31, 255),
            max_depth=trial.suggest_int("max_depth", 4, 12),
            min_child_samples=trial.suggest_int("min_child_samples", 20, 200),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            subsample_freq=1,
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        )

        def fit(Xtr, ytr, Xva, yva):
            m = LGBMClassifier(**params)
            m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="multi_logloss",
                  categorical_feature=cat_cols, callbacks=[early_stopping(50, verbose=False)])
            return m.predict_proba(Xva)

        return _cv_bacc(fit, X, y)

    return objective


def _xgb_objective(X, y, cat_cols, gpu):
    from xgboost import XGBClassifier

    def objective(trial):
        params = dict(
            objective="multi:softprob", num_class=N_CLASS, tree_method="hist",
            device="cuda" if gpu else "cpu", enable_categorical=True,
            n_estimators=2000, n_jobs=-1, random_state=cv.SEED,
            early_stopping_rounds=50, eval_metric="mlogloss",
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
            max_depth=trial.suggest_int("max_depth", 4, 12),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 20),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        )
        w = np.array([1.0 / np.bincount(y, minlength=N_CLASS)[c] for c in y])

        def fit(Xtr, ytr, Xva, yva):
            wtr = w[Xtr.index.to_numpy()] if hasattr(Xtr, "index") else None
            m = XGBClassifier(**params)
            m.fit(Xtr, ytr, sample_weight=wtr, eval_set=[(Xva, yva)], verbose=False)
            return m.predict_proba(Xva)

        return _cv_bacc(fit, X, y)

    return objective


def tune(model: str, trials: int, subsample: int, gpu: bool) -> dict:
    tr = D.load_train()
    y = D.encode_target(tr[D.TARGET])
    X = F.add_features(tr)
    cat_cols = F.CATEGORICAL_FEATURES
    X, y = _subsample(X, y, subsample)
    print(f"tuning {model} on {len(y)} rows, {trials} trials, {TUNE_FOLDS}-fold")

    obj = (_lgbm_objective(X, y, cat_cols) if model == "lgbm"
           else _xgb_objective(X.reset_index(drop=True), y, cat_cols, gpu))
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=cv.SEED))
    study.optimize(obj, n_trials=trials, show_progress_bar=False)

    print(f"best balanced accuracy (subsample {TUNE_FOLDS}-fold): {study.best_value:.5f}")
    print(f"best params: {study.best_params}")
    PARAMS_DIR.mkdir(parents=True, exist_ok=True)
    out = PARAMS_DIR / f"best_params_{model}.json"
    json.dump(study.best_params, open(out, "w"), indent=2)
    print(f"wrote {out}")
    return study.best_params


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["lgbm", "xgb"])
    p.add_argument("--trials", type=int, default=50)
    p.add_argument("--subsample", type=int, default=200000)
    p.add_argument("--gpu", action="store_true")
    a = p.parse_args()
    tune(a.model, a.trials, a.subsample, a.gpu)