"""GBDT CV training -> saved OOF + bagged test probabilities.

Usage:
    uv run python -m src.train --model lgbm                 # also: xgb, cat, all
    uv run python -m src.train --model lgbm --encode        # add fold-safe category encoders
    uv run python -m src.train --model lgbm --use-original  # add SDSS17 rows to train folds only
    uv run python -m src.train --model lgbm --tag fe2 --encode --use-original

Each model writes to experiments/artifacts/ (suffixed by --tag if given):
    {model}{tag}_oof.npy   (n_train, 3)  out-of-fold class probabilities
    {model}{tag}_test.npy  (n_test, 3)   fold-averaged test class probabilities
    {model}{tag}_metrics.json
Shared once:
    y_true.npy, test_id.npy
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import cv
from . import data as D
from . import features as F
from . import metrics as M
from .encoders import CategoryEncoders

ART = Path(__file__).resolve().parent.parent / "experiments" / "artifacts"
PARAMS_DIR = Path(__file__).resolve().parent.parent / "experiments"
N_CLASS = len(D.CLASSES)


def _load_params(name: str) -> dict:
    """Load tuned hyperparameters from experiments/best_params_{name}.json if present."""
    p = PARAMS_DIR / f"best_params_{name}.json"
    return json.load(open(p)) if p.exists() else {}


def _class_weights(y: np.ndarray) -> dict[int, float]:
    counts = np.bincount(y, minlength=N_CLASS)
    n = len(y)
    return {c: n / (N_CLASS * counts[c]) for c in range(N_CLASS)}


def _fit_lgbm(Xtr, ytr, Xva, yva, cat_cols, params=None):
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation

    kw = dict(
        objective="multiclass", num_class=N_CLASS, class_weight="balanced",
        n_estimators=3000, learning_rate=0.03, num_leaves=127, max_depth=-1,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        reg_lambda=1.0, min_child_samples=50, n_jobs=-1, random_state=cv.SEED, verbosity=-1,
    )
    kw.update(params or {})
    model = LGBMClassifier(**kw)
    model.fit(
        Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="multi_logloss",
        categorical_feature=cat_cols,
        callbacks=[early_stopping(100, verbose=False), log_evaluation(0)],
    )
    return model.predict_proba(Xva), model


def _fit_xgb(Xtr, ytr, Xva, yva, cat_cols, params=None):
    from xgboost import XGBClassifier

    w = np.array([_class_weights(ytr)[c] for c in ytr])
    kw = dict(
        objective="multi:softprob", num_class=N_CLASS, tree_method="hist",
        enable_categorical=True, n_estimators=3000, learning_rate=0.03, max_depth=8,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, min_child_weight=5,
        n_jobs=-1, random_state=cv.SEED, early_stopping_rounds=100, eval_metric="mlogloss",
    )
    kw.update(params or {})
    model = XGBClassifier(**kw)
    model.fit(Xtr, ytr, sample_weight=w, eval_set=[(Xva, yva)], verbose=False)
    return model.predict_proba(Xva), model


def _fit_cat(Xtr, ytr, Xva, yva, cat_cols, params=None):
    from catboost import CatBoostClassifier, Pool

    # CatBoost wants categorical columns as strings (NaN -> 'nan').
    Xtr = Xtr.copy(); Xva = Xva.copy()
    for c in cat_cols:
        Xtr[c] = Xtr[c].astype(str); Xva[c] = Xva[c].astype(str)
    train_pool = Pool(Xtr, ytr, cat_features=cat_cols)
    val_pool = Pool(Xva, yva, cat_features=cat_cols)
    kw = dict(
        loss_function="MultiClass", classes_count=N_CLASS, auto_class_weights="Balanced",
        iterations=3000, learning_rate=0.03, depth=8, l2_leaf_reg=3.0,
        random_seed=cv.SEED, thread_count=-1, verbose=False, early_stopping_rounds=100,
    )
    kw.update(params or {})
    model = CatBoostClassifier(**kw)
    model.fit(train_pool, eval_set=val_pool)
    return model.predict_proba(Xva), model


FITTERS = {"lgbm": _fit_lgbm, "xgb": _fit_xgb, "cat": _fit_cat}


def _prep_cat(X, cat_cols):
    """CatBoost needs categorical columns as strings (NaN -> 'nan')."""
    X = X.copy()
    for c in cat_cols:
        X[c] = X[c].astype(str)
    return X


def _add_encoders(Xtr, ytr, Xva, Xte):
    """Fit fold-safe category encoders on the train fold; return augmented (Xtr, Xva, Xte)."""
    enc = CategoryEncoders().fit(Xtr, ytr)
    etr, eva, ete = enc.transform(Xtr), enc.transform(Xva), enc.transform(Xte)
    return (
        pd.concat([Xtr.reset_index(drop=True), etr.reset_index(drop=True)], axis=1),
        pd.concat([Xva.reset_index(drop=True), eva.reset_index(drop=True)], axis=1),
        pd.concat([Xte.reset_index(drop=True), ete.reset_index(drop=True)], axis=1),
    )


def train_model(name: str, encode: bool = False, use_original: bool = False, tag: str = "") -> dict:
    tr = D.load_train()
    te = D.load_test()
    y = D.encode_target(tr[D.TARGET])
    Xtr_all = F.add_features(tr)
    Xte_base = F.add_features(te)
    cat_cols = F.CATEGORICAL_FEATURES
    params = _load_params(name)

    if use_original:
        orig = D.load_original()
        Xorig = F.add_features(orig)
        yorig = D.encode_target(orig[D.TARGET])
        print(f"  + original SDSS17 rows: {len(Xorig)}")

    folds = cv.get_folds(y)
    oof = np.zeros((len(y), N_CLASS))
    test_pred = np.zeros((len(Xte_base), N_CLASS))
    fold_baccs = []

    for k, (tr_idx, va_idx) in enumerate(folds):
        t0 = time.time()
        Xtr, Xva = Xtr_all.iloc[tr_idx], Xtr_all.iloc[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]
        Xte = Xte_base

        if use_original:
            Xtr = pd.concat([Xtr, Xorig], axis=0, ignore_index=True)
            ytr = np.concatenate([ytr, yorig])
            for c in cat_cols:  # concat can widen category dtype; re-cast
                Xtr[c] = Xtr[c].astype("category")

        if encode:
            Xtr, Xva, Xte = _add_encoders(Xtr, ytr, Xva, Xte)

        if name == "cat":
            va_proba, model = _fit_cat(_prep_cat(Xtr, cat_cols), ytr,
                                       _prep_cat(Xva, cat_cols), yva, cat_cols, params)
            test_pred += model.predict_proba(_prep_cat(Xte, cat_cols)) / len(folds)
        else:
            va_proba, model = FITTERS[name](Xtr, ytr, Xva, yva, cat_cols, params)
            test_pred += model.predict_proba(Xte) / len(folds)

        oof[va_idx] = va_proba
        bacc = M.balanced_accuracy(yva, va_proba)
        fold_baccs.append(bacc)
        print(f"  [{name}{tag}] fold {k}: bal-acc(raw)={bacc:.5f}  ({time.time()-t0:.0f}s)")

    raw = M.balanced_accuracy(y, oof)
    mult, corrected = M.tune_multipliers(y, oof)
    print(f"[{name}{tag}] OOF bal-acc raw={raw:.5f}  corrected={corrected:.5f}  "
          f"mult(GALAXY,QSO,STAR)={np.round(mult,3).tolist()}")

    ART.mkdir(parents=True, exist_ok=True)
    key = f"{name}{tag}"
    np.save(ART / f"{key}_oof.npy", oof)
    np.save(ART / f"{key}_test.npy", test_pred)
    np.save(ART / "y_true.npy", y)
    np.save(ART / "test_id.npy", te[D.ID_COL].to_numpy())
    result = {
        "model": key, "encode": encode, "use_original": use_original, "tuned": bool(params),
        "oof_bacc_raw": raw, "oof_bacc_corrected": corrected,
        "mult": mult.tolist(), "fold_baccs": fold_baccs,
    }
    json.dump(result, open(ART / f"{key}_metrics.json", "w"), indent=2)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="lgbm", choices=["lgbm", "xgb", "cat", "all"])
    p.add_argument("--encode", action="store_true", help="add fold-safe category encoders")
    p.add_argument("--use-original", action="store_true", help="append SDSS17 rows to train folds")
    p.add_argument("--tag", default="", help="suffix for artifact filenames, e.g. _fe2")
    a = p.parse_args()
    names = ["lgbm", "xgb", "cat"] if a.model == "all" else [a.model]
    for nm in names:
        train_model(nm, encode=a.encode, use_original=a.use_original, tag=a.tag)
