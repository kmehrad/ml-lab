from __future__ import annotations

import argparse, json, platform, subprocess, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .artifacts import save_run, save_shared
from .cv import make_folds
from .data import CATEGORICAL, ID, TARGET, load_data, map_original
from .features import make_features
from .metric import metrics


def _revision():
    try: return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception: return "unknown"


def _sklearn_model(kind, x, seed):
    cat = [c for c in CATEGORICAL if c in x]
    num = [c for c in x if c not in cat]
    prep = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), num),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                              OneHotEncoder(handle_unknown="ignore", sparse_output=False)), cat),
    ])
    if kind == "dummy": estimator = DummyClassifier(strategy="prior")
    elif kind == "logistic": estimator = LogisticRegression(max_iter=500, C=1.0)
    elif kind == "hist": estimator = HistGradientBoostingClassifier(max_iter=300, learning_rate=.07,
                                                                      l2_regularization=1., random_state=seed)
    else: raise ValueError(kind)
    return make_pipeline(prep, estimator)


def _model(kind, x, seed, params=None):
    params = params or {}
    if kind in {"dummy", "logistic", "hist"}: return _sklearn_model(kind, x, seed)
    if kind == "lgbm":
        from lightgbm import LGBMClassifier
        config=dict(n_estimators=1000, learning_rate=.04, num_leaves=31,
                    subsample=.85, colsample_bytree=.85, random_state=seed, verbosity=-1)
        config.update(params); return LGBMClassifier(**config)
    if kind == "xgb":
        from xgboost import XGBClassifier
        config=dict(n_estimators=1000, learning_rate=.04, max_depth=7, subsample=.85,
                    colsample_bytree=.85, tree_method="hist", eval_metric="auc", random_state=seed)
        config.update(params); return XGBClassifier(**config)
    if kind == "cat":
        from catboost import CatBoostClassifier
        config=dict(iterations=1000, learning_rate=.05, depth=7, loss_function="Logloss",
                    eval_metric="AUC", verbose=False, random_seed=seed, allow_writing_files=False)
        config.update(params); return CatBoostClassifier(**config)
    raise ValueError(kind)


def run_cv(train, test, kind="lgbm", feature_group="base", n_splits=5, seed=42,
           original=None, original_weight=1.0, params=None):
    x, xt, y = make_features(train, feature_group), make_features(test, feature_group), train[TARGET].to_numpy()
    folds = make_folds(y, n_splits, seed); oof = np.full(len(train), np.nan); pred = np.zeros(len(test)); scores = []
    for fold in range(n_splits):
        tr, va = folds != fold, folds == fold
        xtr, ytr = x.loc[tr].copy(), y[tr]
        weights = None
        if original is not None:
            xo, yo = make_features(original, feature_group), original[TARGET].to_numpy()
            xtr = pd.concat([xtr, xo], ignore_index=True); ytr = np.r_[ytr, yo]
            weights = np.r_[np.ones(tr.sum()), np.full(len(xo), original_weight)]
        # Native categorical support; normalize missing categories for CatBoost.
        model = _model(kind, xtr, seed + fold, params)
        fit_kw = {}
        if kind in {"lgbm", "xgb"}:
            cats = [c for c in CATEGORICAL if c in xtr]
            if cats:
                enc = ColumnTransformer([("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                      OneHotEncoder(handle_unknown="ignore", sparse_output=False)), cats)], remainder=SimpleImputer(strategy="median"))
                model = make_pipeline(enc, model)
        elif kind == "cat":
            cats = [c for c in CATEGORICAL if c in xtr]
            for frame in (xtr, x.loc[va], xt): frame[cats] = frame[cats].fillna("__MISSING__").astype(str)
            fit_kw["cat_features"] = cats
        if weights is not None:
            fit_kw[("lgbmclassifier__sample_weight" if kind == "lgbm" and hasattr(model, "steps") else
                    "xgbclassifier__sample_weight" if kind == "xgb" and hasattr(model, "steps") else
                    "sample_weight")] = weights
        model.fit(xtr, ytr, **fit_kw)
        oof[va] = model.predict_proba(x.loc[va])[:, 1]
        pred += model.predict_proba(xt)[:, 1] / n_splits
        scores.append(metrics(y[va], oof[va])["roc_auc"])
    return oof, pred, folds, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["dummy", "logistic", "hist", "lgbm", "xgb", "cat"], default="lgbm")
    ap.add_argument("--features", choices=["base", "behavior", "compact", "source_rule", "missing"], default="base")
    ap.add_argument("--folds", type=int, default=5); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample", type=int); ap.add_argument("--tag"); ap.add_argument("--params")
    ap.add_argument("--original"); ap.add_argument("--original-weight", type=float, default=1.)
    args = ap.parse_args(); started = time.time()
    train, test = load_data()
    if args.sample and args.sample < len(train): train = train.sample(args.sample, random_state=args.seed).sort_index()
    original = map_original(pd.read_csv(args.original)) if args.original else None
    params = json.loads(args.params) if args.params else None
    oof, pred, folds, scores = run_cv(train, test, args.model, args.features, args.folds,
                                      args.seed, original, args.original_weight, params)
    result = {**metrics(train[TARGET], oof), "fold_scores": scores, "fold_mean": float(np.mean(scores)),
              "fold_std": float(np.std(scores)), "model": args.model, "feature_group": args.features,
              "seed": args.seed, "n_splits": args.folds, "n_train": len(train),
              "original_rows": 0 if original is None else len(original), "original_weight": args.original_weight,
              "elapsed_seconds": time.time()-started, "revision": _revision(), "python": platform.python_version(),
              "params": params or {}}
    tag = args.tag or f"{args.model}_{args.features}_s{args.seed}"
    path = Path("experiments/artifacts"); save_shared(path, train[TARGET], folds, test[ID])
    np.save(path / f"{tag}_folds.npy", folds)
    save_run(path, tag, oof, pred, result)
    print(json.dumps(result, indent=2))

if __name__ == "__main__": main()
