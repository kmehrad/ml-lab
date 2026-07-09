"""Train a GBDT regressor with KFold CV; save OOF + bagged-test prediction vectors.

Each base learner writes a 1-D OOF prediction vector ``(n_train,)`` and a fold-bagged test
vector ``(n_test,)`` to ``experiments/artifacts/``, plus shared ``y.npy`` (float target) and
``test_id.npy`` — so blending and submission are decoupled from training. OOF is scored with
**RMSE** (the competition metric); early stopping uses each library's RMSE.

Usage:
    uv run python -m src.train --model lgbm
    uv run python -m src.train --model lgbm --sample 50000   # smoke test on 50k rows
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from .cv import folds
from .features import add_features, categorical_columns, feature_columns
from .metric import rmse

ART = Path(__file__).resolve().parent.parent / "experiments" / "artifacts"


def build_estimator(model: str, device: str = "cpu", seed: int = 42,
                    depth: int | None = None, trees: int | None = None):
    if model == "lgbm":
        import lightgbm as lgb
        return lgb.LGBMRegressor(
            n_estimators=trees or 3000, learning_rate=0.03, num_leaves=2 ** (depth or 6) - 1,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            reg_lambda=1.0, min_child_samples=100, n_jobs=-1, random_state=seed,
            objective="regression", metric="rmse", verbose=-1,
        )
    if model == "xgb":
        import xgboost as xgb
        return xgb.XGBRegressor(
            n_estimators=trees or 3000, learning_rate=0.03, max_depth=depth or 6,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, min_child_weight=5,
            tree_method="hist", device=device, enable_categorical=True,
            objective="reg:squarederror", eval_metric="rmse",
            early_stopping_rounds=100, n_jobs=-1, random_state=seed,
        )
    if model == "cat":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(
            iterations=trees or 3000, learning_rate=0.03, depth=depth or 6, l2_leaf_reg=3.0,
            loss_function="RMSE", eval_metric="RMSE", early_stopping_rounds=100,
            random_seed=seed, task_type="GPU" if device == "cuda" else "CPU",
            thread_count=-1, verbose=False,
        )
    raise ValueError(f"unknown model {model}")


def _fit_predict(model, est, Xtr, ytr, Xva, yva, Xte, cats):
    """Fit with per-library early stopping; return (oof_va_pred, test_pred_or_None, best_iter)."""
    if model == "lgbm":
        import lightgbm as lgb
        est.fit(Xtr, ytr, eval_set=[(Xva, yva)],
                categorical_feature=cats,
                callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        bi = est.best_iteration_
    elif model == "xgb":
        est.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        bi = est.best_iteration
    elif model == "cat":
        # CatBoost wants categoricals as plain strings, not the pandas `category` dtype.
        Xtr, Xva = Xtr.copy(), Xva.copy()
        for c in cats:
            Xtr[c] = Xtr[c].astype(str)
            Xva[c] = Xva[c].astype(str)
        est.fit(Xtr, ytr, eval_set=(Xva, yva), cat_features=cats, verbose=False)
        bi = est.get_best_iteration()
        if Xte is not None:
            Xte = Xte.copy()
            for c in cats:
                Xte[c] = Xte[c].astype(str)
    else:
        raise ValueError(model)
    va_pred = np.asarray(est.predict(Xva), dtype=float)
    te_pred = np.asarray(est.predict(Xte), dtype=float) if Xte is not None else None
    return va_pred, te_pred, bi


def run_cv(model: str, sample: int | None = None, n_splits: int = 5,
           groups=("base",), tag: str = "", device: str = "cpu",
           depth: int | None = None, trees: int | None = None) -> dict:
    groups = tuple(groups)
    df = add_features(D.load_train(), groups)
    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=42).reset_index(drop=True)
    print(f"train rows={len(df):,}  model={model}  features={list(groups)}"
          + (f"  tag={tag}" if tag else ""))

    feats = feature_columns(groups)
    cats = categorical_columns(groups)

    # Test set (fold-bagged). Skipped for smoke tests.
    test = Xte = test_id = None
    if not sample:
        test = add_features(D.load_test(), groups)
        # Align categorical levels across train/test so category codes match at predict time.
        for c in cats:
            levels = pd.unique(pd.concat([df[c].astype("object"), test[c].astype("object")],
                                         ignore_index=True))
            dtype = pd.CategoricalDtype(categories=levels)
            df[c] = df[c].astype("object").astype(dtype)
            test[c] = test[c].astype("object").astype(dtype)
        test_id = test[D.ID].to_numpy()
        Xte = test[feats]
        test_pred = np.zeros(len(test))

    X = df[feats]
    y = df[D.TARGET].to_numpy(dtype=float)

    oof = np.zeros(len(df))
    fold_scores = []
    t0 = time.time()
    for k, (tr, va) in enumerate(folds(y, n_splits)):
        est = build_estimator(model, device=device, depth=depth, trees=trees)
        va_pred, te_pred, bi = _fit_predict(
            model, est, X.iloc[tr], y[tr], X.iloc[va], y[va], Xte, cats)
        oof[va] = va_pred
        if Xte is not None:
            test_pred += te_pred / n_splits
        score = rmse(y[va], va_pred)
        fold_scores.append(score)
        print(f"  fold {k}: RMSE={score:.5f}  (best_iter={bi})")

    oof_rmse = rmse(y, oof)
    res = {
        "model": model, "features": list(groups), "tag": tag, "oof_rmse": float(oof_rmse),
        "fold_mean": float(np.mean(fold_scores)), "fold_std": float(np.std(fold_scores)),
        "fold_scores": [float(s) for s in fold_scores], "n_rows": int(len(df)),
        "n_features": int(len(feats)), "elapsed_s": round(time.time() - t0, 1),
    }
    print(f"\nOOF RMSE = {oof_rmse:.5f}   folds {res['fold_mean']:.5f} +/- {res['fold_std']:.5f}"
          f"   ({res['elapsed_s']}s)")

    if not sample:
        ART.mkdir(parents=True, exist_ok=True)
        suffix = f"_{tag}" if tag else ""
        np.save(ART / f"{model}{suffix}_oof.npy", oof)
        np.save(ART / f"{model}{suffix}_test.npy", test_pred)
        np.save(ART / "y.npy", y)                       # shared, feature-independent
        np.save(ART / "test_id.npy", test_id)
        (ART / f"{model}{suffix}_metrics.json").write_text(json.dumps(res, indent=2))
        print(f"saved -> {ART}/{model}{suffix}_oof.npy, {model}{suffix}_test.npy, "
              f"{model}{suffix}_metrics.json")
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="lgbm", choices=["lgbm", "xgb", "cat"])
    p.add_argument("--sample", type=int, default=None, help="limit to N rows (smoke test; no test preds)")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--features", nargs="+", default=["base"], help="feature groups: base ratios")
    p.add_argument("--tag", default="", help="artifact filename suffix (keeps experiments separate)")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="xgb/cat compute device")
    p.add_argument("--depth", type=int, default=None, help="tree depth override")
    p.add_argument("--trees", type=int, default=None, help="n_estimators override")
    a = p.parse_args()
    run_cv(a.model, a.sample, a.folds, groups=a.features, tag=a.tag,
           device=a.device, depth=a.depth, trees=a.trees)
