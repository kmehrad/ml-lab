"""Train a model with GroupKFold(building_id) CV; save OOF + bagged test predictions.

Usage:
    uv run python -m src.train --model lgbm
    uv run python -m src.train --model lgbm --sample 30   # smoke test on 30 buildings
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import data as D
from .cv import folds
from .features import add_features, feature_columns
from .target_encoding import TargetEncoder, te_columns

ART = Path(__file__).resolve().parent.parent / "experiments" / "artifacts"


def build_estimator(model: str):
    if model == "lgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(
            n_estimators=2000, learning_rate=0.03, num_leaves=127,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            reg_lambda=1.0, min_child_samples=100, n_jobs=-1, random_state=42,
            objective="binary", verbose=-1,
        )
    if model == "xgb":
        import xgboost as xgb
        return xgb.XGBClassifier(
            n_estimators=2000, learning_rate=0.03, max_depth=8,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, min_child_weight=5,
            tree_method="hist", enable_categorical=True, eval_metric="auc",
            early_stopping_rounds=100, n_jobs=-1, random_state=42,
        )
    if model == "cat":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            iterations=2000, learning_rate=0.03, depth=8, l2_leaf_reg=3.0,
            eval_metric="AUC", early_stopping_rounds=100, random_seed=42,
            thread_count=-1, verbose=False,
        )
    if model == "histgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            max_iter=2000, learning_rate=0.03, max_leaf_nodes=127, l2_regularization=1.0,
            min_samples_leaf=100, categorical_features="from_dtype",
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=50, random_state=42,
        )
    raise ValueError(f"unknown model {model}")


def _fit_predict(model, est, Xtr, ytr, Xva, yva, Xte_k, cats):
    """Fit with model-specific early stopping; return (oof_va, test_pred_or_None, best_iter)."""
    if model == "lgbm":
        import lightgbm as lgb
        est.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="auc",
                categorical_feature=cats,
                callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        bi = est.best_iteration_
    elif model == "xgb":
        est.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        bi = est.best_iteration
    elif model == "cat":
        est.fit(Xtr, ytr, eval_set=(Xva, yva), cat_features=cats, verbose=False)
        bi = est.get_best_iteration()
    elif model == "histgb":
        est.fit(Xtr, ytr)
        bi = est.n_iter_
    else:
        raise ValueError(model)
    va_pred = est.predict_proba(Xva)[:, 1]
    te_pred = est.predict_proba(Xte_k)[:, 1] if Xte_k is not None else None
    return va_pred, te_pred, bi


def run_cv(model: str, sample: int | None = None, n_splits: int = 5,
           target_encode: bool = False) -> dict:
    df = D.load_train()
    if sample:
        keep = df[D.GROUP].drop_duplicates().head(sample)
        df = df[df[D.GROUP].isin(keep)].reset_index(drop=True)
    print(f"train rows={len(df):,} buildings={df[D.GROUP].nunique()}  target_encode={target_encode}")

    df = add_features(df)
    feats = feature_columns()
    cats = [c for c in D.CATEGORICAL if c in feats]

    # Test set (fold-bagged predictions). Skipped for smoke tests.
    test = Xte = row_id = None
    if not sample:
        test = add_features(D.load_test())
        # Align categorical levels across train/test so category codes match
        # (XGBoost/LightGBM/HistGB compare category dtypes during predict).
        for c in cats:
            levels = pd.unique(pd.concat([df[c].astype("object"), test[c].astype("object")],
                                         ignore_index=True))
            dtype = pd.CategoricalDtype(categories=levels)
            df[c] = df[c].astype("object").astype(dtype)
            test[c] = test[c].astype("object").astype(dtype)
        row_id = test["row_id"].to_numpy()
        Xte = test[feats]
        test_pred = np.zeros(len(test))

    X, y, groups = df[feats], df[D.TARGET].to_numpy(), df[D.GROUP].to_numpy()

    oof = np.zeros(len(df))
    fold_aucs = []
    t0 = time.time()
    for k, (tr, va) in enumerate(folds(groups, n_splits)):
        # Per-fold target encoding: fit on training rows only, transform valid + test.
        Xtr, Xva, Xte_k = X.iloc[tr], X.iloc[va], Xte
        if target_encode:
            te = TargetEncoder().fit(df.iloc[tr], y[tr])
            Xtr = pd.concat([Xtr, te.transform(df.iloc[tr])], axis=1)
            Xva = pd.concat([Xva, te.transform(df.iloc[va])], axis=1)
            if Xte is not None:
                Xte_k = pd.concat([Xte, te.transform(test)], axis=1)

        est = build_estimator(model)
        va_pred, te_pred, bi = _fit_predict(model, est, Xtr, y[tr], Xva, y[va], Xte_k, cats)
        oof[va] = va_pred
        if Xte is not None:
            test_pred += te_pred / n_splits
        auc = roc_auc_score(y[va], oof[va])
        fold_aucs.append(auc)
        print(f"  fold {k}: AUC={auc:.5f}  (best_iter={bi})")

    oof_auc = roc_auc_score(y, oof)
    res = {
        "model": model, "oof_auc": float(oof_auc),
        "fold_mean": float(np.mean(fold_aucs)), "fold_std": float(np.std(fold_aucs)),
        "fold_aucs": [float(a) for a in fold_aucs], "n_rows": int(len(df)),
        "n_features": len(feats) + (len(te_columns()) if target_encode else 0),
        "target_encode": target_encode, "elapsed_s": round(time.time() - t0, 1),
    }
    print(f"\nOOF AUC = {oof_auc:.5f}   folds {res['fold_mean']:.5f} +/- {res['fold_std']:.5f}"
          f"   ({res['elapsed_s']}s)")

    if not sample:
        ART.mkdir(parents=True, exist_ok=True)
        np.save(ART / f"{model}_oof.npy", oof)
        np.save(ART / "y.npy", y)
        np.save(ART / f"{model}_test.npy", test_pred)
        np.save(ART / "test_row_id.npy", row_id)
        (ART / f"{model}_metrics.json").write_text(json.dumps(res, indent=2))
        print(f"saved -> {ART}/{model}_oof.npy, {model}_test.npy, {model}_metrics.json")
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="lgbm")
    p.add_argument("--sample", type=int, default=None, help="limit to N buildings (smoke test)")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--target-encode", action="store_true", help="enable in-fold target encoding (hurt OOF; off by default)")
    a = p.parse_args()
    run_cv(a.model, a.sample, a.folds, target_encode=a.target_encode)
