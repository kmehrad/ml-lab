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
    raise ValueError(f"unknown model {model}")


def run_cv(model: str, sample: int | None = None, n_splits: int = 5) -> dict:
    df = D.load_train()
    if sample:
        keep = df[D.GROUP].drop_duplicates().head(sample)
        df = df[df[D.GROUP].isin(keep)].reset_index(drop=True)
    print(f"train rows={len(df):,} buildings={df[D.GROUP].nunique()}")

    df = add_features(df)
    feats = feature_columns()
    cats = [c for c in D.CATEGORICAL if c in feats]
    X, y, groups = df[feats], df[D.TARGET].to_numpy(), df[D.GROUP].to_numpy()

    oof = np.zeros(len(df))
    fold_aucs = []
    t0 = time.time()
    for k, (tr, va) in enumerate(folds(groups, n_splits)):
        est = build_estimator(model)
        if model == "lgbm":
            import lightgbm as lgb
            est.fit(
                X.iloc[tr], y[tr],
                eval_set=[(X.iloc[va], y[va])], eval_metric="auc",
                categorical_feature=cats,
                callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
            )
            oof[va] = est.predict_proba(X.iloc[va])[:, 1]
        auc = roc_auc_score(y[va], oof[va])
        fold_aucs.append(auc)
        print(f"  fold {k}: AUC={auc:.5f}  (best_iter={getattr(est,'best_iteration_', '-')})")

    oof_auc = roc_auc_score(y, oof)
    res = {
        "model": model, "oof_auc": float(oof_auc),
        "fold_mean": float(np.mean(fold_aucs)), "fold_std": float(np.std(fold_aucs)),
        "fold_aucs": [float(a) for a in fold_aucs], "n_rows": int(len(df)),
        "n_features": len(feats), "elapsed_s": round(time.time() - t0, 1),
    }
    print(f"\nOOF AUC = {oof_auc:.5f}   folds {res['fold_mean']:.5f} +/- {res['fold_std']:.5f}"
          f"   ({res['elapsed_s']}s)")

    if not sample:
        ART.mkdir(parents=True, exist_ok=True)
        np.save(ART / f"{model}_oof.npy", oof)
        np.save(ART / "y.npy", y)
        (ART / f"{model}_metrics.json").write_text(json.dumps(res, indent=2))
        print(f"saved -> {ART}/{model}_oof.npy, {model}_metrics.json")
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="lgbm")
    p.add_argument("--sample", type=int, default=None, help="limit to N buildings (smoke test)")
    p.add_argument("--folds", type=int, default=5)
    a = p.parse_args()
    run_cv(a.model, a.sample, a.folds)
