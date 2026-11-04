"""Five-fold binary ROC-AUC training with OOF and bagged test predictions."""
import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import data
from .cv import folds
from .features import feature_columns, numeric_and_categorical, tree_features
from .metric import auc

ART = data.ROOT / "experiments" / "artifacts"


def logreg_pipeline(frame: pd.DataFrame) -> Pipeline:
    nums, cats = numeric_and_categorical(frame)
    prep = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), nums),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cats),
    ])
    return Pipeline([("prepare", prep), ("model", LogisticRegression(max_iter=1000))])


def estimator(name: str, seed: int):
    if name == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=1200, learning_rate=0.03, num_leaves=31,
                              min_child_samples=100, colsample_bytree=0.8,
                              random_state=seed, n_jobs=-1, verbosity=-1)
    if name == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=1200, learning_rate=0.03, max_depth=5,
                             subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                             enable_categorical=True, eval_metric="auc", random_state=seed,
                             n_jobs=-1)
    if name == "cat":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=1200, learning_rate=0.03, depth=6,
                                  loss_function="Logloss", eval_metric="AUC", verbose=False,
                                  random_seed=seed, thread_count=-1)
    raise ValueError(name)


def run(model: str, sample: int | None = None, n_splits: int = 5, seed: int = 42,
        device: str = "cpu", tag: str = "") -> dict:
    train, test, _ = data.load_all()
    if sample:
        train = train.sample(n=min(sample, len(train)), random_state=seed).reset_index(drop=True)
    cols = feature_columns(train)
    x = train[cols]
    xt = test[cols]
    y = data.encode_target(train[data.TARGET]).to_numpy()
    if model != "logreg":
        x, xt, cats = tree_features(x, xt)
    else:
        cats = []
    oof = np.zeros(len(train), dtype=float)
    test_pred = np.zeros(len(test), dtype=float) if not sample else None
    fold_scores = []
    t0 = time.time()
    for fold, (tr, va) in enumerate(folds(y, n_splits, seed), 1):
        fit_x, val_x = x.iloc[tr].copy(), x.iloc[va].copy()
        fit_y, val_y = y[tr], y[va]
        if model == "logreg":
            est = logreg_pipeline(fit_x)
            est.fit(fit_x, fit_y)
        else:
            est = estimator(model, seed + fold)
            if model == "cat":
                for col in cats:
                    fit_x[col] = fit_x[col].astype(object).fillna("__NA__").astype(str)
                    val_x[col] = val_x[col].astype(object).fillna("__NA__").astype(str)
                est.set_params(task_type="GPU" if device == "gpu" else "CPU")
                est.fit(fit_x, fit_y, cat_features=cats, eval_set=(val_x, val_y),
                        early_stopping_rounds=100, verbose=False)
            elif model == "lgbm":
                import lightgbm as lgb
                est.fit(fit_x, fit_y, eval_set=[(val_x, val_y)], eval_metric="auc",
                        categorical_feature=cats,
                        callbacks=[lgb.early_stopping(100, verbose=False)])
            else:
                est.set_params(device="cuda" if device == "gpu" else "cpu",
                               early_stopping_rounds=100)
                est.fit(fit_x, fit_y, eval_set=[(val_x, val_y)], verbose=False)
        p = est.predict_proba(val_x)[:, 1]
        oof[va] = p
        score = auc(val_y, p)
        fold_scores.append(score)
        print(f"fold {fold}: AUC={score:.6f}")
        if test_pred is not None:
            pred_x = xt.copy()
            if model == "cat":
                for col in cats:
                    pred_x[col] = pred_x[col].astype(object).fillna("__NA__").astype(str)
            test_pred += est.predict_proba(pred_x)[:, 1] / n_splits
    metrics = {"model": model, "tag": tag, "rows": len(train), "features": cols,
               "oof_auc": auc(y, oof), "fold_auc": fold_scores, "seconds": round(time.time()-t0, 1),
               "seed": seed, "splits": n_splits, "device": device}
    print(json.dumps({k: v for k, v in metrics.items() if k != "features"}, indent=2))
    if test_pred is not None:
        ART.mkdir(parents=True, exist_ok=True)
        name = model + (f"_{tag}" if tag else "")
        np.save(ART / f"{name}_oof.npy", oof)
        np.save(ART / f"{name}_test.npy", test_pred)
        np.save(ART / "y.npy", y)
        np.save(ART / "test_id.npy", test[data.ID].to_numpy())
        (ART / f"{name}_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["logreg", "lgbm", "xgb", "cat"], default="logreg")
    parser.add_argument("--sample", type=int)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    run(args.model, args.sample, args.splits, device=args.device, tag=args.tag)
