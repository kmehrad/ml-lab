"""Cross-validated training, scoring, and bagged test prediction.

The competition metric is **ROC-AUC**, a ranking metric on the positive-class
probability, so every model predicts probabilities and is scored on out-of-fold
predictions with ``roc_auc_score``. AUC is threshold-independent and largely
insensitive to class priors, so models are trained *without* class re-weighting.

The configs here are the regularised winners from the experiment log
(``experiments/README.md``). Two LightGBM seeds plus XGBoost / CatBoost /
HistGB / LogisticRegression give a diverse pool for blending.

Original-data augmentation
--------------------------
``--use-original`` concatenates the source dataset (``data/raw/original.csv``,
see :func:`src.data.load_original`) into **each training fold only** — never the
validation fold — so OOF stays honest. This was the single biggest leaderboard
lever (see the experiment log). Augmented runs are saved under an ``_aug`` suffix
so base and augmented predictions can be blended together.

Each fold also predicts the test set; the fold predictions are averaged
(bagging) into ``{key}_test.npy`` so submissions need no separate full-data refit.

Usage
-----
    python -m src.train --model lgbm --folds 5
    python -m src.train --model all
    python -m src.train --model all --use-original     # augmented pool
    python -m src.train --model lgbm --sample 50000     # quick smoke run
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .data import (
    RAW_FEATURES,
    load_original,
    load_raw,
    split_features_target,
)
from .preprocessing import build_preprocessor

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "experiments" / "artifacts"
RESULTS_JSON = ARTIFACTS_DIR / "model_results.json"
RESULTS_CSV = ARTIFACTS_DIR / "model_results.csv"

# family decides which preprocessing pipeline feeds the estimator.
MODEL_FAMILY = {
    "baseline": "linear",
    "logreg": "linear",
    "mlp": "linear",
    "lgbm": "tree",
    "lgbm2": "tree",
    "xgb": "tree",
    "catboost": "tree",
    "histgb": "tree",
}
ALL_MODELS = ("lgbm", "lgbm2", "xgb", "catboost", "histgb", "logreg")


def build_estimator(name: str, seed: int):
    """Instantiate an unfitted, regularised binary classifier."""
    if name in ("baseline", "logreg"):
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(max_iter=3000, C=0.5, random_state=seed)
    if name == "mlp":
        from sklearn.neural_network import MLPClassifier

        return MLPClassifier(
            hidden_layer_sizes=(96, 32), alpha=1e-3, batch_size=4096,
            learning_rate_init=2e-3, max_iter=60, early_stopping=True,
            n_iter_no_change=6, random_state=seed,
        )
    if name == "lgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            objective="binary", n_estimators=1500, learning_rate=0.02,
            num_leaves=31, max_depth=5, min_child_samples=400, subsample=0.7,
            subsample_freq=1, colsample_bytree=0.6, reg_alpha=1.0, reg_lambda=5.0,
            random_state=seed, n_jobs=-1, verbose=-1,
        )
    if name == "lgbm2":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            objective="binary", n_estimators=1200, learning_rate=0.025,
            num_leaves=24, max_depth=6, min_child_samples=600, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.7, reg_alpha=2.0, reg_lambda=2.0,
            random_state=seed + 7, n_jobs=-1, verbose=-1,
        )
    if name == "xgb":
        from xgboost import XGBClassifier

        return XGBClassifier(
            objective="binary:logistic", eval_metric="auc", n_estimators=1200,
            learning_rate=0.025, max_depth=5, min_child_weight=20, subsample=0.8,
            colsample_bytree=0.7, reg_lambda=5.0, reg_alpha=1.0, gamma=0.1,
            tree_method="hist", random_state=seed, n_jobs=-1,
        )
    if name == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            loss_function="Logloss", eval_metric="AUC", iterations=1500,
            learning_rate=0.025, depth=6, l2_leaf_reg=8.0, random_seed=seed,
            thread_count=-1, allow_writing_files=False, verbose=False,
        )
    if name == "histgb":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            learning_rate=0.03, max_iter=800, max_leaf_nodes=31,
            min_samples_leaf=200, l2_regularization=5.0, random_state=seed,
        )
    raise ValueError(f"Unknown model {name!r}; choose from {sorted(MODEL_FAMILY)}")


def artifact_key(name: str, use_original: bool) -> str:
    """Stable key for saved predictions (``_aug`` suffix when augmented)."""
    return f"{name}_aug" if use_original else name


def run_cv(
    name: str,
    folds: int = 5,
    seed: int = 42,
    sample: int | None = None,
    use_original: bool = False,
) -> dict:
    """5-fold CV producing OOF + bagged test probabilities; saves and scores them.

    The preprocessor is fit on the training fold only. When ``use_original`` is
    set, the source dataset is concatenated into the training fold (never the
    validation fold), so the OOF AUC is computed on competition rows alone.
    """
    family = MODEL_FAMILY[name]
    train = load_raw("train")
    if sample is not None:
        train = train.sample(sample, random_state=seed).reset_index(drop=True)
    X, y = split_features_target(train)
    y = y.to_numpy()
    X_test = load_raw("test")[list(RAW_FEATURES)]

    Xo = yo = None
    if use_original:
        orig = load_original()
        Xo, yo = split_features_target(orig)
        yo = yo.to_numpy()

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.full(len(y), np.nan, dtype="float64")
    test_pred = np.zeros(len(X_test), dtype="float64")
    fold_aucs = []
    start = time.perf_counter()
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        X_fit, y_fit = X.iloc[tr_idx], y[tr_idx]
        if use_original:
            X_fit = pd.concat([X_fit, Xo], ignore_index=True)
            y_fit = np.r_[y_fit, yo]
        pre = build_preprocessor(family).fit(X_fit, y_fit)
        est = build_estimator(name, seed)
        est.fit(pre.transform(X_fit), y_fit)
        oof[va_idx] = est.predict_proba(pre.transform(X.iloc[va_idx]))[:, 1]
        test_pred += est.predict_proba(pre.transform(X_test))[:, 1] / folds
        fold_auc = roc_auc_score(y[va_idx], oof[va_idx])
        fold_aucs.append(fold_auc)
        print(f"  [{name}{'+orig' if use_original else ''}] fold {fold}/{folds} auc={fold_auc:.5f}", flush=True)
    elapsed = time.perf_counter() - start

    key = artifact_key(name, use_original)
    metrics = {
        "model": key,
        "base_model": name,
        "family": family,
        "folds": folds,
        "seed": seed,
        "n_rows": int(len(y)),
        "use_original": use_original,
        "auc": round(float(roc_auc_score(y, oof)), 5),
        "auc_fold_mean": round(float(np.mean(fold_aucs)), 5),
        "auc_fold_std": round(float(np.std(fold_aucs)), 5),
        "avg_precision": round(float(average_precision_score(y, oof)), 5),
        "train_seconds": round(elapsed, 1),
    }
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(ARTIFACTS_DIR / f"{key}_oof.npy", oof)
    np.save(ARTIFACTS_DIR / f"{key}_test.npy", test_pred)
    if sample is None and not use_original:
        np.save(ARTIFACTS_DIR / "y_true.npy", y)
    _record_results(metrics)
    return metrics


def _record_results(metrics: dict) -> None:
    """Persist metrics to JSON (keyed by artifact key) and a flat CSV by AUC."""
    results = {}
    if RESULTS_JSON.exists():
        results = json.loads(RESULTS_JSON.read_text())
    results[metrics["model"]] = metrics
    RESULTS_JSON.write_text(json.dumps(results, indent=2))
    pd.DataFrame(list(results.values())).sort_values("auc", ascending=False).to_csv(
        RESULTS_CSV, index=False
    )


def _print_metrics(metrics: dict) -> None:
    print(
        f"{metrics['model']:>10}: auc={metrics['auc']:.5f}  "
        f"(folds {metrics['auc_fold_mean']:.5f}+/-{metrics['auc_fold_std']:.5f})  "
        f"ap={metrics['avg_precision']:.5f}  ({metrics['train_seconds']}s)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="lgbm", help="model name or 'all'")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample", type=int, default=None, help="subsample rows for a quick run")
    parser.add_argument("--use-original", action="store_true", help="augment each training fold with the source dataset")
    args = parser.parse_args()

    models = ALL_MODELS if args.model == "all" else (args.model,)
    for name in models:
        print(f"== {name}{'+orig' if args.use_original else ''} ==", flush=True)
        metrics = run_cv(
            name, folds=args.folds, seed=args.seed, sample=args.sample,
            use_original=args.use_original,
        )
        _print_metrics(metrics)


if __name__ == "__main__":
    main()
