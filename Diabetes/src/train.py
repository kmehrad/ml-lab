"""Cross-validated model training and scoring for the Diabetes challenge.

The competition metric is **ROC-AUC**, a ranking metric on the positive-class
probability, so every model predicts probabilities and is scored on out-of-fold
predictions with ``roc_auc_score``. AUC is threshold-independent and largely
insensitive to class priors, so models are trained *without* class re-weighting
(unlike a balanced-accuracy task) to keep probabilities well calibrated.

The preprocessing pipeline is fit inside each fold (on the training rows only)
so the scaler never sees validation data. Out-of-fold probabilities are saved to
``experiments/artifacts/{model}_oof.npy`` for the blender (see ``src.blend``).

Usage
-----
    python -m src.train --model baseline
    python -m src.train --model lgbm --folds 5
    python -m src.train --model all
    python -m src.train --model lgbm --sample 50000   # quick smoke run
    python -m src.train --model all --use-original     # augment with CDC data
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

from .data import load_original, load_raw, split_features_target
from .preprocessing import build_preprocessor

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "experiments" / "artifacts"
RESULTS_JSON = ARTIFACTS_DIR / "model_results.json"
RESULTS_CSV = ARTIFACTS_DIR / "model_results.csv"

# family decides which preprocessing pipeline feeds the estimator.
MODEL_FAMILY = {
    "baseline": "linear",
    "logreg": "linear",
    "lgbm": "tree",
    "xgb": "tree",
    "catboost": "tree",
    "histgb": "tree",
}
ALL_MODELS = ("baseline", "lgbm", "xgb", "catboost", "histgb")


def build_estimator(name: str, seed: int):
    """Instantiate an unfitted binary classifier with sensible defaults."""
    if name in ("baseline", "logreg"):
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
    if name == "lgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            objective="binary",
            n_estimators=1000,
            learning_rate=0.03,
            num_leaves=63,
            min_child_samples=80,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
    if name == "xgb":
        from xgboost import XGBClassifier

        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=6,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
        )
    if name == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            iterations=1000,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=seed,
            thread_count=-1,
            allow_writing_files=False,
            verbose=False,
        )
    if name == "histgb":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=600,
            max_leaf_nodes=63,
            l2_regularization=1.0,
            random_state=seed,
        )
    raise ValueError(f"Unknown model {name!r}; choose from {sorted(MODEL_FAMILY)}")


def fit_predict_proba(name, estimator, X_tr, y_tr, X_va) -> np.ndarray:
    """Fit ``estimator`` and return positive-class probabilities for ``X_va``.

    Shared by cross-validation (fit on a training fold, predict the validation
    fold) and by submission generation (fit on all train, predict test).
    """
    estimator.fit(X_tr, y_tr)
    return estimator.predict_proba(X_va)[:, 1]


def _load_training_data(seed: int, sample: int | None, use_original: bool):
    """Return ``(X, y)`` for training, optionally augmented and subsampled."""
    from .data import RAW_FEATURES, TARGET_COLUMN

    train = load_raw("train")
    if use_original:
        cols = list(RAW_FEATURES) + [TARGET_COLUMN]
        original = load_original()
        train = pd.concat([train[cols], original[cols]], ignore_index=True)
    if sample is not None:
        train = train.sample(sample, random_state=seed).reset_index(drop=True)
    X, y = split_features_target(train)
    return X, y.to_numpy()


def run_cv(
    name: str,
    folds: int = 5,
    seed: int = 42,
    sample: int | None = None,
    use_original: bool = False,
) -> dict:
    """Run stratified K-fold CV for one model and return its metrics."""
    family = MODEL_FAMILY[name]
    X, y = _load_training_data(seed, sample, use_original)

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.full(len(y), np.nan, dtype="float64")
    fold_aucs = []
    start = time.perf_counter()
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        pre = build_preprocessor(family).fit(X.iloc[tr_idx], y[tr_idx])
        X_tr = pre.transform(X.iloc[tr_idx])
        X_va = pre.transform(X.iloc[va_idx])
        estimator = build_estimator(name, seed)
        oof[va_idx] = fit_predict_proba(name, estimator, X_tr, y[tr_idx], X_va)
        fold_auc = roc_auc_score(y[va_idx], oof[va_idx])
        fold_aucs.append(fold_auc)
        print(f"  [{name}] fold {fold}/{folds} auc={fold_auc:.5f}", flush=True)
    elapsed = time.perf_counter() - start

    metrics = {
        "model": name,
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
    np.save(ARTIFACTS_DIR / f"{name}_oof.npy", oof)
    if not use_original:  # keep the canonical label vector aligned to OOF arrays
        np.save(ARTIFACTS_DIR / "y_true.npy", y)
    _record_results(metrics)
    return metrics


def _record_results(metrics: dict) -> None:
    """Persist metrics to JSON (keyed by model) and a flat CSV sorted by AUC."""
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
        f"{metrics['model']:>9}: auc={metrics['auc']:.5f}  "
        f"(folds {metrics['auc_fold_mean']:.5f}+/-{metrics['auc_fold_std']:.5f})  "
        f"ap={metrics['avg_precision']:.5f}  ({metrics['train_seconds']}s)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="baseline", help="model name or 'all'")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample", type=int, default=None, help="subsample rows for a quick run")
    parser.add_argument("--use-original", action="store_true", help="augment with CDC original data")
    args = parser.parse_args()

    models = ALL_MODELS if args.model == "all" else (args.model,)
    for name in models:
        print(f"== {name} ==", flush=True)
        metrics = run_cv(
            name, folds=args.folds, seed=args.seed, sample=args.sample,
            use_original=args.use_original,
        )
        _print_metrics(metrics)


if __name__ == "__main__":
    main()
