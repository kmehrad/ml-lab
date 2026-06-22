"""Cross-validated model training and scoring for Predicting Irrigation Need.

The competition metric is **balanced accuracy**, so every model is trained with
balanced class weights and scored on out-of-fold predictions with balanced
accuracy plus per-class recall. The preprocessing pipeline is fit inside each
fold (on the training rows only) so the ``aridity_index`` statistics, the
standard scaler, and the one-hot encoder never see validation data.

Usage
-----
    python -m src.train --model baseline
    python -m src.train --model lgbm --folds 5
    python -m src.train --model all
    python -m src.train --model lgbm --sample 50000   # quick smoke run
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

from .data import (
    CATEGORICAL_FEATURES,
    CLASS_ORDER,
    encode_target,
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
    "lgbm": "tree",
    "xgb": "tree",
    "catboost": "tree",
    "histgb": "tree",
}
ALL_MODELS = ("baseline", "lgbm", "xgb", "catboost", "histgb")


def build_estimator(name: str, seed: int):
    """Instantiate an unfitted estimator. Class balancing is applied at fit time."""
    if name in ("baseline", "logreg"):
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            C=1.0,
            random_state=seed,
        )
    if name == "lgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            objective="multiclass",
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
    if name == "xgb":
        from xgboost import XGBClassifier

        return XGBClassifier(
            objective="multi:softprob",
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            enable_categorical=True,
            random_state=seed,
            n_jobs=-1,
        )
    if name == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            loss_function="MultiClass",
            iterations=500,
            learning_rate=0.05,
            depth=6,
            auto_class_weights="Balanced",
            random_seed=seed,
            thread_count=-1,
            allow_writing_files=False,
            verbose=False,
        )
    if name == "histgb":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=500,
            max_leaf_nodes=63,
            categorical_features=list(CATEGORICAL_FEATURES),
            random_state=seed,
        )
    raise ValueError(f"Unknown model {name!r}; choose from {sorted(MODEL_FAMILY)}")


def _fit_predict_fold(name, estimator, X_tr, y_tr, X_va):
    """Fit one fold with balanced weights and return validation class codes."""
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in X_tr.columns]
    if name in ("baseline", "logreg"):
        # class_weight handles balancing inside LogisticRegression.
        estimator.fit(X_tr, y_tr)
    elif name == "lgbm":
        estimator.fit(X_tr, y_tr, categorical_feature=cat_cols)
    elif name == "xgb":
        weight = compute_sample_weight("balanced", y_tr)
        estimator.fit(X_tr, y_tr, sample_weight=weight)
    elif name == "catboost":
        # CatBoost wants string categoricals; auto_class_weights handles balance.
        X_tr = X_tr.assign(**{c: X_tr[c].astype(str) for c in cat_cols})
        X_va = X_va.assign(**{c: X_va[c].astype(str) for c in cat_cols})
        estimator.fit(X_tr, y_tr, cat_features=cat_cols)
    elif name == "histgb":
        weight = compute_sample_weight("balanced", y_tr)
        estimator.fit(X_tr, y_tr, sample_weight=weight)
    else:  # pragma: no cover - guarded by build_estimator
        raise ValueError(name)
    return estimator.predict(X_va).astype(int).ravel()


def run_cv(name: str, folds: int = 5, seed: int = 42, sample: int | None = None) -> dict:
    """Run stratified K-fold CV for one model and return its metrics."""
    family = MODEL_FAMILY[name]
    train = load_raw("train")
    if sample is not None:
        train = train.sample(sample, random_state=seed).reset_index(drop=True)
    X, y_labels = split_features_target(train)
    y = encode_target(y_labels).to_numpy()

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.full(len(y), -1, dtype=int)
    start = time.perf_counter()
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        pre = build_preprocessor(family).fit(X.iloc[tr_idx], y[tr_idx])
        X_tr = pre.transform(X.iloc[tr_idx])
        X_va = pre.transform(X.iloc[va_idx])
        estimator = build_estimator(name, seed)
        oof[va_idx] = _fit_predict_fold(name, estimator, X_tr, y[tr_idx], X_va)
        print(f"  [{name}] fold {fold}/{folds} done", flush=True)
    elapsed = time.perf_counter() - start

    recalls = recall_score(y, oof, average=None, labels=range(len(CLASS_ORDER)))
    metrics = {
        "model": name,
        "family": family,
        "folds": folds,
        "seed": seed,
        "n_rows": int(len(y)),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, oof)), 5),
        "accuracy": round(float(accuracy_score(y, oof)), 5),
        "recall_per_class": {cls: round(float(r), 5) for cls, r in zip(CLASS_ORDER, recalls)},
        "train_seconds": round(elapsed, 1),
    }
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(ARTIFACTS_DIR / f"{name}_oof.npy", oof)
    _record_results(metrics)
    return metrics


def _record_results(metrics: dict) -> None:
    """Persist metrics to JSON (keyed by model) and a flat CSV."""
    results = {}
    if RESULTS_JSON.exists():
        results = json.loads(RESULTS_JSON.read_text())
    results[metrics["model"]] = metrics
    RESULTS_JSON.write_text(json.dumps(results, indent=2))

    rows = []
    for m in results.values():
        row = {k: v for k, v in m.items() if k != "recall_per_class"}
        row.update({f"recall_{cls}": m["recall_per_class"][cls] for cls in CLASS_ORDER})
        rows.append(row)
    pd.DataFrame(rows).sort_values("balanced_accuracy", ascending=False).to_csv(
        RESULTS_CSV, index=False
    )


def _print_metrics(metrics: dict) -> None:
    pc = " | ".join(f"{c}={metrics['recall_per_class'][c]:.3f}" for c in CLASS_ORDER)
    print(
        f"{metrics['model']:>9}: balanced_acc={metrics['balanced_accuracy']:.5f}  "
        f"acc={metrics['accuracy']:.5f}  recall[{pc}]  ({metrics['train_seconds']}s)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="baseline", help="model name or 'all'")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample", type=int, default=None, help="subsample rows for a quick run")
    args = parser.parse_args()

    models = ALL_MODELS if args.model == "all" else (args.model,)
    for name in models:
        print(f"== {name} ==", flush=True)
        metrics = run_cv(name, folds=args.folds, seed=args.seed, sample=args.sample)
        _print_metrics(metrics)


if __name__ == "__main__":
    main()
