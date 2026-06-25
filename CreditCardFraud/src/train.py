"""Cross-validated training, scoring, and held-out test prediction.

The competition has no leaderboard, so :func:`src.data.make_holdout_split` fixes a
stratified **test set** once and everything below operates on the *remainder*:
``run_cv`` does 5-fold ``StratifiedKFold`` on the remainder, scores out-of-fold
(OOF) predictions, and saves both the OOF array and the fold-averaged (bagged)
predictions on the fixed test set. ``src.evaluate`` later reads those bagged test
predictions to produce the final report.

The primary metric is **AUPRC / average precision** (``average_precision_score``),
which is the meaningful summary under 0.17% positives; ROC-AUC is recorded too but
is optimistic here. Unlike a balanced AUC task, **imbalance handling matters**, so
each model can be trained under one of several strategies (``--imbalance``):

* ``none``        — train on the raw distribution.
* ``classweight`` — cost-reweighting (``class_weight="balanced"`` /
  ``scale_pos_weight = n_neg / n_pos``); no data is added or removed.
* ``undersample`` — random-undersample the majority in each *training* fold only.
* ``smote``       — SMOTE-oversample the minority in each *training* fold only.

Resampling/weighting is applied **inside the training fold only**, never the
validation fold, so OOF AUPRC stays honest. The strategy is encoded into the
artifact key (e.g. ``lgbm_cw``) so strategies are directly comparable.

Usage
-----
    python -m src.train --model lgbm --sample 50000          # quick smoke run
    python -m src.train --model all                          # all models, no reweighting
    python -m src.train --model lgbm --imbalance classweight
    python -m src.train --model all --imbalance smote
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
    load_raw,
    make_holdout_split,
    split_features_target,
)
from .preprocessing import build_preprocessor

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "experiments" / "artifacts"
RESULTS_JSON = ARTIFACTS_DIR / "model_results.json"
RESULTS_CSV = ARTIFACTS_DIR / "model_results.csv"

# family decides which preprocessing pipeline feeds the estimator.
MODEL_FAMILY = {
    "logreg": "linear",
    "lgbm": "tree",
    "xgb": "tree",
    "catboost": "tree",
    "histgb": "tree",
}
ALL_MODELS = ("logreg", "lgbm", "xgb", "catboost", "histgb")

IMBALANCE_STRATEGIES = ("none", "classweight", "undersample", "smote")
# Short suffixes keep artifact keys readable (lgbm_cw, lgbm_us, lgbm_smote).
_IMBALANCE_SUFFIX = {
    "none": "",
    "classweight": "_cw",
    "undersample": "_us",
    "smote": "_smote",
}


def build_estimator(name: str, seed: int, imbalance: str, pos_weight: float = 1.0):
    """Instantiate an unfitted classifier, applying cost-reweighting when asked.

    Only ``classweight`` touches the estimator (via ``class_weight`` /
    ``scale_pos_weight``); ``undersample`` / ``smote`` resample the data instead
    and leave the estimator at its base configuration. ``pos_weight`` is
    ``n_neg / n_pos`` for XGBoost's ``scale_pos_weight``.
    """
    reweight = imbalance == "classweight"
    if name == "logreg":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(
            max_iter=3000, C=1.0, random_state=seed,
            class_weight="balanced" if reweight else None,
        )
    if name == "lgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            objective="binary", n_estimators=1200, learning_rate=0.02,
            num_leaves=31, max_depth=6, min_child_samples=50, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=5.0,
            random_state=seed, n_jobs=-1, verbose=-1,
            class_weight="balanced" if reweight else None,
        )
    if name == "xgb":
        from xgboost import XGBClassifier

        return XGBClassifier(
            objective="binary:logistic", eval_metric="aucpr", n_estimators=1000,
            learning_rate=0.025, max_depth=6, min_child_weight=1, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=5.0, reg_alpha=1.0, gamma=0.0,
            tree_method="hist", random_state=seed, n_jobs=-1,
            scale_pos_weight=pos_weight if reweight else 1.0,
        )
    if name == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            loss_function="Logloss", eval_metric="PRAUC", iterations=1200,
            learning_rate=0.03, depth=6, l2_leaf_reg=6.0, random_seed=seed,
            thread_count=-1, allow_writing_files=False, verbose=False,
            auto_class_weights="Balanced" if reweight else None,
        )
    if name == "histgb":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=600, max_leaf_nodes=31,
            min_samples_leaf=50, l2_regularization=1.0, random_state=seed,
            class_weight="balanced" if reweight else None,
        )
    raise ValueError(f"Unknown model {name!r}; choose from {sorted(MODEL_FAMILY)}")


def artifact_key(name: str, imbalance: str) -> str:
    """Stable key for saved predictions, e.g. ``lgbm`` or ``lgbm_cw``."""
    return f"{name}{_IMBALANCE_SUFFIX[imbalance]}"


def _resample(X_fit: pd.DataFrame, y_fit: np.ndarray, imbalance: str, seed: int):
    """Apply training-fold-only resampling for ``undersample`` / ``smote``."""
    if imbalance == "undersample":
        from imblearn.under_sampling import RandomUnderSampler

        return RandomUnderSampler(random_state=seed).fit_resample(X_fit, y_fit)
    if imbalance == "smote":
        from imblearn.over_sampling import SMOTE

        return SMOTE(random_state=seed).fit_resample(X_fit, y_fit)
    return X_fit, y_fit


def run_cv(
    name: str,
    folds: int = 5,
    seed: int = 42,
    sample: int | None = None,
    imbalance: str = "none",
) -> dict:
    """5-fold CV on the train remainder; saves OOF + bagged held-out test preds.

    The preprocessor (and any resampling) is fit on the training fold only. OOF
    AUPRC is computed on the held-out validation folds of the *remainder*; the
    fixed test set is predicted by every fold and the predictions are averaged.
    """
    if imbalance not in IMBALANCE_STRATEGIES:
        raise ValueError(f"imbalance must be one of {IMBALANCE_STRATEGIES}")
    family = MODEL_FAMILY[name]

    frame = load_raw()
    train_idx, test_idx = make_holdout_split(seed=seed)
    train = frame.iloc[train_idx].reset_index(drop=True)
    test = frame.iloc[test_idx].reset_index(drop=True)
    if sample is not None:
        # Keep all positives so a subsample still has signal.
        pos = train[train["Class"] == 1]
        neg = train[train["Class"] == 0].sample(
            max(sample - len(pos), 1), random_state=seed
        )
        train = pd.concat([pos, neg]).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    X, y = split_features_target(train)
    y = y.to_numpy()
    X_test, y_test = split_features_target(test)
    y_test = y_test.to_numpy()
    pos_weight = float((y == 0).sum() / max((y == 1).sum(), 1))

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.full(len(y), np.nan, dtype="float64")
    test_pred = np.zeros(len(X_test), dtype="float64")
    fold_aps = []
    start = time.perf_counter()
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        X_fit, y_fit = X.iloc[tr_idx], y[tr_idx]
        pre = build_preprocessor(family).fit(X_fit, y_fit)
        X_fit_t = pre.transform(X_fit)
        X_fit_t, y_fit_r = _resample(X_fit_t, y_fit, imbalance, seed)
        est = build_estimator(name, seed, imbalance, pos_weight)
        est.fit(X_fit_t, y_fit_r)
        oof[va_idx] = est.predict_proba(pre.transform(X.iloc[va_idx]))[:, 1]
        test_pred += est.predict_proba(pre.transform(X_test))[:, 1] / folds
        fold_ap = average_precision_score(y[va_idx], oof[va_idx])
        fold_aps.append(fold_ap)
        print(f"  [{artifact_key(name, imbalance)}] fold {fold}/{folds} ap={fold_ap:.5f}", flush=True)
    elapsed = time.perf_counter() - start

    key = artifact_key(name, imbalance)
    metrics = {
        "model": key,
        "base_model": name,
        "family": family,
        "imbalance": imbalance,
        "folds": folds,
        "seed": seed,
        "n_rows": int(len(y)),
        "oof_ap": round(float(average_precision_score(y, oof)), 5),
        "oof_ap_fold_mean": round(float(np.mean(fold_aps)), 5),
        "oof_ap_fold_std": round(float(np.std(fold_aps)), 5),
        "oof_roc_auc": round(float(roc_auc_score(y, oof)), 5),
        "test_ap": round(float(average_precision_score(y_test, test_pred)), 5),
        "test_roc_auc": round(float(roc_auc_score(y_test, test_pred)), 5),
        "train_seconds": round(elapsed, 1),
    }
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(ARTIFACTS_DIR / f"{key}_oof.npy", oof)
    np.save(ARTIFACTS_DIR / f"{key}_test.npy", test_pred)
    if sample is None and imbalance == "none":
        np.save(ARTIFACTS_DIR / "y_oof.npy", y)
        np.save(ARTIFACTS_DIR / "y_test.npy", y_test)
    _record_results(metrics)
    return metrics


def _record_results(metrics: dict) -> None:
    """Persist metrics to JSON (keyed by artifact key) and a flat CSV by OOF AP."""
    results = {}
    if RESULTS_JSON.exists():
        results = json.loads(RESULTS_JSON.read_text())
    results[metrics["model"]] = metrics
    RESULTS_JSON.write_text(json.dumps(results, indent=2))
    pd.DataFrame(list(results.values())).sort_values("oof_ap", ascending=False).to_csv(
        RESULTS_CSV, index=False
    )


def _print_metrics(metrics: dict) -> None:
    print(
        f"{metrics['model']:>12}: oof_ap={metrics['oof_ap']:.5f}  "
        f"(folds {metrics['oof_ap_fold_mean']:.5f}+/-{metrics['oof_ap_fold_std']:.5f})  "
        f"oof_auc={metrics['oof_roc_auc']:.5f}  "
        f"test_ap={metrics['test_ap']:.5f}  ({metrics['train_seconds']}s)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="lgbm", help="model name or 'all'")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample", type=int, default=None, help="subsample majority rows for a quick run")
    parser.add_argument(
        "--imbalance", default="none", choices=IMBALANCE_STRATEGIES,
        help="imbalance handling strategy",
    )
    args = parser.parse_args()

    models = ALL_MODELS if args.model == "all" else (args.model,)
    for name in models:
        print(f"== {artifact_key(name, args.imbalance)} ==", flush=True)
        metrics = run_cv(
            name, folds=args.folds, seed=args.seed, sample=args.sample,
            imbalance=args.imbalance,
        )
        _print_metrics(metrics)


if __name__ == "__main__":
    main()
