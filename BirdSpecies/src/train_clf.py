"""Classical per-class baseline on aggregated-MFCC features → OOF/test probs.

For each of the 87 classes we fit a one-vs-rest probabilistic classifier
(LogReg / RandomForest / LightGBM) with the shared 5-fold multilabel-stratified
split, producing out-of-fold predictions on the 687 train clips and fold-averaged
predictions on the 1000 test clips.

Scoring is the competition metric: **AUC pooled over all `(file × class)` pairs**
(``roc_auc_score(Y.ravel(), P.ravel())``). Because the metric pools across
classes, we also report a **rank-normalized** variant (per-class scores mapped to
uniform ranks before pooling) to test cross-class calibration.

Artifacts (aligned via the shared folds, for later blending):
    experiments/artifacts/{model}_oof.npy    (687, 87)
    experiments/artifacts/{model}_test.npy   (1000, 87)
    experiments/artifacts/{model}_metrics.json
    experiments/artifacts/y.npy              (687, 87)  shared target, written once

Usage:
    uv run python -m src.train_clf --model lgbm
    uv run python -m src.train_clf --model logreg --sample   # quick smoke run
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import date

import numpy as np
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import cv, data, features

ART = data.ROOT / "experiments" / "artifacts"
RUN_LOG = data.ROOT / "experiments" / "README.md"


# ── model registry ───────────────────────────────────────────────────
def make_model(name: str, seed: int):
    """One-vs-rest binary classifier for a single class. All use class weighting
    to cope with the ≤20 positives/class."""
    if name == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0, max_iter=2000, class_weight="balanced", random_state=seed
            ),
        )
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )
    if name == "lgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=15,
            min_child_samples=5,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
    raise ValueError(f"unknown model {name!r}")


def _fit_predict(name, seed, Xtr, ytr, Xva, Xte):
    """Fit one class; return (val_probs, test_probs). Falls back to the class
    prior if a training fold happens to contain a single label value."""
    if ytr.min() == ytr.max():  # only one class present → predict prior
        p = float(ytr.mean())
        return np.full(len(Xva), p), np.full(len(Xte), p)
    clf = make_model(name, seed)
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xva)[:, 1], clf.predict_proba(Xte)[:, 1]


# ── metrics ──────────────────────────────────────────────────────────
def rank_normalize(P: np.ndarray) -> np.ndarray:
    """Map each class column to uniform ranks in (0, 1] — removes cross-class
    score-scale differences before pooling."""
    return np.column_stack([rankdata(P[:, c]) / len(P) for c in range(P.shape[1])])


def pooled_auc(Y: np.ndarray, P: np.ndarray) -> float:
    return float(roc_auc_score(Y.ravel(), P.ravel()))


def per_class_auc(Y: np.ndarray, P: np.ndarray) -> list[float]:
    out = []
    for c in range(Y.shape[1]):
        yc = Y[:, c]
        out.append(float(roc_auc_score(yc, P[:, c])) if yc.min() != yc.max() else float("nan"))
    return out


# ── run ──────────────────────────────────────────────────────────────
def run(name: str, seed: int = 42, sample: bool = False) -> dict:
    Xtr = features.feature_matrix("train")
    Xte = features.feature_matrix("test")
    Y = data.train_targets()

    classes = range(Y.shape[1])
    if sample:  # smoke: first 12 classes only (folds unchanged)
        classes = range(12)

    folds = cv.make_folds(Y)
    oof = np.zeros_like(Y, dtype=float)
    test = np.zeros((len(Xte), Y.shape[1]), dtype=float)

    t0 = time.time()
    for tr, va in folds:
        for c in classes:
            vp, tp = _fit_predict(name, seed, Xtr[tr], Y[tr, c], Xtr[va], Xte)
            oof[va, c] = vp
            test[:, c] += tp / len(folds)
    secs = time.time() - t0

    cols = list(classes)
    Ys, oofs, tests = Y[:, cols], oof[:, cols], test[:, cols]
    metrics = {
        "model": name,
        "date": str(date.today()),
        "seconds": round(secs, 1),
        "n_classes": len(cols),
        "pooled_auc": pooled_auc(Ys, oofs),
        "pooled_auc_ranknorm": pooled_auc(Ys, rank_normalize(oofs)),
        "macro_auc": float(np.nanmean(per_class_auc(Ys, oofs))),
        "sample": sample,
    }

    if not sample:
        ART.mkdir(parents=True, exist_ok=True)
        np.save(ART / f"{name}_oof.npy", oof)
        np.save(ART / f"{name}_test.npy", test)
        np.save(ART / "y.npy", Y)
        (ART / f"{name}_metrics.json").write_text(json.dumps(metrics, indent=2))
        _append_run_log(name, metrics)

    print(json.dumps(metrics, indent=2))
    return metrics


def _append_run_log(name: str, m: dict) -> None:
    row = (
        f"| {name} | {m['date']} | {name} | agg-MFCC(86) | "
        f"{m['pooled_auc']:.5f} |  |  | ranknorm={m['pooled_auc_ranknorm']:.5f}, "
        f"macro={m['macro_auc']:.4f}, {m['seconds']:.0f}s |\n"
    )
    with open(RUN_LOG, "a") as f:
        f.write(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lgbm", choices=["logreg", "rf", "lgbm"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample", action="store_true", help="quick 12-class smoke run")
    args = ap.parse_args()
    run(args.model, seed=args.seed, sample=args.sample)


if __name__ == "__main__":
    main()
