"""Classic TF-IDF baseline with out-of-fold F1 and threshold tuning.

Usage:
    uv run python -m src.train --model tfidf            # full 5-fold run
    uv run python -m src.train --model tfidf --sample 1500   # quick smoke run
    uv run python -m src.train --model all              # tfidf_lr, tfidf_svc, tfidf_nb

Each model writes ``experiments/artifacts/{key}_oof.npy``, ``{key}_test.npy`` and
``{key}_metrics.json``. The metric is **F1**; the decision threshold is tuned on the OOF
scores and stored in the metrics JSON for the submission step.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from src import cv, data, features

ART = Path(__file__).resolve().parent.parent / "experiments" / "artifacts"
ART.mkdir(parents=True, exist_ok=True)


def _vectorizer() -> FeatureUnion:
    """Word (1-2gram) + char (3-5gram) TF-IDF — char grams handle OOV/typos/hashtags."""
    return FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                                 min_df=2, sublinear_tf=True, strip_accents="unicode")),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                 min_df=3, sublinear_tf=True)),
    ])


def _estimator(model: str):
    if model == "tfidf_lr":
        return LogisticRegression(C=1.0, max_iter=1000, class_weight=None)
    if model == "tfidf_svc":
        return LinearSVC(C=0.5)  # scored via decision_function -> sigmoid
    if model == "tfidf_nb":
        return ComplementNB(alpha=0.3)
    raise ValueError(f"unknown model {model!r}")


def _scores(est, X) -> np.ndarray:
    """Return a 1-D positive-class score in [0, 1] regardless of estimator type."""
    if hasattr(est, "predict_proba"):
        return est.predict_proba(X)[:, 1]
    return expit(est.decision_function(X))  # LinearSVC


def tune_threshold(y: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Pick the threshold maximizing F1 on (y, scores). Returns (threshold, f1)."""
    order = np.unique(scores)
    cands = (order[:-1] + order[1:]) / 2 if len(order) > 1 else order
    cands = np.r_[0.0, cands, 1.0]
    best_t, best_f1 = 0.5, -1.0
    for t in cands:
        f1 = f1_score(y, (scores >= t).astype(int))
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    return best_t, best_f1


def run_cv(model: str, sample: int | None = None, fix_dups: bool = False) -> dict:
    key = model
    tr = data.load_train()
    te = data.load_test()
    if fix_dups:
        tr = features.fix_duplicate_labels(tr)
    if sample:
        tr = tr.sample(n=min(sample, len(tr)), random_state=cv.SEED).reset_index(drop=True)

    X_text = features.build_field(tr).to_numpy()
    X_test_text = features.build_field(te).to_numpy()
    y = tr[data.TARGET].to_numpy()

    folds = cv.make_folds(y)
    oof = np.zeros(len(y))
    test_pred = np.zeros(len(te))
    fold_f1s = []
    t0 = time.time()

    for f, (tr_idx, va_idx) in enumerate(folds):
        pipe = Pipeline([("vec", _vectorizer()), ("clf", _estimator(model))])
        pipe.fit(X_text[tr_idx], y[tr_idx])
        va_scores = _scores(pipe.named_steps["clf"], pipe.named_steps["vec"].transform(X_text[va_idx]))
        oof[va_idx] = va_scores
        test_pred += _scores(pipe.named_steps["clf"], pipe.named_steps["vec"].transform(X_test_text)) / len(folds)
        # per-fold F1 at 0.5 (diagnostic; global threshold tuned on full OOF below)
        fold_f1s.append(f1_score(y[va_idx], (va_scores >= 0.5).astype(int)))
        print(f"  fold {f}: F1@0.5 = {fold_f1s[-1]:.4f}")

    threshold, oof_f1 = tune_threshold(y, oof)
    elapsed = time.time() - t0

    np.save(ART / f"{key}_oof.npy", oof)
    np.save(ART / f"{key}_test.npy", test_pred)
    if not sample:  # persist shared label/id vectors once, from a full run
        np.save(ART / "y.npy", y)
        np.save(ART / "test_ids.npy", te[data.ID_COL].to_numpy())

    metrics = {
        "model": key,
        "family": "classic",
        "oof_f1": round(oof_f1, 5),
        "f1_at_0.5": round(f1_score(y, (oof >= 0.5).astype(int)), 5),
        "threshold": round(threshold, 4),
        "fold_mean": round(float(np.mean(fold_f1s)), 5),
        "fold_std": round(float(np.std(fold_f1s)), 5),
        "fold_f1s": [round(x, 5) for x in fold_f1s],
        "n_rows": int(len(y)),
        "fix_dups": fix_dups,
        "sample": sample,
        "elapsed_s": round(elapsed, 1),
    }
    with open(ART / f"{key}_metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)

    print(f"[{key}] OOF F1 = {oof_f1:.5f} @ t={threshold:.3f}  "
          f"(F1@0.5 {metrics['f1_at_0.5']:.5f}; fold {metrics['fold_mean']:.5f}±{metrics['fold_std']:.5f}; "
          f"{elapsed:.1f}s)")
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tfidf",
                    help="tfidf|tfidf_lr|tfidf_svc|tfidf_nb|all")
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--fix-dups", action="store_true")
    args = ap.parse_args()

    alias = {"tfidf": "tfidf_lr"}
    if args.model == "all":
        models = ["tfidf_lr", "tfidf_svc", "tfidf_nb"]
    else:
        models = [alias.get(args.model, args.model)]

    for m in models:
        print(f"=== {m} ===")
        run_cv(m, sample=args.sample, fix_dups=args.fix_dups)


if __name__ == "__main__":
    main()
