"""Unsupervised anomaly-detection baselines.

The supervised models in ``src.train`` are the project's primary approach (labels
exist and dominate), but the classic framing of this dataset is *anomaly
detection*. These baselines ignore the labels at fit time and flag transactions
purely by how unusual they look, then are scored against the labels with AUPRC so
they sit in the same comparison table as the supervised models.

* **Isolation Forest** — isolates points with random splits; fraud is rarer/easier
  to isolate, so a higher (negated) path-based score means more anomalous.
* **Local Outlier Factor** (``novelty=True``) — density relative to neighbours;
  fit on the train remainder, scored on the held-out test set.

Both fit on the train-remainder rows only and predict the fixed held-out test set,
saving ``{name}_test.npy`` so ``src.evaluate`` can score them like any other model.
Anomaly scores are min-max scaled to ``[0, 1]`` so they read as a pseudo-probability.

Usage
-----
    python -m src.anomaly --model iforest
    python -m src.anomaly --model all
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .data import load_raw, make_holdout_split, split_features_target
from .preprocessing import build_preprocessor
from .train import _record_results

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "experiments" / "artifacts"
ANOMALY_MODELS = ("iforest", "lof")


def _build_detector(name: str, contamination: float, seed: int):
    if name == "iforest":
        from sklearn.ensemble import IsolationForest

        return IsolationForest(
            n_estimators=300, contamination=contamination, random_state=seed, n_jobs=-1
        )
    if name == "lof":
        from sklearn.neighbors import LocalOutlierFactor

        return LocalOutlierFactor(n_neighbors=20, contamination=contamination, novelty=True)
    raise ValueError(f"Unknown detector {name!r}; choose from {ANOMALY_MODELS}")


def _minmax(scores: np.ndarray) -> np.ndarray:
    lo, hi = scores.min(), scores.max()
    return (scores - lo) / (hi - lo) if hi > lo else np.zeros_like(scores)


def run_detector(name: str, seed: int = 42, sample: int | None = None) -> dict:
    """Fit an unsupervised detector on the train remainder, score the test set."""
    frame = load_raw()
    train_idx, test_idx = make_holdout_split(seed=seed)
    train = frame.iloc[train_idx].reset_index(drop=True)
    test = frame.iloc[test_idx].reset_index(drop=True)
    if sample is not None:
        train = train.sample(min(sample, len(train)), random_state=seed).reset_index(drop=True)

    X, y = split_features_target(train)
    X_test, y_test = split_features_target(test)
    y_test = y_test.to_numpy()
    contamination = float(max(y.mean(), 1e-4))

    # Tree-based detectors are scale-invariant, but LOF is distance-based; scale both
    # the same way (RobustScaler on raw-scale cols) for a fair comparison.
    pre = build_preprocessor("linear").fit(X)
    X_t, X_test_t = pre.transform(X), pre.transform(X_test)

    start = time.perf_counter()
    det = _build_detector(name, contamination, seed)
    det.fit(X_t)
    # Higher score_samples == more normal, so negate for an "anomaly score".
    test_scores = _minmax(-det.score_samples(X_test_t))
    elapsed = time.perf_counter() - start

    metrics = {
        "model": name,
        "base_model": name,
        "family": "anomaly",
        "imbalance": "unsupervised",
        "folds": 0,
        "seed": seed,
        "n_rows": int(len(y)),
        "oof_ap": None,
        "oof_ap_fold_mean": None,
        "oof_ap_fold_std": None,
        "oof_roc_auc": None,
        "test_ap": round(float(average_precision_score(y_test, test_scores)), 5),
        "test_roc_auc": round(float(roc_auc_score(y_test, test_scores)), 5),
        "train_seconds": round(elapsed, 1),
    }
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(ARTIFACTS_DIR / f"{name}_test.npy", test_scores)
    _record_results(metrics)
    print(
        f"{name:>12}: test_ap={metrics['test_ap']:.5f}  "
        f"test_auc={metrics['test_roc_auc']:.5f}  ({metrics['train_seconds']}s)"
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="iforest", help="iforest, lof, or 'all'")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()

    models = ANOMALY_MODELS if args.model == "all" else (args.model,)
    for name in models:
        run_detector(name, seed=args.seed, sample=args.sample)


if __name__ == "__main__":
    main()
