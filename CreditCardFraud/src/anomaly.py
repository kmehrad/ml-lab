"""Unsupervised anomaly-detection baselines.

The supervised models in ``src.train`` are the project's primary approach (labels
exist and dominate), but the classic framing of this dataset is *anomaly
detection*. These baselines ignore the labels at fit time and flag transactions
purely by how unusual they look, then are scored against the labels with AUPRC so
they sit in the same comparison table as the supervised models.

* **Isolation Forest** (``iforest``) — isolates points with random splits; fraud is
  rarer/easier to isolate, so a higher (negated) path-based score is more anomalous.
* **Local Outlier Factor** (``lof``, ``novelty=True``) — local density relative to
  neighbours.
* **One-Class SVM** (``ocsvm``) — ``SGDOneClassSVM`` on a ``Nystroem`` RBF kernel
  approximation: the linear/SGD form scales to the full train remainder where the
  exact kernel ``OneClassSVM`` (O(n^2)) does not. Score is signed distance to the
  learned boundary.
* **Elliptic Envelope** (``elliptic``) — robust (MCD) Gaussian fit; score is the
  Mahalanobis distance. A natural fit since ``V1``..``V28`` are already PCA outputs.
* **Gaussian Mixture** (``gmm``) — density model; score is the negative
  log-likelihood (low density == anomalous).
* **PCA reconstruction** (``pca``) — projects to a low-rank subspace and measures
  reconstruction error; points off the normal manifold reconstruct poorly.

Every detector fits on the train-remainder rows only and predicts the fixed
held-out test set, saving ``{name}_test.npy`` so ``src.evaluate`` can score it like
any other model. Anomaly scores are min-max scaled to ``[0, 1]`` so they read as a
pseudo-probability. By convention a detector's ``score_samples`` returns *higher ==
more normal*; ``run_detector`` negates it to get an anomaly score.

Usage
-----
    python -m src.anomaly --model ocsvm
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
ANOMALY_MODELS = ("iforest", "lof", "ocsvm", "elliptic", "gmm", "pca")

# Rank of the PCA subspace kept for the reconstruction-error detector.
PCA_RECON_COMPONENTS = 15


class _PCAReconstruction:
    """PCA reconstruction-error detector with an sklearn-style ``score_samples``.

    Fits a low-rank PCA on the (normal-dominated) training data; the anomaly signal
    is the squared reconstruction error after projecting to and back from that
    subspace. ``score_samples`` returns the *negative* error so that, as with the
    other detectors, higher means more normal.
    """

    def __init__(self, n_components: int, seed: int) -> None:
        self.n_components = n_components
        self.seed = seed

    def fit(self, X, y=None):  # noqa: N803
        from sklearn.decomposition import PCA

        n = min(self.n_components, np.asarray(X).shape[1])
        self.pca_ = PCA(n_components=n, random_state=self.seed).fit(X)
        return self

    def score_samples(self, X):  # noqa: N803
        X = np.asarray(X)
        reconstructed = self.pca_.inverse_transform(self.pca_.transform(X))
        return -((X - reconstructed) ** 2).sum(axis=1)


def _build_detector(name: str, contamination: float, seed: int):
    if name == "iforest":
        from sklearn.ensemble import IsolationForest

        return IsolationForest(
            n_estimators=300, contamination=contamination, random_state=seed, n_jobs=-1
        )
    if name == "lof":
        from sklearn.neighbors import LocalOutlierFactor

        return LocalOutlierFactor(n_neighbors=20, contamination=contamination, novelty=True)
    if name == "ocsvm":
        from sklearn.kernel_approximation import Nystroem
        from sklearn.linear_model import SGDOneClassSVM
        from sklearn.pipeline import make_pipeline

        # nu ~ outlier fraction; floor it so the SGD fit stays well-conditioned
        # (the score ranking, hence AUPRC, is insensitive to the exact value).
        nu = float(np.clip(contamination, 0.01, 0.5))
        return make_pipeline(
            Nystroem(n_components=300, random_state=seed),
            SGDOneClassSVM(nu=nu, max_iter=2000, tol=1e-4, random_state=seed),
        )
    if name == "elliptic":
        from sklearn.covariance import EllipticEnvelope

        return EllipticEnvelope(contamination=contamination, random_state=seed)
    if name == "gmm":
        from sklearn.mixture import GaussianMixture

        return GaussianMixture(
            n_components=8, covariance_type="full", reg_covar=1e-4,
            max_iter=200, random_state=seed,
        )
    if name == "pca":
        return _PCAReconstruction(n_components=PCA_RECON_COMPONENTS, seed=seed)
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

    # Most detectors here are distance/kernel/density-based, so scale consistently
    # (RobustScaler on raw-scale cols) for a fair comparison; the scale-invariant
    # tree detector is unaffected. Use numpy arrays — detectors need no feature names.
    pre = build_preprocessor("linear").fit(X)
    X_t = pre.transform(X).to_numpy()
    X_test_t = pre.transform(X_test).to_numpy()

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
    parser.add_argument(
        "--model", default="iforest",
        help="iforest, lof, ocsvm, elliptic, gmm, pca, or 'all'",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()

    models = ANOMALY_MODELS if args.model == "all" else (args.model,)
    for name in models:
        run_detector(name, seed=args.seed, sample=args.sample)


if __name__ == "__main__":
    main()
