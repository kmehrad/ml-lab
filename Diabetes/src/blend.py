"""Out-of-fold probability blending for the Diabetes challenge.

Each model in ``src.train`` saves its out-of-fold positive-class probabilities to
``experiments/artifacts/{model}_oof.npy`` and the shared label vector to
``y_true.npy``. This module loads those, searches for non-negative weights that
maximize the blended out-of-fold ROC-AUC, and persists the chosen weights so
``src.submit`` can apply the same blend to the test predictions.

Because AUC depends only on the ranking, blending is done on the *rank* of each
model's probabilities (rank-averaging), which is robust to differing probability
calibration across libraries. Weights are optimised with SLSQP on the simplex.

Usage
-----
    python -m src.blend                       # blend all available OOF arrays
    python -m src.blend --models lgbm xgb catboost
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "experiments" / "artifacts"
WEIGHTS_JSON = ARTIFACTS_DIR / "blend_weights.json"


def _load_oof(models: list[str]) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return ``(names, ranks_matrix, y_true)`` for models with saved OOF arrays."""
    y = np.load(ARTIFACTS_DIR / "y_true.npy")
    names, cols = [], []
    for name in models:
        path = ARTIFACTS_DIR / f"{name}_oof.npy"
        if not path.exists():
            print(f"  skipping {name!r}: {path.name} not found")
            continue
        oof = np.load(path)
        if np.isnan(oof).any():
            print(f"  skipping {name!r}: OOF contains NaN (incomplete CV)")
            continue
        names.append(name)
        cols.append(rankdata(oof) / len(oof))  # normalised ranks in (0, 1]
    if not names:
        raise RuntimeError("No usable OOF arrays found; run src.train first.")
    return names, np.column_stack(cols), y


def find_blend(models: list[str]) -> dict:
    """Search simplex weights that maximise blended OOF AUC; persist and return."""
    names, ranks, y = _load_oof(models)
    n = len(names)
    singles = {nm: round(float(roc_auc_score(y, ranks[:, i])), 5) for i, nm in enumerate(names)}

    def neg_auc(w: np.ndarray) -> float:
        return -roc_auc_score(y, ranks @ w)

    best = None
    # Multi-start (equal weights + each model dominant) to avoid local optima.
    starts = [np.full(n, 1.0 / n)] + list(np.eye(n))
    constraints = {"type": "eq", "fun": lambda w: w.sum() - 1.0}
    bounds = [(0.0, 1.0)] * n
    for x0 in starts:
        res = minimize(neg_auc, x0, method="SLSQP", bounds=bounds, constraints=constraints)
        if best is None or res.fun < best.fun:
            best = res

    weights = np.clip(best.x, 0.0, None)
    weights = weights / weights.sum()
    blended_auc = round(float(roc_auc_score(y, ranks @ weights)), 5)

    result = {
        "models": names,
        "weights": [round(float(w), 4) for w in weights],
        "rank_average": True,
        "single_auc": singles,
        "blended_auc": blended_auc,
        "best_single_auc": max(singles.values()),
    }
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_JSON.write_text(json.dumps(result, indent=2))

    print("Single-model OOF AUC (rank-scaled):")
    for nm in names:
        print(f"  {nm:>9}: {singles[nm]:.5f}")
    print("Blend weights:")
    for nm, w in zip(names, weights):
        print(f"  {nm:>9}: {w:.4f}")
    print(f"Blended OOF AUC: {blended_auc:.5f}  (best single {result['best_single_auc']:.5f})")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", nargs="+", default=["lgbm", "xgb", "catboost", "histgb", "baseline"],
        help="candidate models to blend (uses those with saved OOF arrays)",
    )
    args = parser.parse_args()
    find_blend(args.models)


if __name__ == "__main__":
    main()
