"""Equal-weight blend of base learners; scored on OOF MAP@3.

Reads each model's saved OOF/test probability matrices from ``experiments/artifacts/`` and
combines them with equal weight, either by averaging probabilities directly (``proba``) or by
averaging per-column normalised ranks (``rank``). The blended score matrix is argsorted per row
for the top-3, exactly like a single model — so ``submit.py --model blend`` works unchanged.

**Method finding (this task):** rank-averaging *hurts* here. MAP@3 ranks the 7 classes *within
each row*; per-column rank-normalising across the whole dataset flattens every class to a
uniform marginal and destroys the calibration MAP@3 needs. Plain probability-averaging preserves
it, so ``proba`` is the default — contrary to the sibling-project rank-average preference, which
assumed binary/low-class ranking. Both scores are always printed for comparison.

Usage:
    uv run python -m src.blend                          # proba-average lgbm + xgb (default)
    uv run python -m src.blend --method rank            # rank-average (worse here)
    uv run python -m src.blend --models lgbm xgb cat    # include catboost for diversity
    uv run python -m src.blend --models lgbm xgb --tag blend
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from .metric import mapk3

ART = Path(__file__).resolve().parent.parent / "experiments" / "artifacts"


def _rank_normalise(proba: np.ndarray) -> np.ndarray:
    """Per-column average-rank in ``(0, 1]`` so columns are comparable within a row."""
    n = proba.shape[0]
    ranks = np.empty_like(proba, dtype=float)
    for j in range(proba.shape[1]):
        ranks[:, j] = rankdata(proba[:, j], method="average") / n
    return ranks


def combine(mats: list[np.ndarray], method: str) -> np.ndarray:
    """Equal-weight blend: mean of probabilities (``proba``) or of normalised ranks (``rank``)."""
    if method == "proba":
        return np.mean(mats, axis=0)
    if method == "rank":
        return np.mean([_rank_normalise(m) for m in mats], axis=0)
    raise ValueError(f"unknown method {method}")


def run_blend(models: list[str], method: str = "proba", tag: str = "blend") -> dict:
    y = np.load(ART / "y.npy")
    oof_mats = [np.load(ART / f"{m}_oof.npy") for m in models]
    test_mats = [np.load(ART / f"{m}_test.npy") for m in models]

    per_model = {m: mapk3(y, o) for m, o in zip(models, oof_mats)}
    print("base OOF MAP@3:")
    for m, s in per_model.items():
        print(f"  {m:6s} {s:.5f}")

    # Score both methods for comparison; save the requested one.
    scores = {mth: mapk3(y, combine(oof_mats, mth)) for mth in ("proba", "rank")}
    best_base = max(per_model.values())
    print(f"\nOOF MAP@3   proba-average {scores['proba']:.5f}   rank-average {scores['rank']:.5f}"
          f"   (best base {best_base:.5f})")
    print(f"chosen method '{method}': {scores[method]:.5f}   "
          f"delta vs best base {scores[method] - best_base:+.5f}")

    oof_blend = combine(oof_mats, method)
    test_blend = combine(test_mats, method)

    res = {
        "method": method, "models": models, "tag": tag,
        "oof_map3": float(scores[method]),
        "oof_map3_by_method": {k: float(v) for k, v in scores.items()},
        "base_oof_map3": {m: float(s) for m, s in per_model.items()},
        "best_base_oof_map3": float(best_base),
        "delta_vs_best_base": float(scores[method] - best_base),
        "n_rows": int(len(y)),
    }

    np.save(ART / f"{tag}_oof.npy", oof_blend)
    np.save(ART / f"{tag}_test.npy", test_blend)
    (ART / f"{tag}_metrics.json").write_text(json.dumps(res, indent=2))
    print(f"saved -> {ART}/{tag}_oof.npy, {tag}_test.npy, {tag}_metrics.json")
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["lgbm", "xgb"],
                   help="base-learner artifact prefixes to blend (default: lgbm xgb)")
    p.add_argument("--method", default="proba", choices=["proba", "rank"],
                   help="equal-weight probability-average (default) or rank-average")
    p.add_argument("--tag", default="blend", help="artifact filename prefix for the blend")
    a = p.parse_args()
    run_blend(a.models, a.method, a.tag)
