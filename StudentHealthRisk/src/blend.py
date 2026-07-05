"""Equal-weight blend of base learners; scored on OOF balanced accuracy.

Reads each model's saved OOF/test probability matrices from ``experiments/artifacts/`` and combines
them with equal weight by averaging probabilities. The blended OOF is scored on balanced accuracy
after tuning per-class decision weights on it (``metric.tune_weights``) — the same decision correction
used everywhere. The blended matrices are saved so ``submit.py --model blend`` works unchanged.

Usage:
    uv run python -m src.blend --models lgbm xgb cat
    uv run python -m src.blend --models lgbm xgb cat nn_s10 --tag blend
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .metric import score_proba, tune_weights

ART = Path(__file__).resolve().parent.parent / "experiments" / "artifacts"


def combine(mats: list[np.ndarray]) -> np.ndarray:
    """Equal-weight probability average."""
    return np.mean(mats, axis=0)


def run_blend(models: list[str], tag: str = "blend") -> dict:
    y = np.load(ART / "y.npy")
    oof_mats = [np.load(ART / f"{m}_oof.npy") for m in models]
    test_mats = [np.load(ART / f"{m}_test.npy") for m in models]

    print("base OOF balanced accuracy (tuned):")
    per_model = {}
    for m, o in zip(models, oof_mats):
        w = tune_weights(y, o)
        per_model[m] = score_proba(y, o, w)
        print(f"  {m:10s} {per_model[m]:.5f}")

    oof_blend = combine(oof_mats)
    test_blend = combine(test_mats)
    weights = tune_weights(y, oof_blend)
    blend_raw = score_proba(y, oof_blend)
    blend_tuned = score_proba(y, oof_blend, weights)
    best_base = max(per_model.values())
    print(f"\nblend OOF balanced accuracy: raw {blend_raw:.5f}   tuned {blend_tuned:.5f}"
          f"   (best base {best_base:.5f}, delta {blend_tuned - best_base:+.5f})")

    res = {
        "models": models, "tag": tag,
        "oof_bacc_raw": float(blend_raw), "oof_bacc_tuned": float(blend_tuned),
        "decision_weights": [float(w) for w in weights],
        "base_oof_bacc_tuned": {m: float(s) for m, s in per_model.items()},
        "best_base_oof_bacc": float(best_base),
        "delta_vs_best_base": float(blend_tuned - best_base),
        "n_rows": int(len(y)),
    }
    np.save(ART / f"{tag}_oof.npy", oof_blend)
    np.save(ART / f"{tag}_test.npy", test_blend)
    (ART / f"{tag}_metrics.json").write_text(json.dumps(res, indent=2))
    print(f"saved -> {ART}/{tag}_oof.npy, {tag}_test.npy, {tag}_metrics.json")
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["lgbm", "xgb", "cat"],
                   help="base-learner artifact prefixes to blend")
    p.add_argument("--tag", default="blend", help="artifact filename prefix for the blend")
    a = p.parse_args()
    run_blend(a.models, a.tag)
