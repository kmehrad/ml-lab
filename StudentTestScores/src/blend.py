"""Equal-weight blend of base learners; scored on OOF RMSE.

Reads each model's saved OOF/test prediction vectors from ``experiments/artifacts/`` and
averages them. The blended vector is scored exactly like a single model, so
``submit.py --model blend`` works unchanged.

For regression, simple probability-style averaging doesn't apply — we average the predicted
scores directly. An equal-weight average of decorrelated learners typically beats the best
single model and generalises better than OOF-tuned weights.

Usage:
    uv run python -m src.blend                          # average lgbm + xgb (default)
    uv run python -m src.blend --models lgbm xgb cat    # include catboost for diversity
    uv run python -m src.blend --models lgbm xgb --tag blend
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .metric import rmse

ART = Path(__file__).resolve().parent.parent / "experiments" / "artifacts"


def run_blend(models: list[str], tag: str = "blend") -> dict:
    y = np.load(ART / "y.npy")
    oof_mats = [np.load(ART / f"{m}_oof.npy") for m in models]
    test_mats = [np.load(ART / f"{m}_test.npy") for m in models]

    per_model = {m: rmse(y, o) for m, o in zip(models, oof_mats)}
    print("base OOF RMSE:")
    for m, s in per_model.items():
        print(f"  {m:6s} {s:.5f}")

    oof_blend = np.mean(oof_mats, axis=0)
    test_blend = np.mean(test_mats, axis=0)
    blend_rmse = rmse(y, oof_blend)
    best_base = min(per_model.values())
    print(f"\nOOF RMSE   blend {blend_rmse:.5f}   (best base {best_base:.5f}"
          f"   delta {blend_rmse - best_base:+.5f}, negative = better)")

    res = {
        "models": models, "tag": tag,
        "oof_rmse": float(blend_rmse),
        "base_oof_rmse": {m: float(s) for m, s in per_model.items()},
        "best_base_oof_rmse": float(best_base),
        "delta_vs_best_base": float(blend_rmse - best_base),
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
    p.add_argument("--tag", default="blend", help="artifact filename prefix for the blend")
    a = p.parse_args()
    run_blend(a.models, a.tag)
