"""Greedy hill-climbing ensemble over saved OOF prediction vectors.

Forward selection with *repetition* = weighted averaging: at each step add whichever base model's
inclusion most improves (lowers) OOF RMSE. Unlike a plain equal-weight blend, this lets strong or
diverse models earn more than one "share" of the final average and lets weak ones be skipped
entirely — it can also mix in weaker/tuned variants when they still add residual diversity even if
their solo score is worse. The same integer pick-counts are applied to the test vectors. Writes
``ensemble_{oof,test}.npy`` + metrics so ``submit.py`` builds from it by name.

    uv run python -m src.hillclimb --models lgbm xgb cat nn_nn_s3 --steps 40
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .metric import rmse

ART = Path(__file__).resolve().parent.parent / "experiments" / "artifacts"


def _load(model: str):
    return np.load(ART / f"{model}_oof.npy"), np.load(ART / f"{model}_test.npy")


def hillclimb(models: list[str], steps: int = 40, tag: str = "ensemble") -> dict:
    y = np.load(ART / "y.npy")
    oofs, tests = {}, {}
    for m in models:
        oofs[m], tests[m] = _load(m)
    solo = {m: rmse(y, oofs[m]) for m in models}
    print("base OOF RMSE:")
    for m in sorted(solo, key=solo.get):
        print(f"  {m:20s} {solo[m]:.5f}")

    picks = Counter()
    best_model = min(solo, key=solo.get)
    ens_sum = oofs[best_model].copy()
    picks[best_model] += 1
    best = solo[best_model]
    print(f"\n  step  1: start   {best_model:20s} -> OOF {best:.5f}")
    for step in range(steps - 1):
        n = sum(picks.values())
        cand = {m: rmse(y, (ens_sum + oofs[m]) / (n + 1)) for m in models}
        m = min(cand, key=cand.get)
        if cand[m] >= best - 1e-6:                   # no further above-noise improvement
            break
        ens_sum += oofs[m]
        picks[m] += 1
        best = cand[m]
        print(f"  step {step + 2:2d}: +{m:20s} -> OOF {best:.5f}")

    total = sum(picks.values())
    weights = {m: picks[m] / total for m in picks}
    ens_oof = sum(oofs[m] * picks[m] for m in picks) / total
    ens_test = sum(tests[m] * picks[m] for m in picks) / total
    oof_rmse = rmse(y, ens_oof)

    res = {
        "tag": tag, "models": models, "weights": weights, "picks": dict(picks),
        "oof_rmse": float(oof_rmse), "base_oof_rmse": {m: float(solo[m]) for m in models},
        "best_single": float(min(solo.values())),
    }
    print(f"\nensemble OOF RMSE = {oof_rmse:.5f}  "
          f"(best single {res['best_single']:.5f}, delta {oof_rmse - res['best_single']:+.5f})")
    print("weights:", {m: round(w, 3) for m, w in weights.items()})

    np.save(ART / f"{tag}_oof.npy", ens_oof)
    np.save(ART / f"{tag}_test.npy", ens_test)
    (ART / f"{tag}_metrics.json").write_text(json.dumps(res, indent=2))
    print(f"saved -> {tag}_oof.npy, {tag}_test.npy, {tag}_metrics.json")
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True, help="artifact prefixes to ensemble")
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--tag", default="ensemble")
    a = p.parse_args()
    hillclimb(a.models, steps=a.steps, tag=a.tag)
