"""Greedy hill-climbing ensemble over saved OOF probability matrices (Phase 6).

Forward selection with *repetition* = weighted probability averaging: at each step add the base
model whose inclusion most improves OOF MAP@3. Probabilities are summed (not rank-normalised —
exp-009 showed rank-averaging breaks the within-row calibration MAP@3 needs); since MAP@3 depends
only on the per-row argsort, summing unnormalised probabilities is equivalent to averaging. The
same integer pick-counts are applied to the test matrices. Writes ``ensemble_{oof,test}.npy`` +
metrics so ``submit.py`` can build from it by name.

    uv run python -m src.hillclimb --models xgb_te2_cuml nn_te2 xgb_te2_d4 --steps 60
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .metric import mapk3

ART = Path(__file__).resolve().parent.parent / "experiments" / "artifacts"


def _load(model: str):
    return (np.load(ART / f"{model}_oof.npy"), np.load(ART / f"{model}_test.npy"))


def hillclimb(models: list[str], steps: int = 60, tag: str = "ensemble") -> dict:
    y = np.load(ART / "y.npy")
    oofs, tests = {}, {}
    for m in models:
        oofs[m], tests[m] = _load(m)
    solo = {m: mapk3(y, oofs[m]) for m in models}
    print("base OOF MAP@3:")
    for m in sorted(solo, key=solo.get, reverse=True):
        print(f"  {m:22s} {solo[m]:.5f}")

    # seed the ensemble with the single best model, then greedily add (with repetition).
    picks = Counter()
    best_model = max(solo, key=solo.get)
    ens = oofs[best_model].copy()
    picks[best_model] += 1
    best = solo[best_model]
    for step in range(steps - 1):
        cand_scores = {m: mapk3(y, ens + oofs[m]) for m in models}
        m = max(cand_scores, key=cand_scores.get)
        if cand_scores[m] <= best + 1e-7:            # no further above-noise improvement
            break
        ens += oofs[m]
        picks[m] += 1
        best = cand_scores[m]
        print(f"  step {step + 2:2d}: +{m:22s} -> OOF {best:.5f}")

    total = sum(picks.values())
    weights = {m: picks[m] / total for m in picks}
    ens_oof = sum(oofs[m] * picks[m] for m in picks) / total
    ens_test = sum(tests[m] * picks[m] for m in picks) / total
    oof_map3 = mapk3(y, ens_oof)

    res = {"tag": tag, "models": models, "weights": weights, "picks": dict(picks),
           "oof_map3": float(oof_map3), "base_oof": {m: float(solo[m]) for m in models},
           "best_single": max(solo.values())}
    print(f"\nensemble OOF MAP@3 = {oof_map3:.5f}  "
          f"(best single {res['best_single']:.5f}, +{oof_map3 - res['best_single']:.5f})")
    print("weights:", {m: round(w, 3) for m, w in weights.items()})

    np.save(ART / f"{tag}_oof.npy", ens_oof)
    np.save(ART / f"{tag}_test.npy", ens_test)
    (ART / f"{tag}_metrics.json").write_text(json.dumps(res, indent=2))
    print(f"saved -> {tag}_oof.npy / _test.npy / _metrics.json")
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True, help="artifact prefixes to ensemble")
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--tag", default="ensemble")
    a = p.parse_args()
    hillclimb(a.models, steps=a.steps, tag=a.tag)
