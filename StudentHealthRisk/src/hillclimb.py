"""Greedy hill-climbing ensemble over saved OOF probability matrices.

Forward selection with *repetition* = weighted probability averaging: at each step add the base model
whose inclusion most improves OOF **balanced accuracy** (scored after re-tuning decision weights on the
candidate blend, since the metric is decision-sensitive). The same integer pick-counts are applied to
the test matrices. Writes ``ensemble_{oof,test}.npy`` + metrics so ``submit.py`` builds from it by name.

    uv run python -m src.hillclimb --models lgbm xgb cat nn_s10 --steps 40
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .metric import score_proba, tune_weights

ART = Path(__file__).resolve().parent.parent / "experiments" / "artifacts"


def _load(model: str):
    return np.load(ART / f"{model}_oof.npy"), np.load(ART / f"{model}_test.npy")


def _tuned(y, mat):
    return score_proba(y, mat, tune_weights(y, mat))


def hillclimb(models: list[str], steps: int = 40, tag: str = "ensemble") -> dict:
    y = np.load(ART / "y.npy")
    oofs, tests = {}, {}
    for m in models:
        oofs[m], tests[m] = _load(m)
    solo = {m: _tuned(y, oofs[m]) for m in models}
    print("base OOF balanced accuracy (tuned):")
    for m in sorted(solo, key=solo.get, reverse=True):
        print(f"  {m:22s} {solo[m]:.5f}")

    picks = Counter()
    best_model = max(solo, key=solo.get)
    ens = oofs[best_model].copy()
    picks[best_model] += 1
    best = solo[best_model]
    for step in range(steps - 1):
        cand = {m: _tuned(y, ens + oofs[m]) for m in models}
        m = max(cand, key=cand.get)
        if cand[m] <= best + 1e-6:                   # no further above-noise improvement
            break
        ens += oofs[m]
        picks[m] += 1
        best = cand[m]
        print(f"  step {step + 2:2d}: +{m:22s} -> OOF {best:.5f}")

    total = sum(picks.values())
    weights = {m: picks[m] / total for m in picks}
    ens_oof = sum(oofs[m] * picks[m] for m in picks) / total
    ens_test = sum(tests[m] * picks[m] for m in picks) / total
    dw = tune_weights(y, ens_oof)
    oof_bacc = score_proba(y, ens_oof, dw)

    res = {"tag": tag, "models": models, "weights": weights, "picks": dict(picks),
           "oof_bacc_tuned": float(oof_bacc), "decision_weights": [float(w) for w in dw],
           "base_oof_bacc": {m: float(solo[m]) for m in models},
           "best_single": float(max(solo.values()))}
    print(f"\nensemble OOF balanced accuracy = {oof_bacc:.5f}  "
          f"(best single {res['best_single']:.5f}, +{oof_bacc - res['best_single']:.5f})")
    print("weights:", {m: round(w, 3) for m, w in weights.items()})

    np.save(ART / f"{tag}_oof.npy", ens_oof)
    np.save(ART / f"{tag}_test.npy", ens_test)
    (ART / f"{tag}_metrics.json").write_text(json.dumps(res, indent=2))
    print(f"saved -> {tag}_oof.npy / _test.npy / _metrics.json")
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True, help="artifact prefixes to ensemble")
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--tag", default="ensemble")
    a = p.parse_args()
    hillclimb(a.models, steps=a.steps, tag=a.tag)
