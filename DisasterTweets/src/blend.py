"""Rank-average ensemble of saved OOF/test predictions, with F1 threshold tuning.

Usage:
    uv run python -m src.blend                       # blend default members
    uv run python -m src.blend --members tfidf_lr roberta --weights 1 2

Reads ``{model}_oof.npy`` / ``{model}_test.npy`` from experiments/artifacts, rank-averages
them (rank-averaging is scale-free, so a logistic-regression probability and a softmax
probability combine sanely), tunes the decision threshold on the blended OOF, and writes
``blend_*`` artifacts. Only worth adopting if blended OOF F1 beats the best single member
by more than its fold std.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import f1_score

from src.train import tune_threshold

ART = Path(__file__).resolve().parent.parent / "experiments" / "artifacts"

DEFAULT_MEMBERS = ["tfidf_lr", "roberta"]


def _rank01(x: np.ndarray) -> np.ndarray:
    """Map scores to ranks in [0, 1] (ties averaged)."""
    return (rankdata(x) - 1) / (len(x) - 1)


def blend(members: list[str], weights: list[float] | None = None) -> dict:
    y = np.load(ART / "y.npy")
    weights = weights or [1.0] * len(members)
    w = np.asarray(weights, dtype=float)
    w /= w.sum()

    oof = np.zeros(len(y))
    test = None
    member_f1s = {}
    for wi, m in zip(w, members):
        m_oof = np.load(ART / f"{m}_oof.npy")
        m_test = np.load(ART / f"{m}_test.npy")
        oof += wi * _rank01(m_oof)
        test = wi * _rank01(m_test) if test is None else test + wi * _rank01(m_test)
        with open(ART / f"{m}_metrics.json") as fh:
            member_f1s[m] = json.load(fh)["oof_f1"]

    threshold, oof_f1 = tune_threshold(y, oof)

    np.save(ART / "blend_oof.npy", oof)
    np.save(ART / "blend_test.npy", test)
    metrics = {
        "model": "blend",
        "family": "ensemble",
        "members": members,
        "weights": [round(float(x), 4) for x in w],
        "member_oof_f1": member_f1s,
        "best_single": max(member_f1s.values()),
        "oof_f1": round(oof_f1, 5),
        "f1_at_0.5": round(f1_score(y, (oof >= 0.5).astype(int)), 5),
        "threshold": round(threshold, 4),
        "n_rows": int(len(y)),
    }
    with open(ART / "blend_metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)

    delta = oof_f1 - metrics["best_single"]
    print(f"members: {dict(zip(members, [round(x,4) for x in w]))}")
    for m, f in member_f1s.items():
        print(f"  {m:12s} OOF F1 {f:.5f}")
    print(f"[blend] OOF F1 = {oof_f1:.5f} @ t={threshold:.3f}  "
          f"(best single {metrics['best_single']:.5f}; delta {delta:+.5f})")
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", nargs="+", default=DEFAULT_MEMBERS)
    ap.add_argument("--weights", nargs="+", type=float, default=None)
    args = ap.parse_args()
    blend(args.members, args.weights)


if __name__ == "__main__":
    main()
