"""Blend model probability arrays → OOF pooled AUC + a test artifact to submit.

Members are combined by **global rank-average**: each model's full (N×87) array
is rank-normalized over *all* entries (a monotonic transform that preserves the
cross-class ordering the pooled-AUC metric needs) and the ranks are averaged.
This puts models with different score scales (RF proba vs CNN sigmoid) on common
footing. ``--method mean`` averages raw probabilities instead.

Writes ``experiments/artifacts/{name}_oof.npy`` / ``{name}_test.npy`` so
``src.submit --name {name}`` can build the submission.

Usage:
    uv run python -m src.blend --models rf_full,cnn --name blend
"""
from __future__ import annotations

import argparse

import numpy as np
from scipy.stats import rankdata

from src.train_clf import ART, pooled_auc


def _global_rank(A: np.ndarray) -> np.ndarray:
    r = rankdata(A.ravel()).reshape(A.shape)
    return r / r.size


def combine(arrays: list[np.ndarray], method: str = "rank") -> np.ndarray:
    mats = [_global_rank(A) for A in arrays] if method == "rank" else arrays
    return np.mean(mats, axis=0)


def run(names: list[str], method: str = "rank", save_as: str | None = None) -> float:
    Y = np.load(ART / "y.npy")
    oofs = [np.load(ART / f"{n}_oof.npy") for n in names]
    tests = [np.load(ART / f"{n}_test.npy") for n in names]

    for n, o in zip(names, oofs):
        print(f"  member {n:14s} OOF pooled AUC {pooled_auc(Y, o):.5f}")
    b_oof = combine(oofs, method)
    auc = pooled_auc(Y, b_oof)
    print(f"  BLEND [{method}] {'+'.join(names)} → OOF pooled AUC {auc:.5f}")

    if save_as:
        np.save(ART / f"{save_as}_oof.npy", b_oof)
        np.save(ART / f"{save_as}_test.npy", combine(tests, method))
        print(f"  saved {save_as}_oof.npy / {save_as}_test.npy")
    return auc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="comma-separated artifact prefixes")
    ap.add_argument("--method", default="rank", choices=["rank", "mean"])
    ap.add_argument("--name", default=None, help="save blended arrays under this prefix")
    args = ap.parse_args()
    run([m.strip() for m in args.models.split(",")], method=args.method, save_as=args.name)


if __name__ == "__main__":
    main()
