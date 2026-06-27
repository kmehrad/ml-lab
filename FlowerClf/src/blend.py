"""Ensemble several runs by averaging softmax probs (aligned by image id).

Reports blended **val macro F1** vs each member, and writes blended test probs under a
new run id so ``src/submit.py --run {out}`` can build the submission.

Adopt a blend only if it beats the best single member by more than noise.

Usage:
    uv run python -m src.blend --runs b3_r224,convnext_tiny_r224 --out blend_b3_cvnxt
"""
from __future__ import annotations

import argparse

import numpy as np
from sklearn.metrics import f1_score

from src import data
from src.train import ARTIFACTS


def _load(run: str, split: str):
    probs = np.load(ARTIFACTS / f"{run}_{split}_probs.npy")
    ids = np.load(ARTIFACTS / f"{run}_{split}_ids.npy", allow_pickle=True).astype(str)
    return probs, ids


def _align(runs, split, ref_ids):
    """Stack each run's probs reordered to ``ref_ids``. Shape: (n_runs, n, n_classes)."""
    stacked = []
    for run in runs:
        probs, ids = _load(run, split)
        order = {i: k for k, i in enumerate(ids)}
        if set(ids) != set(ref_ids):
            raise ValueError(f"{run} {split} ids differ from reference")
        idx = [order[i] for i in ref_ids]
        stacked.append(probs[idx])
    return np.stack(stacked)


def main() -> None:
    ap = argparse.ArgumentParser(description="Blend runs by averaging softmax probs")
    ap.add_argument("--runs", required=True, help="comma-separated run ids")
    ap.add_argument("--weights", default=None, help="comma-separated weights (default: equal)")
    ap.add_argument("--out", default="blend", help="output run id")
    ap.add_argument("--res", type=int, default=224, help="res for fetching true val labels")
    args = ap.parse_args()

    runs = [r.strip() for r in args.runs.split(",") if r.strip()]
    weights = np.array([float(w) for w in args.weights.split(",")]) if args.weights \
        else np.ones(len(runs))
    weights = weights / weights.sum()
    if len(weights) != len(runs):
        ap.error("number of weights must match number of runs")

    # ---- val: report blended macro F1 ----
    _, ref_val_ids = _load(runs[0], "val")
    val_stack = _align(runs, "val", ref_val_ids)
    blended_val = np.tensordot(weights, val_stack, axes=(0, 0))

    val_df = data.split_frame(args.res, "val")
    id2lbl = dict(zip(val_df["id"].astype(str), val_df["label"].astype(int)))
    y_true = np.array([id2lbl[i] for i in ref_val_ids])

    print("member val macro-F1:")
    for run, w in zip(runs, weights):
        m_probs, m_ids = _load(run, "val")
        order = {i: k for k, i in enumerate(m_ids)}
        m = m_probs[[order[i] for i in ref_val_ids]]
        print(f"  {run:40s} w={w:.3f}  {f1_score(y_true, m.argmax(1), average='macro'):.5f}")
    blended_f1 = f1_score(y_true, blended_val.argmax(1), average="macro")
    print(f"BLEND ({args.out}) val macro-F1 = {blended_f1:.5f}")

    # ---- test: write blended probs for submission ----
    _, ref_test_ids = _load(runs[0], "test")
    test_stack = _align(runs, "test", ref_test_ids)
    blended_test = np.tensordot(weights, test_stack, axes=(0, 0))
    np.save(ARTIFACTS / f"{args.out}_test_probs.npy", blended_test.astype(np.float32))
    np.save(ARTIFACTS / f"{args.out}_test_ids.npy", np.array(ref_test_ids))
    print(f"saved blended test probs -> {args.out}_test_probs.npy  "
          f"(build submission: python -m src.submit --run {args.out})")


if __name__ == "__main__":
    main()
