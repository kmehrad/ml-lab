"""Level-2 stacking over base-model OOF probabilities (Phase 4).

Trains a small MLP meta-learner on the concatenated out-of-fold probability matrices of the base
models (each leak-safe), CV'd with the canonical folds, and applies it to the base test matrices.
A nonlinear meta-model can beat a fixed weighted average (hillclimb) when base models are
complementary. Saves ``{tag}_{oof,test}.npy`` + metrics like the other learners.

    uv run python -m src.stack --models nn_te2_s10 xgb_te2_d4_s3 nn_te2b_s10 --tag stack
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .cv import folds
from .metric import mapk3

ART = Path(__file__).resolve().parent.parent / "experiments" / "artifacts"
N_CLASSES = 7


def _meta_mlp(n_in):
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(n_in, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, N_CLASSES))


def _fit_predict(Xtr, ytr, predict_sets, seed, epochs=120, lr=1e-3, bs=8192):
    """Train one meta-MLP; return softmax predictions for each array in ``predict_sets``."""
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr_t = torch.as_tensor(Xtr, device=dev)
    ytr_t = torch.as_tensor(ytr, device=dev, dtype=torch.long)
    m = _meta_mlp(Xtr.shape[1]).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-5)
    n = len(ytr_t)
    for _ in range(epochs):
        m.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            F.cross_entropy(m(Xtr_t[idx]), ytr_t[idx]).backward()
            opt.step()
    m.eval()
    with torch.no_grad():
        return [torch.softmax(m(torch.as_tensor(X, device=dev)), 1).cpu().numpy()
                for X in predict_sets]


def stack(models, tag="stack", seeds=3, n_splits=5):
    y = np.load(ART / "y.npy")
    Xoof = np.hstack([np.load(ART / f"{m}_oof.npy") for m in models]).astype(np.float32)
    Xtest = np.hstack([np.load(ART / f"{m}_test.npy") for m in models]).astype(np.float32)
    ntest = Xtest.shape[0]
    print(f"stacking {len(models)} models -> {Xoof.shape[1]} meta-features")
    for m in models:
        print(f"  base {m:22s} {mapk3(y, np.load(ART / f'{m}_oof.npy')):.5f}")

    oof = np.zeros((len(y), N_CLASSES))
    test = np.zeros((ntest, N_CLASSES))
    for s in range(seeds):
        seed = 42 + s
        for tr, va in folds(y, n_splits, seed=seed):
            p_va, p_test = _fit_predict(Xoof[tr], y[tr], [Xoof[va], Xtest], seed)
            oof[va] += p_va / seeds
            test += p_test / (seeds * n_splits)
    score = mapk3(y, oof)
    res = {"tag": tag, "models": models, "oof_map3": float(score), "seeds": seeds}
    print(f"\nstack OOF MAP@3 = {score:.5f}")
    np.save(ART / f"{tag}_oof.npy", oof)
    np.save(ART / f"{tag}_test.npy", test)
    (ART / f"{tag}_metrics.json").write_text(json.dumps(res, indent=2))
    print(f"saved -> {tag}_oof.npy / _test.npy")
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--tag", default="stack")
    p.add_argument("--seeds", type=int, default=3)
    a = p.parse_args()
    stack(a.models, tag=a.tag, seeds=a.seeds)
