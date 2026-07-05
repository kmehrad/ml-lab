"""PyTorch MLP on cuML combination-TE features — the diverse NN learner for the ensemble.

NNs extract distributed weak signal differently from GBDTs, so they add ensemble diversity (the
1st-place solution leaned heavily on NNs). Trains a standardized-input MLP with cross-entropy,
per-fold early stopping on validation logloss, and seed-averaging. Saves OOF/test/metrics in the
exact format of ``train.py`` (``{model}{suffix}_oof.npy`` etc.) so ``blend``/``hillclimb``/``submit``
consume it by name.

    uv run python -m src.models_nn --te-order 2 --seeds 3 --tag nn_te2
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import data as D
from .cv import folds
from .metric import mapk3
from .gpu_te import build_te_features, combos, COMBO_FEATURES

ART = Path(__file__).resolve().parent.parent / "experiments" / "artifacts"
N_CLASSES = D.N_CLASSES


def _mlp(n_in: int, hidden=(512, 256, 128), p: float = 0.3):
    import torch.nn as nn
    layers, d = [], n_in
    for h in hidden:
        layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p)]
        d = h
    layers.append(nn.Linear(d, N_CLASSES))
    return nn.Sequential(*layers)


def _to_torch(cp_arr):
    """Zero-copy cupy -> torch cuda tensor via DLPack."""
    import torch
    return torch.from_dlpack(cp_arr)


def _train_fold(Xtr, ytr, Xva, yva, Xte, seed, epochs=80, bs=4096, lr=1e-3,
                patience=10, hidden=(512, 256, 128), dropout=0.3):
    """Train one MLP fold on cupy feature arrays; return (va_proba, te_proba, epochs_run) as numpy."""
    import cupy as cp
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed)

    mu = Xtr.mean(axis=0, keepdims=True)
    sd = Xtr.std(axis=0, keepdims=True) + 1e-6

    def prep(X):
        return _to_torch(((X - mu) / sd).astype(cp.float32)).float()

    Xtr_t, Xva_t = prep(Xtr), prep(Xva)
    ytr_t = torch.as_tensor(np.asarray(ytr), device="cuda", dtype=torch.long)
    yva_t = torch.as_tensor(np.asarray(yva), device="cuda", dtype=torch.long)

    model = _mlp(Xtr.shape[1], hidden, dropout).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    lossf = torch.nn.CrossEntropyLoss()
    n = Xtr_t.shape[0]
    best_ll, best_state, bad, ran = float("inf"), None, 0, 0
    for ep in range(epochs):
        ran = ep + 1
        model.train()
        perm = torch.randperm(n, device="cuda")
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            ll = F.cross_entropy(model(Xva_t), yva_t).item()
        if ll < best_ll - 1e-5:
            best_ll, best_state, bad = ll, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        va_proba = torch.softmax(model(Xva_t), 1).cpu().numpy()
        te_proba = None
        if Xte is not None:
            te_proba = torch.softmax(model(prep(Xte)), 1).cpu().numpy()
    return va_proba, te_proba, ran


def run_cv(sample=None, n_splits=5, tag="", te_order=2, te_m=0.0, te_original=True,
           seeds=1, seed_base=42, epochs=80, hidden=(512, 256, 128), dropout=0.3,
           lr=1e-3) -> dict:
    import cupy as cp
    df = D.load_train()
    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=42).reset_index(drop=True)
    y = D.encode_target(df[D.TARGET])
    combo_list = combos(te_order)
    base_te = df[COMBO_FEATURES]
    num = cp.asarray(df[list(D.NUMERIC)].to_numpy(np.float32))

    test = Xte_raw = test_id = test_proba = None
    if not sample:
        test = D.load_test()
        test_id = test[D.ID].to_numpy()
        Xte_raw = test[COMBO_FEATURES]
        num_test = cp.asarray(test[list(D.NUMERIC)].to_numpy(np.float32))
        test_proba = np.zeros((len(test), N_CLASSES))
    print(f"NN  rows={len(df):,}  TE order<={te_order} ({len(combo_list)} combos)  seeds={seeds}")

    seed_list = [seed_base + s for s in range(max(1, seeds))]
    n_avg = len(seed_list)
    oof = np.zeros((len(df), N_CLASSES))
    fold_scores, seed_scores = [], []
    t0 = time.time()
    for seed in seed_list:
        oof_s = np.zeros((len(df), N_CLASSES))
        for k, (tr, va) in enumerate(folds(y, n_splits, seed=seed)):
            Xtr_te, Xva_te, Xte_te, _ = build_te_features(
                base_te.iloc[tr], y[tr], base_te.iloc[va], Xte_raw, combo_list, te_m,
                use_original=te_original)
            Xtr = cp.hstack([num[tr], Xtr_te])
            Xva = cp.hstack([num[va], Xva_te])
            Xte = cp.hstack([num_test, Xte_te]) if Xte_te is not None else None
            va_proba, te_proba, ep = _train_fold(Xtr, y[tr], Xva, y[va], Xte, seed, epochs=epochs,
                                                 lr=lr, hidden=hidden, dropout=dropout)
            oof_s[va] = va_proba
            if test_proba is not None:
                test_proba += te_proba / (n_splits * n_avg)
            score = mapk3(y[va], va_proba)
            fold_scores.append(score)
            pre = f"seed {seed} " if n_avg > 1 else ""
            print(f"  {pre}fold {k}: MAP@3={score:.5f}  (epochs={ep})")
        oof += oof_s / n_avg
        if n_avg > 1:
            seed_scores.append(float(mapk3(y, oof_s)))
            print(f"  seed {seed} OOF MAP@3={seed_scores[-1]:.5f}")

    oof_map3 = mapk3(y, oof)
    res = {"model": "nn", "tag": tag, "oof_map3": float(oof_map3),
           "fold_mean": float(np.mean(fold_scores)), "fold_std": float(np.std(fold_scores)),
           "n_rows": int(len(df)), "te_order": te_order, "seeds": n_avg,
           "seed_scores": seed_scores, "elapsed_s": round(time.time() - t0, 1)}
    print(f"\nOOF MAP@3 = {oof_map3:.5f}   folds {res['fold_mean']:.5f} +/- {res['fold_std']:.5f}"
          f"   ({res['elapsed_s']}s)")

    if not sample:
        ART.mkdir(parents=True, exist_ok=True)
        suffix = f"_{tag}" if tag else ""
        np.save(ART / f"nn{suffix}_oof.npy", oof)
        np.save(ART / f"nn{suffix}_test.npy", test_proba)
        np.save(ART / "y.npy", y)
        np.save(ART / "classes.npy", np.array(D.CLASSES, dtype=object))
        np.save(ART / "test_id.npy", test_id)
        (ART / f"nn{suffix}_metrics.json").write_text(json.dumps(res, indent=2))
        print(f"saved -> nn{suffix}_oof.npy / _test.npy / _metrics.json")
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=None)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--tag", default="")
    p.add_argument("--te-order", type=int, default=2)
    p.add_argument("--te-m", type=float, default=0.0)
    p.add_argument("--no-te-original", action="store_true")
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--seed-base", type=int, default=42)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--hidden", default="512,256,128", help="comma-separated MLP hidden sizes")
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--lr", type=float, default=1e-3)
    a = p.parse_args()
    run_cv(a.sample, a.folds, tag=a.tag, te_order=a.te_order, te_m=a.te_m,
           te_original=not a.no_te_original, seeds=a.seeds, seed_base=a.seed_base, epochs=a.epochs,
           hidden=tuple(int(h) for h in a.hidden.split(",")), dropout=a.dropout, lr=a.lr)
