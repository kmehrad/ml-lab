"""PyTorch MLP tabular learner — the diverse NN base learner for the ensemble.

GBDT tuning (depth/lr grids, CatBoost tree count, the `ratios` feature group) all landed inside
fold noise (see `reports/RESULTS.md`), so the remaining lever is genuine model diversity for the
blend rather than more tree tuning. NNs fit a smooth global function instead of axis-aligned
splits, so their errors decorrelate from the GBDTs even if their raw accuracy is similar or a
touch worse. This is a self-contained tabular MLP: numeric features are standardized per-fold,
categoricals are one-hot encoded (no missing-value handling needed — this dataset has zero NaNs,
unlike the sibling `StudentHealthRisk`). Trains with MSE loss, per-fold early stopping on
validation RMSE, and optional seed-averaging. Runs on CUDA if available, else CPU. Saves
OOF/test/metrics in the exact format of `train.py` so `blend`/`submit` consume it by name.

Intended to run on the remote GPU box (torch is the `gpu` extra):
    uv run python -m src.models_nn --seeds 5 --tag nn_s5
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from .cv import folds
from .features import add_features, categorical_columns, feature_columns
from .metric import rmse

ART = Path(__file__).resolve().parent.parent / "experiments" / "artifacts"


def _design_matrix(train_df: pd.DataFrame, test_df: pd.DataFrame | None, groups):
    """Build aligned dense float32 design matrices: numerics passthrough, categoricals one-hot."""
    feats = feature_columns(groups)
    cat = categorical_columns(groups)
    num = [c for c in feats if c not in cat]

    frames = [train_df] + ([test_df] if test_df is not None else [])
    for c in cat:  # shared categories so get_dummies yields identical columns train vs test
        levels = pd.unique(pd.concat([f[c].astype("object") for f in frames], ignore_index=True))
        for f in frames:
            f[c] = f[c].astype("object").astype(pd.CategoricalDtype(categories=levels))

    def build(df):
        X_num = df[num].astype("float32")
        X_cat = pd.get_dummies(df[cat], dtype="float32") if cat else pd.DataFrame(index=df.index)
        return pd.concat([X_num, X_cat], axis=1)

    Xtr = build(train_df)
    Xte = build(test_df) if test_df is not None else None
    if Xte is not None:  # align one-hot columns (identical by construction, but be defensive)
        Xte = Xte.reindex(columns=Xtr.columns, fill_value=0.0)
    return Xtr.to_numpy(np.float32), (Xte.to_numpy(np.float32) if Xte is not None else None)


def _mlp(n_in: int, hidden=(256, 128, 64), p: float = 0.2):
    import torch.nn as nn
    layers, d = [], n_in
    for h in hidden:
        layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p)]
        d = h
    layers.append(nn.Linear(d, 1))
    return nn.Sequential(*layers)


def _train_fold(Xtr, ytr, Xva, yva, Xte, seed, device,
                epochs=300, bs=4096, lr=1e-3, patience=20, dropout=0.2):
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed)

    mu = Xtr.mean(axis=0, keepdims=True)
    sd = Xtr.std(axis=0, keepdims=True) + 1e-6
    to = lambda A: torch.as_tensor((A - mu) / sd, dtype=torch.float32, device=device)
    Xtr_t, Xva_t = to(Xtr), to(Xva)
    ytr_t = torch.as_tensor(ytr, dtype=torch.float32, device=device).unsqueeze(1)
    yva_t = torch.as_tensor(yva, dtype=torch.float32, device=device).unsqueeze(1)

    model = _mlp(Xtr.shape[1], p=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    n = Xtr_t.shape[0]
    best_rmse, best_state, bad = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = F.mse_loss(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            va_rmse = F.mse_loss(model(Xva_t), yva_t).sqrt().item()
        if va_rmse < best_rmse - 1e-5:
            best_rmse, best_state, bad = va_rmse, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        va = model(Xva_t).cpu().numpy().ravel()
        te = model(to(Xte)).cpu().numpy().ravel() if Xte is not None else None
    return va, te, best_rmse


def run_nn(sample: int | None = None, n_splits: int = 5, groups=("base",), tag: str = "nn",
           seeds: int = 1, seed_base: int = 42) -> dict:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    groups = tuple(groups)
    df = add_features(D.load_train(), groups)
    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=42).reset_index(drop=True)
    test = add_features(D.load_test(), groups) if not sample else None
    print(f"NN device={device}  train rows={len(df):,}  features={list(groups)}  tag={tag}")

    y = df[D.TARGET].to_numpy(dtype=float)
    X, Xte_full = _design_matrix(df, test, groups)
    test_id = test[D.ID].to_numpy() if test is not None else None

    seed_list = [seed_base + s for s in range(max(1, seeds))]
    n_avg = len(seed_list)
    oof = np.zeros(len(df))
    test_pred = np.zeros(len(Xte_full)) if Xte_full is not None else None
    fold_scores = []
    t0 = time.time()
    for seed in seed_list:
        for k, (tr, va) in enumerate(folds(y, n_splits, seed=seed)):
            va_p, te_p, bi = _train_fold(X[tr], y[tr], X[va], y[va], Xte_full, seed, device)
            oof[va] += va_p / n_avg
            if test_pred is not None:
                test_pred += te_p / (n_splits * n_avg)
            s = rmse(y[va], va_p)
            fold_scores.append(s)
            print(f"  seed {seed} fold {k}: RMSE={s:.5f}")

    oof_rmse = rmse(y, oof)
    res = {
        "model": "nn", "features": list(groups), "tag": tag, "oof_rmse": float(oof_rmse),
        "fold_mean": float(np.mean(fold_scores)), "fold_std": float(np.std(fold_scores)),
        "fold_scores": [float(s) for s in fold_scores], "n_rows": int(len(df)),
        "seeds": n_avg, "elapsed_s": round(time.time() - t0, 1),
    }
    print(f"\nOOF RMSE = {oof_rmse:.5f}   ({res['elapsed_s']}s)")

    if not sample:
        ART.mkdir(parents=True, exist_ok=True)
        suffix = f"_{tag}" if tag else ""
        np.save(ART / f"nn{suffix}_oof.npy", oof)
        np.save(ART / f"nn{suffix}_test.npy", test_pred)
        np.save(ART / "y.npy", y)
        np.save(ART / "test_id.npy", test_id)
        (ART / f"nn{suffix}_metrics.json").write_text(json.dumps(res, indent=2))
        print(f"saved -> {ART}/nn{suffix}_oof.npy, nn{suffix}_test.npy, nn{suffix}_metrics.json")
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=None, help="limit to N rows (smoke test; no test preds)")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--features", nargs="+", default=["base"], help="feature groups: base ratios")
    p.add_argument("--tag", default="nn", help="artifact filename suffix")
    p.add_argument("--seeds", type=int, default=1, help="seed-average over N seeds")
    p.add_argument("--seed-base", type=int, default=42)
    a = p.parse_args()
    run_nn(a.sample, a.folds, groups=a.features, tag=a.tag, seeds=a.seeds, seed_base=a.seed_base)
