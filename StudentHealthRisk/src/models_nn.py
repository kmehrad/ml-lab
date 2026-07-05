"""PyTorch MLP tabular learner — the diverse NN base learner for the ensemble.

NNs extract distributed weak signal differently from GBDTs, so they add blend diversity. This is a
self-contained tabular MLP: numeric features are median-imputed + standardized (with per-column
missing-indicator inputs), categoricals are one-hot encoded (NaN is its own column). Trains with
class-weighted cross-entropy (aligning with the balanced-accuracy metric), per-fold early stopping on
validation logloss, and seed-averaging. Runs on CUDA if available, else CPU. Saves OOF/test/metrics in
the exact format of ``train.py`` so ``blend``/``hillclimb``/``submit`` consume it by name.

Intended to run on the remote GPU box (torch is the ``gpu`` extra):
    uv run python -m src.models_nn --seeds 10 --tag nn_s10
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
from .features import add_features, feature_columns
from .metric import score_proba, tune_weights

ART = Path(__file__).resolve().parent.parent / "experiments" / "artifacts"
N_CLASSES = D.N_CLASSES


def _design_matrix(train_df: pd.DataFrame, test_df: pd.DataFrame | None, groups):
    """Build aligned dense float32 design matrices for train (+ optional test).

    Numeric cols: median-imputed (+ an ``_na`` indicator). Categoricals: one-hot via a shared level
    union so train/test columns line up. Standardization is done per-fold in the trainer.
    """
    feats = feature_columns(groups)
    num = [c for c in feats if c not in D.CATEGORICAL]
    cat = [c for c in feats if c in D.CATEGORICAL]

    frames = [train_df] + ([test_df] if test_df is not None else [])
    for c in cat:  # shared categories so get_dummies yields identical columns
        levels = pd.unique(pd.concat([f[c].astype("object") for f in frames], ignore_index=True))
        levels = [lv for lv in levels if pd.notna(lv)]   # NaN handled by get_dummies(dummy_na=True)
        for f in frames:
            f[c] = f[c].astype("object").astype(pd.CategoricalDtype(categories=levels))

    med = train_df[num].median()

    def build(df):
        X_num = df[num].astype("float32")
        na = X_num.isna().astype("float32")
        na.columns = [f"{c}_na" for c in num]
        X_num = X_num.fillna(med)
        X_cat = pd.get_dummies(df[cat], dummy_na=True, dtype="float32") if cat else pd.DataFrame(index=df.index)
        return pd.concat([X_num, na, X_cat], axis=1)

    Xtr = build(train_df)
    Xte = build(test_df) if test_df is not None else None
    if Xte is not None:  # align one-hot columns (identical by construction, but be defensive)
        Xte = Xte.reindex(columns=Xtr.columns, fill_value=0.0)
    return Xtr.to_numpy(np.float32), (Xte.to_numpy(np.float32) if Xte is not None else None)


def _mlp(n_in: int, hidden=(512, 256, 128), p: float = 0.3):
    import torch.nn as nn
    layers, d = [], n_in
    for h in hidden:
        layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p)]
        d = h
    layers.append(nn.Linear(d, N_CLASSES))
    return nn.Sequential(*layers)


def _train_fold(Xtr, ytr, Xva, yva, Xte, class_w, seed, device,
                epochs=100, bs=4096, lr=1e-3, patience=12, dropout=0.3):
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed)

    mu = Xtr.mean(axis=0, keepdims=True)
    sd = Xtr.std(axis=0, keepdims=True) + 1e-6
    to = lambda A: torch.as_tensor((A - mu) / sd, dtype=torch.float32, device=device)
    Xtr_t, Xva_t = to(Xtr), to(Xva)
    ytr_t = torch.as_tensor(ytr, dtype=torch.long, device=device)
    yva_t = torch.as_tensor(yva, dtype=torch.long, device=device)
    w_t = torch.as_tensor(class_w, dtype=torch.float32, device=device)

    model = _mlp(Xtr.shape[1], p=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    n = Xtr_t.shape[0]
    best_ll, best_state, bad = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = F.cross_entropy(model(Xtr_t[idx]), ytr_t[idx], weight=w_t)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            ll = F.cross_entropy(model(Xva_t), yva_t, weight=w_t).item()
        if ll < best_ll - 1e-5:
            best_ll, best_state, bad = ll, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        va = torch.softmax(model(Xva_t), 1).cpu().numpy()
        te = torch.softmax(model(to(Xte)), 1).cpu().numpy() if Xte is not None else None
    return va, te


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

    y = D.encode_target(df[D.TARGET])
    X, Xte_full = _design_matrix(df, test, groups)
    counts = np.bincount(y, minlength=N_CLASSES).astype(np.float64)
    class_w = (len(y) / (N_CLASSES * counts)).astype(np.float32)  # balanced class weights

    test_id = test[D.ID].to_numpy() if test is not None else None
    seed_list = [seed_base + s for s in range(max(1, seeds))]
    n_avg = len(seed_list)
    oof = np.zeros((len(df), N_CLASSES))
    test_proba = np.zeros((len(Xte_full), N_CLASSES)) if Xte_full is not None else None
    fold_scores = []
    t0 = time.time()
    for seed in seed_list:
        for k, (tr, va) in enumerate(folds(y, n_splits, seed=seed)):
            va_p, te_p = _train_fold(X[tr], y[tr], X[va], y[va], Xte_full, class_w, seed, device)
            oof[va] += va_p / n_avg
            if test_proba is not None:
                test_proba += te_p / (n_splits * n_avg)
            s = score_proba(y[va], va_p)
            fold_scores.append(s)
            print(f"  seed {seed} fold {k}: bal_acc(raw)={s:.5f}")

    raw = score_proba(y, oof)
    w = tune_weights(y, oof)
    tuned = score_proba(y, oof, w)
    res = {"model": "nn", "features": list(groups), "tag": tag,
           "oof_bacc_raw": float(raw), "oof_bacc_tuned": float(tuned),
           "decision_weights": [float(x) for x in w],
           "fold_mean_raw": float(np.mean(fold_scores)), "fold_std_raw": float(np.std(fold_scores)),
           "n_rows": int(len(df)), "seeds": n_avg, "elapsed_s": round(time.time() - t0, 1)}
    print(f"\nOOF balanced accuracy: raw {raw:.5f}   tuned {tuned:.5f}   ({res['elapsed_s']}s)")

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
    p.add_argument("--features", nargs="+", default=["base"])
    p.add_argument("--tag", default="nn")
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--seed-base", type=int, default=42)
    a = p.parse_args()
    run_nn(a.sample, a.folds, groups=a.features, tag=a.tag, seeds=a.seeds, seed_base=a.seed_base)
