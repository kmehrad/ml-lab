"""GPU MLP ensemble member — produces OOF/test probabilities in the same .npy format as the GBDTs.

Runs on the remote RTX 3090 Ti box (the dev Mac has no CUDA). It reuses the **same** StratifiedKFold
folds (`src.cv.get_folds`, deterministic given y + seed) so its OOF rows align with the GBDT OOF and
the arrays can be blended directly.

On the GPU box (after syncing this repo + data/raw and installing torch):
    uv run python -m src.dl.train_nn
    # writes experiments/artifacts/nn_oof.npy, nn_test.npy  -> scp back to the Mac

Categoricals are one-hot encoded; numerics are standardized with train-fold statistics (no leakage).
Training uses class-weighted cross-entropy and early-stops on validation balanced accuracy.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from .. import cv
from .. import data as D
from .. import features as F
from .. import metrics as M

ART = Path(__file__).resolve().parent.parent.parent / "experiments" / "artifacts"
N_CLASS = len(D.CLASSES)

EPOCHS = 60
BATCH = 4096
LR = 1e-3
PATIENCE = 8
HIDDEN = (256, 128, 64)
DROPOUT = 0.2


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_matrix(Xtr: pd.DataFrame, Xte: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """One-hot the categoricals consistently across train/test; return numeric+dummy matrices."""
    both = pd.concat([Xtr, Xte], axis=0)
    dummies = pd.get_dummies(both[F.CATEGORICAL_FEATURES].astype(str), dtype="float32")
    num = both[F.NUMERIC_FEATURES].astype("float32")
    full = pd.concat([num, dummies], axis=1)
    ntr = len(Xtr)
    return full.iloc[:ntr].to_numpy(), full.iloc[ntr:].to_numpy()


class MLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for h in HIDDEN:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(DROPOUT)]
            d = h
        layers.append(nn.Linear(d, N_CLASS))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _train_fold(Xtr, ytr, Xva, yva, class_w, device) -> tuple[np.ndarray, MLP, StandardScaler]:
    scaler = StandardScaler().fit(Xtr)
    Xtr = scaler.transform(Xtr).astype("float32")
    Xva = scaler.transform(Xva).astype("float32")

    model = MLP(Xtr.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    lossf = nn.CrossEntropyLoss(weight=torch.tensor(class_w, dtype=torch.float32, device=device))

    Xtr_t = torch.tensor(Xtr, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=device)
    Xva_t = torch.tensor(Xva, device=device)

    best_bacc, best_state, best_proba, wait = -1.0, None, None, 0
    n = len(Xtr_t)
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, BATCH):
            idx = perm[i : i + BATCH]
            opt.zero_grad()
            loss = lossf(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            proba = torch.softmax(model(Xva_t), dim=1).cpu().numpy()
        bacc = balanced_accuracy_score(yva, proba.argmax(1))
        sched.step(bacc)
        if bacc > best_bacc:
            best_bacc, best_proba, wait = bacc, proba, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return best_proba, model, scaler


def train() -> dict:
    device = pick_device()
    print(f"device: {device}")
    tr = D.load_train()
    te = D.load_test()
    y = D.encode_target(tr[D.TARGET])
    Xtr_all, Xte = build_matrix(F.add_features(tr), F.add_features(te))

    counts = np.bincount(y, minlength=N_CLASS)
    class_w = (len(y) / (N_CLASS * counts)).astype("float32")

    folds = cv.get_folds(y)
    oof = np.zeros((len(y), N_CLASS), dtype="float32")
    test_pred = np.zeros((len(Xte), N_CLASS), dtype="float32")

    for k, (tr_idx, va_idx) in enumerate(folds):
        proba, model, scaler = _train_fold(
            Xtr_all[tr_idx], y[tr_idx], Xtr_all[va_idx], y[va_idx], class_w, device
        )
        oof[va_idx] = proba
        model.eval()
        with torch.no_grad():
            Xte_s = torch.tensor(scaler.transform(Xte).astype("float32"), device=device)
            test_pred += torch.softmax(model(Xte_s), dim=1).cpu().numpy() / len(folds)
        print(f"  [nn] fold {k}: bal-acc(raw)={balanced_accuracy_score(y[va_idx], proba.argmax(1)):.5f}")

    raw = M.balanced_accuracy(y, oof)
    mult, corrected = M.tune_multipliers(y, oof)
    print(f"[nn] OOF bal-acc raw={raw:.5f}  corrected={corrected:.5f}  mult={np.round(mult,3).tolist()}")

    ART.mkdir(parents=True, exist_ok=True)
    np.save(ART / "nn_oof.npy", oof)
    np.save(ART / "nn_test.npy", test_pred)
    # keep shared arrays in sync if the GBDTs were not run on this box
    if not (ART / "y_true.npy").exists():
        np.save(ART / "y_true.npy", y)
        np.save(ART / "test_id.npy", te[D.ID_COL].to_numpy())
    result = {"model": "nn", "oof_bacc_raw": raw, "oof_bacc_corrected": corrected, "mult": mult.tolist()}
    json.dump(result, open(ART / "nn_metrics.json", "w"), indent=2)
    return result


if __name__ == "__main__":
    train()
