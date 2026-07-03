"""Mel-spectrogram transfer CNN (multi-label) → OOF/test probs.

A ``timm`` backbone (single-channel input, 87-way sigmoid head) fine-tuned with
BCE on log-mel spectrograms, using the shared 5-fold split. Device-portable
(``cuda→mps→cpu``), AMP on CUDA. Given ≤20 positives/class this is an
ensemble-diversity member, not expected to beat the classical RF outright.

Artifacts follow the same contract as ``train_clf`` (blendable):
    experiments/artifacts/{run}_oof.npy  (687, 87)
    experiments/artifacts/{run}_test.npy (1000, 87)
    experiments/artifacts/{run}_metrics.json

Usage:
    uv run python -m src.train_cnn --model efficientnet_b0 --epochs 20        # GPU box
    uv run python -m src.train_cnn --sample                                   # CPU smoke
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import date

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src import audio, cv, data
from src.dataset import MelDataset
from src.train_clf import ART, RUN_LOG, pooled_auc, per_class_auc


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _make_model(name: str):
    import timm

    return timm.create_model(name, pretrained=True, in_chans=1, num_classes=data.N_CLASSES)


def _train_fold(name, Xtr, Ytr, Xva, Xte, device, epochs, bs, lr, augment, seed):
    torch.manual_seed(seed)
    model = _make_model(name).to(device)
    pos = Ytr.sum(0)
    pos_weight = torch.tensor((len(Ytr) - pos) / np.clip(pos, 1, None), dtype=torch.float32, device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    tr_dl = DataLoader(MelDataset(Xtr, Ytr, augment=augment), batch_size=bs, shuffle=True, drop_last=False)
    model.train()
    for _ in range(epochs):
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            with torch.autocast(device_type="cuda", enabled=amp):
                loss = crit(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()

    @torch.no_grad()
    def predict(X):
        model.eval()
        dl = DataLoader(MelDataset(X), batch_size=bs, shuffle=False)
        out = [torch.sigmoid(model(xb.to(device))).cpu().numpy() for xb in dl]
        return np.concatenate(out)

    return predict(Xva), predict(Xte)


def run(name="efficientnet_b0", epochs=20, bs=16, lr=3e-4, seed=42, sample=False) -> dict:
    Xmel = audio.mel_cache("train")
    Xte = audio.mel_cache("test")
    Y = data.train_targets().astype(np.float32)
    device = pick_device()

    if sample:
        epochs, name = 1, "efficientnet_b0"
        keep = np.arange(60)
        Xmel, Y = Xmel[keep], Y[keep]
        Xte = Xte[:20]

    folds = cv.make_folds(Y) if not sample else [(np.arange(40), np.arange(40, 60))]
    oof = np.zeros((len(Xmel), data.N_CLASSES), dtype=float)
    test = np.zeros((len(Xte), data.N_CLASSES), dtype=float)

    t0 = time.time()
    for tr, va in folds:
        vp, tp = _train_fold(name, Xmel[tr], Y[tr], Xmel[va], Xte, device, epochs, bs, lr,
                             augment=not sample, seed=seed)
        oof[va] = vp
        test += tp / len(folds)
    secs = time.time() - t0

    metrics = {
        "model": f"cnn-{name}", "features": "logmel", "n_features": audio.N_MELS * audio.T_FRAMES,
        "date": str(date.today()), "seconds": round(secs, 1), "device": device,
        "epochs": epochs, "n_classes": data.N_CLASSES,
        "pooled_auc": pooled_auc(Y, oof),
        "macro_auc": float(np.nanmean(per_class_auc(Y, oof))),
        "sample": sample,
    }

    if not sample:
        ART.mkdir(parents=True, exist_ok=True)
        np.save(ART / "cnn_oof.npy", oof)
        np.save(ART / "cnn_test.npy", test)
        (ART / "cnn_metrics.json").write_text(json.dumps(metrics, indent=2))
        with open(RUN_LOG, "a") as f:
            f.write(f"| cnn | {metrics['date']} | cnn-{name} | logmel | "
                    f"{metrics['pooled_auc']:.5f} |  |  | macro={metrics['macro_auc']:.4f}, "
                    f"{epochs}ep/{device}, {secs:.0f}s |\n")

    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="efficientnet_b0")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample", action="store_true", help="tiny CPU smoke run")
    args = ap.parse_args()
    run(args.model, epochs=args.epochs, bs=args.batch_size, lr=args.lr, seed=args.seed, sample=args.sample)


if __name__ == "__main__":
    main()
