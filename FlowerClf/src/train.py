"""Fine-tune a timm backbone on the flower images; report **val macro F1**.

Saves to ``experiments/artifacts/``:
    {run}_val_probs.npy / _val_ids.npy   softmax over the provided val split
    {run}_test_probs.npy / _test_ids.npy softmax over test (if converted)
    {run}_metrics.json                   {"val_macro_f1", "per_class_f1", ...}
and the best checkpoint to ``models/{run}.pt``; appends a row to experiments/README.md.

Validation = the competition's provided ``val`` split (macro F1) until merged-K-fold is added.

Usage:
    uv run python -m src.train --model tf_efficientnet_b3_ns --res 224
    uv run python -m src.train --model efficientnet_b0 --sample   # tiny CPU smoke run
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import date
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler

from src import data
from src.dataset import FlowerDataset, build_transforms, class_counts, sample_weights

ARTIFACTS = data.ROOT / "experiments" / "artifacts"
MODELS = data.ROOT / "models"
RUNLOG = data.ROOT / "experiments" / "README.md"


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loader(df, transform, *, batch_size, num_workers, train, balanced):
    ds = FlowerDataset(df, transform)
    sampler = None
    shuffle = train
    if train and balanced:
        sampler = WeightedRandomSampler(sample_weights(df), num_samples=len(df), replacement=True)
        shuffle = False
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
        num_workers=num_workers, pin_memory=True, drop_last=train,
        persistent_workers=num_workers > 0,
    )


@torch.no_grad()
def predict_probs(model, loader, device, use_amp) -> tuple[np.ndarray, np.ndarray, list[str]]:
    model.eval()
    all_probs, all_labels, all_ids = [], [], []
    for x, y, ids in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x)
        all_probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        all_labels.append(np.asarray(y))
        all_ids.extend(ids)
    return np.concatenate(all_probs), np.concatenate(all_labels), all_ids


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune a timm backbone for flower classification")
    ap.add_argument("--model", default="tf_efficientnet_b3_ns")
    ap.add_argument("--res", type=int, default=224, choices=data.RESOLUTIONS)
    ap.add_argument("--run", default=None, help="run id (default: {model}_r{res})")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--freeze-epochs", type=int, default=1, help="head-only warmup epochs")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-epochs", type=int, default=2)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--mixup", type=float, default=0.0, help="mixup alpha (0 = off)")
    ap.add_argument("--balanced-sampler", action="store_true", help="inverse-freq sampling")
    ap.add_argument("--auto-augment", default="rand-m9-mstd0.5", help="timm AA policy or 'none'")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--sample", action="store_true", help="tiny subset + 1 epoch smoke run")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import timm
    from timm.data import resolve_data_config

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run = args.run or f"{args.model}_r{args.res}"
    device = pick_device()
    use_amp = (not args.no_amp) and device.type == "cuda"
    aa = None if args.auto_augment.lower() == "none" else args.auto_augment
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    train_df = data.split_frame(args.res, "train")
    val_df = data.split_frame(args.res, "val")
    if args.sample:
        train_df = train_df.groupby("label", group_keys=False).head(3).reset_index(drop=True)
        val_df = val_df.groupby("label", group_keys=False).head(2).reset_index(drop=True)
        args.epochs, args.freeze_epochs, args.batch_size = 1, 0, 8
        args.num_workers, aa = 0, None
    print(f"run={run} device={device} amp={use_amp} train={len(train_df)} val={len(val_df)}")

    model = timm.create_model(args.model, pretrained=True, num_classes=data.N_CLASSES)
    model.to(device)
    cfg = resolve_data_config({}, model=model)

    train_tf = build_transforms(args.res, cfg, train=True, auto_augment=aa)
    eval_tf = build_transforms(args.res, cfg, train=False, auto_augment=None)
    train_loader = make_loader(train_df, train_tf, batch_size=args.batch_size,
                               num_workers=args.num_workers, train=True,
                               balanced=args.balanced_sampler)
    val_loader = make_loader(val_df, eval_tf, batch_size=args.batch_size,
                             num_workers=args.num_workers, train=False, balanced=False)

    mixup_fn = None
    if args.mixup > 0:
        from timm.data import Mixup
        mixup_fn = Mixup(mixup_alpha=args.mixup, label_smoothing=args.label_smoothing,
                         num_classes=data.N_CLASSES)
        from timm.loss import SoftTargetCrossEntropy
        criterion = SoftTargetCrossEntropy()
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    sched = None
    if not args.sample:
        from timm.scheduler import CosineLRScheduler
        sched = CosineLRScheduler(opt, t_initial=args.epochs, warmup_t=args.warmup_epochs,
                                  warmup_lr_init=args.lr * 0.05, lr_min=args.lr * 0.01)

    def set_backbone_frozen(frozen: bool):
        for p in model.parameters():
            p.requires_grad = not frozen
        for p in model.get_classifier().parameters():
            p.requires_grad = True

    best_f1, best_state = -1.0, None
    t0 = time.time()
    for epoch in range(args.epochs):
        set_backbone_frozen(epoch < args.freeze_epochs)
        model.train()
        running = 0.0
        for x, y, _ in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if mixup_fn is not None:
                x, y = mixup_fn(x, y)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item() * x.size(0)
        if sched is not None:
            sched.step(epoch + 1)

        probs, labels, _ = predict_probs(model, val_loader, device, use_amp)
        f1 = f1_score(labels, probs.argmax(1), average="macro")
        print(f"epoch {epoch+1}/{args.epochs}  loss={running/len(train_loader.dataset):.4f}  "
              f"val_macroF1={f1:.5f}")
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # ---- save best ----
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save({"model": args.model, "res": args.res, "state_dict": model.state_dict(),
                "classes": data.CLASSES}, MODELS / f"{run}.pt")

    val_probs, val_labels, val_ids = predict_probs(model, val_loader, device, use_amp)
    per_class = f1_score(val_labels, val_probs.argmax(1), average=None,
                         labels=list(range(data.N_CLASSES)), zero_division=0)
    np.save(ARTIFACTS / f"{run}_val_probs.npy", val_probs.astype(np.float32))
    np.save(ARTIFACTS / f"{run}_val_ids.npy", np.array(val_ids))

    test_df = data.split_frame(args.res, "test")
    n_test = len(test_df)
    if n_test:
        test_loader = make_loader(test_df, eval_tf, batch_size=args.batch_size,
                                  num_workers=args.num_workers, train=False, balanced=False)
        test_probs, _, test_ids = predict_probs(model, test_loader, device, use_amp)
        np.save(ARTIFACTS / f"{run}_test_probs.npy", test_probs.astype(np.float32))
        np.save(ARTIFACTS / f"{run}_test_ids.npy", np.array(test_ids))

    elapsed = time.time() - t0
    metrics = {
        "run": run, "model": args.model, "res": args.res,
        "val_macro_f1": float(best_f1),
        "per_class_f1": [round(float(x), 4) for x in per_class],
        "n_train": len(train_df), "n_val": len(val_df), "n_test": n_test,
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
        "mixup": args.mixup, "balanced_sampler": args.balanced_sampler,
        "elapsed_sec": round(elapsed, 1),
    }
    (ARTIFACTS / f"{run}_metrics.json").write_text(json.dumps(metrics, indent=2))

    _append_runlog(run, args, best_f1)
    print(f"\nDONE run={run}  best val_macroF1={best_f1:.5f}  ({elapsed:.0f}s)")
    print(f"  metrics -> {ARTIFACTS / f'{run}_metrics.json'}")


def _append_runlog(run: str, args, f1: float) -> None:
    if not RUNLOG.exists():
        return
    row = (f"| {run} | {date.today().isoformat()} | {args.model} | {args.res} "
           f"| {f1:.5f} |  | {args.epochs} | mixup={args.mixup} bal={args.balanced_sampler} |\n")
    with RUNLOG.open("a") as fh:
        fh.write(row)


if __name__ == "__main__":
    main()
