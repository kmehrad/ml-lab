"""Run a saved checkpoint over the test split → softmax probs (optional hflip TTA).

Writes ``experiments/artifacts/{run}_test_probs.npy`` and ``{run}_test_ids.npy``.

Usage:
    uv run python -m src.predict --run tf_efficientnet_b3_ns_r224 --tta
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from src import data
from src.dataset import FlowerDataset, build_transforms
from src.train import ARTIFACTS, MODELS, make_loader, pick_device


@torch.no_grad()
def _infer(model, loader, device, use_amp, tta):
    model.eval()
    probs, ids = [], []
    for x, _, batch_ids in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x)
            if tta:
                logits = logits + model(torch.flip(x, dims=[3]))
        probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        ids.extend(batch_ids)
    return np.concatenate(probs), ids


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict test softmax probs from a checkpoint")
    ap.add_argument("--run", required=True, help="run id (loads models/{run}.pt)")
    ap.add_argument("--tta", action="store_true", help="add horizontal-flip TTA")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args()

    import timm
    from timm.data import resolve_data_config

    ckpt = torch.load(MODELS / f"{args.run}.pt", map_location="cpu")
    device = pick_device()
    use_amp = (not args.no_amp) and device.type == "cuda"

    model = timm.create_model(ckpt["model"], pretrained=False, num_classes=data.N_CLASSES)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    cfg = resolve_data_config({}, model=model)

    res = ckpt["res"]
    eval_tf = build_transforms(res, cfg, train=False, auto_augment=None)
    test_df = data.split_frame(res, "test")
    if test_df.empty:
        raise SystemExit(f"No test rows in metadata_{res}.parquet — convert the test split first.")
    loader = make_loader(test_df, eval_tf, batch_size=args.batch_size,
                         num_workers=args.num_workers, train=False, balanced=False)

    probs, ids = _infer(model, loader, device, use_amp, args.tta)
    np.save(ARTIFACTS / f"{args.run}_test_probs.npy", probs.astype(np.float32))
    np.save(ARTIFACTS / f"{args.run}_test_ids.npy", np.array(ids))
    print(f"saved {probs.shape} test probs (tta={args.tta}) for run={args.run}")


if __name__ == "__main__":
    main()
