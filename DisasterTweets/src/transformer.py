"""Fine-tune a transformer (RoBERTa / DistilBERT) with shared-fold OOF + test predictions.

Usage:
    uv run python -m src.transformer --model roberta              # roberta-base
    uv run python -m src.transformer --model distilbert           # light / CPU fallback
    uv run python -m src.transformer --model roberta --epochs 3 --batch 16 --folds 5

Device auto-detects cuda -> mps -> cpu. Uses the SAME StratifiedKFold splits as
``src.train`` (via ``src.cv``) so OOF/test arrays line up for blending. Saves
``experiments/artifacts/{key}_oof.npy``, ``{key}_test.npy`` and ``{key}_metrics.json``
with the OOF-tuned F1 threshold. A plain PyTorch loop (no Trainer) keeps it robust
across transformers versions.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from src import cv, data, features
from src.train import tune_threshold  # reuse F1 threshold search

ART = Path(__file__).resolve().parent.parent / "experiments" / "artifacts"
ART.mkdir(parents=True, exist_ok=True)

MODELS = {
    "roberta": "roberta-base",
    "distilbert": "distilbert-base-uncased",
    "bertweet": "vinai/bertweet-base",
}


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class TweetDS(Dataset):
    def __init__(self, enc, labels=None):
        self.enc = enc
        self.labels = labels

    def __len__(self):
        return self.enc["input_ids"].shape[0]

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[i], dtype=torch.long)
        return item


def _encode(tokenizer, texts, max_len):
    return tokenizer(list(texts), truncation=True, padding="max_length",
                     max_length=max_len, return_tensors="pt")


@torch.no_grad()
def _predict(model, loader, device) -> np.ndarray:
    model.eval()
    probs = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")):
            logits = model(**batch).logits.float()
        probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probs)


def run_cv(model_key: str, epochs: int, batch: int, lr: float, max_len: int,
           folds: int, seed: int, sample: int | None) -> dict:
    name = MODELS.get(model_key, model_key)
    key = model_key
    device = pick_device()
    print(f"model={name} device={device}")

    tr = data.load_train()
    te = data.load_test()
    if sample:
        tr = tr.sample(n=min(sample, len(tr)), random_state=seed).reset_index(drop=True)

    X = features.build_field(tr, light=True).tolist()
    X_test = features.build_field(te, light=True).tolist()
    y = tr[data.TARGET].to_numpy()

    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
    test_enc = _encode(tokenizer, X_test, max_len)
    test_loader = DataLoader(TweetDS(test_enc), batch_size=batch * 2)

    splits = cv.make_folds(y, n_splits=folds, seed=seed)
    oof = np.zeros(len(y))
    test_pred = np.zeros(len(te))
    fold_f1s = []
    t0 = time.time()

    for f, (tr_idx, va_idx) in enumerate(splits):
        torch.manual_seed(seed + f)
        model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2).to(device)

        tr_enc = _encode(tokenizer, [X[i] for i in tr_idx], max_len)
        va_enc = _encode(tokenizer, [X[i] for i in va_idx], max_len)
        tr_loader = DataLoader(TweetDS(tr_enc, y[tr_idx]), batch_size=batch, shuffle=True)
        va_loader = DataLoader(TweetDS(va_enc), batch_size=batch * 2)

        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        total_steps = len(tr_loader) * epochs
        sched = get_linear_schedule_with_warmup(opt, int(0.06 * total_steps), total_steps)
        scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

        model.train()
        for ep in range(epochs):
            for batch_data in tr_loader:
                batch_data = {k: v.to(device) for k, v in batch_data.items()}
                opt.zero_grad()
                with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")):
                    loss = model(**batch_data).loss
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()

        va_scores = _predict(model, va_loader, device)
        oof[va_idx] = va_scores
        test_pred += _predict(model, test_loader, device) / len(splits)
        from sklearn.metrics import f1_score
        fold_f1s.append(f1_score(y[va_idx], (va_scores >= 0.5).astype(int)))
        print(f"  fold {f}: F1@0.5 = {fold_f1s[-1]:.4f}  ({time.time()-t0:.0f}s elapsed)")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    threshold, oof_f1 = tune_threshold(y, oof)
    elapsed = time.time() - t0

    np.save(ART / f"{key}_oof.npy", oof)
    np.save(ART / f"{key}_test.npy", test_pred)
    if not sample:
        np.save(ART / "y.npy", y)
        np.save(ART / "test_ids.npy", te[data.ID_COL].to_numpy())

    from sklearn.metrics import f1_score
    metrics = {
        "model": key, "hf_name": name, "family": "transformer",
        "oof_f1": round(oof_f1, 5),
        "f1_at_0.5": round(f1_score(y, (oof >= 0.5).astype(int)), 5),
        "threshold": round(threshold, 4),
        "fold_mean": round(float(np.mean(fold_f1s)), 5),
        "fold_std": round(float(np.std(fold_f1s)), 5),
        "fold_f1s": [round(x, 5) for x in fold_f1s],
        "epochs": epochs, "batch": batch, "lr": lr, "max_len": max_len,
        "n_rows": int(len(y)), "sample": sample, "device": device,
        "elapsed_s": round(elapsed, 1),
    }
    with open(ART / f"{key}_metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"[{key}] OOF F1 = {oof_f1:.5f} @ t={threshold:.3f}  "
          f"(fold {metrics['fold_mean']:.5f}±{metrics['fold_std']:.5f}; {elapsed:.0f}s)")
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="roberta", help="roberta|distilbert|bertweet|<hf-name>")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--folds", type=int, default=cv.N_SPLITS)
    ap.add_argument("--seed", type=int, default=cv.SEED)
    ap.add_argument("--sample", type=int, default=None)
    args = ap.parse_args()
    run_cv(args.model, args.epochs, args.batch, args.lr, args.max_len,
           args.folds, args.seed, args.sample)


if __name__ == "__main__":
    main()
