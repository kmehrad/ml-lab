"""Phase 2: DeBERTa-v3 fine-tune for 3-class Chatbot Arena preference.

Sequence-pair framing: each example is [CLS] prompt [SEP] response_a [SEP]
response_b [SEP]. Each field is token-budgeted with **head+tail** truncation so
one very long answer cannot crowd out the prompt or the other answer
(CLAUDE.md). Trains on the shared stratified folds so the out-of-fold log loss
is directly comparable to the LightGBM baseline (1.01541).

Evaluation uses **A/B-swap TTA**: every validation row is scored as (A,B) and
(B,A); the two probability vectors are averaged after swapping the A/B columns
of the swapped pass. This removes position bias and matches how the inference
notebook will score the hidden test set.

Run (smoke, single fold):
  python -m src.train_deberta --fold 0 --epochs 2
Run (full CV):
  python -m src.train_deberta --fold all --epochs 2
Outputs:
  data/processed/oof_deberta.npy            (full run; OOF probs, n x 3)
  data/processed/oof_deberta_fold{f}.npy    (single-fold run; val probs only)
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from sklearn.metrics import log_loss
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from .data import PROC, SEED, TARGETS, join_turns, load_train


# ---------------------------------------------------------------- encoding ---
def head_tail(ids: list[int], budget: int) -> list[int]:
    """Keep the first and last tokens when a field exceeds its budget."""
    if len(ids) <= budget:
        return ids
    head = budget // 2
    tail = budget - head
    return ids[:head] + ids[-tail:]


def build_example(tok, prompt: str, resp_a: str, resp_b: str, max_len: int) -> list[int]:
    """Assemble [CLS] <prompt> [SEP] <a> [SEP] <b> [SEP] with per-field budgets.

    Each field is prefixed with a textual role marker so the model can tell the
    prompt from the two candidate answers (this is a *preference* task, so which
    answer is A vs B must be unambiguous). Markers sit at the head of each field
    and so survive head+tail truncation.
    """
    avail = max_len - 4  # CLS + 3 SEP
    p_budget = int(avail * 0.25)
    r_budget = (avail - p_budget) // 2
    p = head_tail(tok.encode("Prompt: " + prompt, add_special_tokens=False), p_budget)
    a = head_tail(tok.encode("Response A: " + resp_a, add_special_tokens=False), r_budget)
    b = head_tail(tok.encode("Response B: " + resp_b, add_special_tokens=False), r_budget)
    cls, sep = tok.cls_token_id, tok.sep_token_id
    ids = [cls] + p + [sep] + a + [sep] + b + [sep]
    return ids[:max_len]


class PairDataset(Dataset):
    """Pre-tokenized rows. `swap` returns the (B,A) ordering for TTA."""

    def __init__(self, tok, df, max_len: int, swap: bool = False):
        self.ids = []
        for _, row in df.iterrows():
            p = join_turns(row["prompt"])
            a = join_turns(row["response_a"])
            b = join_turns(row["response_b"])
            if swap:
                a, b = b, a
            self.ids.append(build_example(tok, p, a, b, max_len))
        self.labels = df[TARGETS].values.argmax(1).astype(np.int64)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return self.ids[i], self.labels[i]


def make_collate(pad_id: int):
    def collate(batch):
        seqs, labels = zip(*batch)
        m = max(len(s) for s in seqs)
        input_ids = torch.full((len(seqs), m), pad_id, dtype=torch.long)
        attn = torch.zeros((len(seqs), m), dtype=torch.long)
        for i, s in enumerate(seqs):
            input_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            attn[i, : len(s)] = 1
        return input_ids, attn, torch.tensor(labels, dtype=torch.long)

    return collate


# ------------------------------------------------------------------ predict ---
@torch.no_grad()
def predict(model, loader, device, amp_dtype) -> np.ndarray:
    model.eval()
    out = []
    for input_ids, attn, _ in loader:
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            logits = model(input_ids=input_ids.to(device),
                           attention_mask=attn.to(device)).logits
        out.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(out, 0)


def predict_tta(model, tok, df, max_len, bs, device, amp_dtype) -> np.ndarray:
    """Average (A,B) and (B,A) passes; swap A/B columns of the swapped pass."""
    collate = make_collate(tok.pad_token_id)
    ld = lambda swap: DataLoader(  # noqa: E731
        PairDataset(tok, df, max_len, swap=swap),
        batch_size=bs, shuffle=False, collate_fn=collate)
    p_norm = predict(model, ld(False), device, amp_dtype)
    p_swap = predict(model, ld(True), device, amp_dtype)
    p_swap = p_swap[:, [1, 0, 2]]  # undo A<->B so columns mean the same thing
    return (p_norm + p_swap) / 2


# -------------------------------------------------------------------- train ---
def train_fold(args, tok, train, fold, device, amp_dtype):
    tr_df = train[train["fold"] != fold].reset_index(drop=True)
    va_df = train[train["fold"] == fold].reset_index(drop=True)
    if args.limit:  # smoke: shrink both splits for a fast sanity pass
        tr_df = tr_df.head(args.limit)
        va_df = va_df.head(max(256, args.limit // 4))
    print(f"  fold {fold}: train={len(tr_df)} val={len(va_df)}")

    collate = make_collate(tok.pad_token_id)
    tr_loader = DataLoader(PairDataset(tok, tr_df, args.max_len),
                           batch_size=args.batch_size, shuffle=True,
                           collate_fn=collate, drop_last=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=3).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)
    steps = (len(tr_loader) // args.grad_accum) * args.epochs
    sched = get_linear_schedule_with_warmup(
        optim, int(steps * args.warmup), steps)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_ll, best_prob = None, None
    for epoch in range(args.epochs):
        model.train()
        optim.zero_grad()
        t0 = time.time()
        for step, (input_ids, attn, labels) in enumerate(tr_loader):
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits = model(input_ids=input_ids.to(device),
                               attention_mask=attn.to(device)).logits
                loss = loss_fn(logits.float(), labels.to(device)) / args.grad_accum
            loss.backward()
            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                sched.step()
                optim.zero_grad()
            if step % 200 == 0:
                print(f"    ep{epoch} step {step}/{len(tr_loader)} "
                      f"loss {loss.item()*args.grad_accum:.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)

        prob = predict_tta(model, tok, va_df, args.max_len,
                           args.batch_size, device, amp_dtype)
        y = va_df[TARGETS].values.argmax(1)
        ll = log_loss(y, prob, labels=[0, 1, 2])
        print(f"  fold {fold} epoch {epoch}: val log loss {ll:.5f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if best_ll is None or ll < best_ll:
            best_ll, best_prob = ll, prob

    va_idx = train.index[train["fold"] == fold].to_numpy()
    if args.limit:
        va_idx = va_idx[: len(va_df)]
    return best_ll, best_prob, va_idx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/deberta-v3-base")
    ap.add_argument("--fold", default="0", help="fold index or 'all'")
    ap.add_argument("--max_len", type=int, default=1280)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--limit", type=int, default=0,
                    help="cap train rows for a fast smoke test (0 = full)")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda"
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"device={device} amp={amp_dtype} model={args.model} "
          f"max_len={args.max_len} bs={args.batch_size}x{args.grad_accum}")

    tok = AutoTokenizer.from_pretrained(args.model)
    train = load_train(with_folds=True)
    n_folds = int(train["fold"].max()) + 1
    folds = range(n_folds) if args.fold == "all" else [int(args.fold)]

    oof = np.full((len(train), 3), np.nan)
    lls = []
    for f in folds:
        ll, prob, va_idx = train_fold(args, tok, train, f, device, amp_dtype)
        oof[va_idx] = prob
        lls.append(ll)
        if args.fold != "all":
            np.save(PROC / f"oof_deberta_fold{f}.npy", prob)
            print(f"\nfold {f} log loss: {ll:.5f}  (baseline 1.01541)")

    if args.fold == "all":
        y = train[TARGETS].values.argmax(1)
        oof_ll = log_loss(y, oof, labels=[0, 1, 2])
        np.save(PROC / "oof_deberta.npy", oof)
        print(f"\nper-fold: {[round(x, 5) for x in lls]}")
        print(f"OOF log loss: {oof_ll:.5f}  (baseline 1.01541)")


if __name__ == "__main__":
    main()
