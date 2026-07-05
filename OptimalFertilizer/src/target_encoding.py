"""In-fold (leak-safe) target encoding of feature combinations — the S5E6 key lever.

Following the competition's 1st-place approach: treat all 8 features as categorical, form every
combination up to a given ``order`` (singles, pairs, triples, …), and target-encode each combo
against the 7 one-vs-rest class indicators. This injects *label* information about high-order
feature interactions that axis-aligned trees reach only inefficiently on this low-signal data —
the piece our raw-combo feature engineering (Step 4) was missing.

Leakage control (all encodings computed **inside** each outer CV fold):
  * outer validation / test rows are encoded from a mapping built on the **full** outer-training
    fold;
  * outer training rows are encoded **leave-one-out** — the row's own label is subtracted from its
    bucket before smoothing — so a row never sees its own target.
All mappings use Bayesian smoothing toward the per-class global prior (weight ``m``); unseen keys
fall back to the prior. The outer-validation encoding (the one that drives OOF MAP@3) is fully
leak-free, so OOF stays an honest gate.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .data import CATEGORICAL, N_CLASSES, NUMERIC

# All 8 raw features are treated as categorical keys for combination encoding.
COMBO_FEATURES = list(NUMERIC) + list(CATEGORICAL)
_SEP = "\x1f"  # unit-separator: safe join char that cannot appear in the values


def combos(order: int, order_min: int = 1) -> list[tuple[str, ...]]:
    """All feature combinations with size in ``[order_min, order]`` (singles..order-tuples)."""
    out: list[tuple[str, ...]] = []
    for r in range(order_min, order + 1):
        out += list(itertools.combinations(COMBO_FEATURES, r))
    return out


def _key(df: pd.DataFrame, combo: tuple[str, ...]) -> pd.Series:
    """Single string key joining the member columns of ``combo``."""
    s = df[combo[0]].astype(str)
    for c in combo[1:]:
        s = s + _SEP + df[c].astype(str)
    return s


def _bucket_table(keys: pd.Series, y: np.ndarray):
    """Per-bucket class counts ``S`` (buckets×7) and bucket sizes ``N`` for the given keys."""
    g = pd.DataFrame({"k": keys.to_numpy(), "y": y})
    S = (g.groupby("k")["y"].value_counts().unstack(fill_value=0)
         .reindex(columns=range(N_CLASSES), fill_value=0))
    N = S.sum(axis=1)
    return S, N


def encode_combo(ktr, ytr, kva, kte, m: float, prior: np.ndarray):
    """Leak-safe TE of one combo: LOO for train, full-train mapping for valid/test.

    Returns ``(tr_enc, va_enc, te_enc)`` each ``(n, 7)`` float arrays; ``te_enc`` is ``None`` when
    ``kte`` is ``None``.
    """
    S, N = _bucket_table(ktr, ytr)
    Sv, Nv = S.to_numpy(), N.to_numpy()[:, None]

    # --- train: leave-one-out (subtract the row's own label from its bucket) ---
    S_tr = S.reindex(ktr.to_numpy()).to_numpy()
    N_tr = N.reindex(ktr.to_numpy()).to_numpy()[:, None]
    onehot = np.zeros((len(ytr), N_CLASSES))
    onehot[np.arange(len(ytr)), ytr] = 1.0
    tr_enc = (S_tr - onehot + m * prior) / (N_tr - 1.0 + m)

    def lookup(k):
        Sx = S.reindex(k.to_numpy()).to_numpy()
        Nx = N.reindex(k.to_numpy()).to_numpy()[:, None]
        enc = (Sx + m * prior) / (Nx + m)
        miss = np.isnan(Sx).any(axis=1)          # keys unseen in outer-train
        enc[miss] = prior
        return enc

    va_enc = lookup(kva)
    te_enc = lookup(kte) if kte is not None else None
    return tr_enc.astype(np.float32), va_enc.astype(np.float32), \
        (te_enc.astype(np.float32) if te_enc is not None else None)


def build_te_features(Xtr_raw, ytr, Xva_raw, Xte_raw, combo_list, m: float = 10.0):
    """Concatenate leak-safe TE features for every combo. Returns (Xtr, Xva, Xte, names)."""
    prior = np.bincount(ytr, minlength=N_CLASSES).astype(float) / len(ytr)
    tr_parts, va_parts, te_parts, names = [], [], [], []
    for combo in combo_list:
        ktr, kva = _key(Xtr_raw, combo), _key(Xva_raw, combo)
        kte = _key(Xte_raw, combo) if Xte_raw is not None else None
        t, v, e = encode_combo(ktr, ytr, kva, kte, m, prior)
        tr_parts.append(t)
        va_parts.append(v)
        if e is not None:
            te_parts.append(e)
        names += [f"te_{'_'.join(combo)}_{c}" for c in range(N_CLASSES)]
    Xtr = np.hstack(tr_parts)
    Xva = np.hstack(va_parts)
    Xte = np.hstack(te_parts) if te_parts else None
    return Xtr, Xva, Xte, names
