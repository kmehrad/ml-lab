"""Leak-safe k-fold OOF combination target encoding.

Gives the GBDTs direct numeric access to ``P(class | feature-combo)`` for combinations of features that
axis-aligned splits reach only indirectly (e.g. ``stress_level × sleep_duration-bin``). Numerics are
binned to deciles and treated as categorical, then combined with the 6 raw categoricals; every
combination up to ``order`` is encoded against each of the 3 class indicators.

**Leakage discipline** (the load-bearing part — cf. [[optimalfertilizer-te-negative-result]]): train-row
encodings use an inner k-fold OOF so a row never sees its own label; validation/test rows use the full
train-fold mapping. Both are computed *inside* the outer CV fold from the fold's training rows only.
Smoothing ``m`` shrinks small-group estimates toward the class prior.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from .data import CATEGORICAL, NUMERIC, N_CLASSES

BINS = 10
COMBO_FEATURES = list(NUMERIC) + list(CATEGORICAL)


def combos(order: int) -> list[tuple[str, ...]]:
    """All feature combinations of size 1..order over the 13 base features."""
    out: list[tuple[str, ...]] = []
    for o in range(1, order + 1):
        out += list(combinations(COMBO_FEATURES, o))
    return out


def build_keys(df: pd.DataFrame, combo_list: list[tuple[str, ...]]) -> pd.DataFrame:
    """Turn each feature combo into a single string key column (numerics decile-binned; NaN → 'na')."""
    disc = {}
    for c in NUMERIC:
        disc[c] = pd.qcut(df[c], BINS, labels=False, duplicates="drop").astype("Int32").astype(str).fillna("na")
    for c in CATEGORICAL:
        disc[c] = df[c].astype("object").fillna("na").astype(str)
    disc = pd.DataFrame(disc, index=df.index)
    keys = {}
    for combo in combo_list:
        name = "|".join(combo)
        key = disc[combo[0]]
        for c in combo[1:]:                           # vectorised string concat (not row-wise agg)
            key = key + "¦" + disc[c]
        keys[name] = key
    return pd.DataFrame(keys, index=df.index)


def _encode_one(ktr: pd.Series, Y: np.ndarray, kva: pd.Series, kte: pd.Series | None,
                m: float, prior: np.ndarray, inner_folds) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """OOF-encode a single key column → (n_tr,C), (n_va,C), (n_te,C) class-probability arrays."""
    C = N_CLASSES
    ktr = ktr.to_numpy()
    enc_tr = np.empty((len(ktr), C), dtype=np.float32)

    def _map(keys_from, Yfrom, keys_to):
        d = pd.DataFrame(Yfrom, columns=list(range(C))); d["k"] = keys_from
        g = d.groupby("k", observed=True).agg(["sum", "count"])
        s = g.xs("sum", axis=1, level=1).to_numpy(); n = g.xs("count", axis=1, level=1).to_numpy()[:, :1]
        enc = (s + m * prior) / (n + m)                       # (n_groups, C)
        out = np.array(pd.DataFrame(enc, index=g.index).reindex(keys_to), dtype=np.float32)  # vectorized
        miss = np.isnan(out[:, 0])                            # unseen keys → class prior
        if miss.any():
            out[miss] = prior
        return out

    for itr, iva in inner_folds:                      # inner OOF for train rows
        enc_tr[iva] = _map(ktr[itr], Y[itr], ktr[iva])
    enc_va = _map(ktr, Y, kva.to_numpy())             # full-fold map for valid
    enc_te = _map(ktr, Y, kte.to_numpy()) if kte is not None else None
    return enc_tr, enc_va, enc_te


def build_te_features(keys_tr: pd.DataFrame, y: np.ndarray, keys_va: pd.DataFrame,
                      keys_te: pd.DataFrame | None, m: float = 20.0,
                      n_inner: int = 5, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Encode every key column → horizontally-stacked (n, n_combos*C) OOF TE matrices for tr/va/te."""
    from sklearn.model_selection import StratifiedKFold
    prior = np.bincount(y, minlength=N_CLASSES).astype(np.float64)
    prior = prior / prior.sum()
    Y = np.eye(N_CLASSES, dtype=np.float32)[y]
    inner = list(StratifiedKFold(n_inner, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))

    tr_parts, va_parts, te_parts = [], [], []
    for col in keys_tr.columns:
        a, b, c = _encode_one(keys_tr[col], Y, keys_va[col],
                              keys_te[col] if keys_te is not None else None, m, prior, inner)
        tr_parts.append(a); va_parts.append(b)
        if c is not None:
            te_parts.append(c)
    Xtr = np.hstack(tr_parts); Xva = np.hstack(va_parts)
    Xte = np.hstack(te_parts) if te_parts else None
    return Xtr, Xva, Xte
