"""Fixed-length per-file features for the classical models.

The provided MFCC cepstra are variable length (``n_frames × 17``). To feed them
to per-class classical models we summarize each file's frames into a fixed
vector of per-coefficient statistics. Shared by EDA (Phase 2) and the classical
baseline (Phase 3); Phase 4 extends this with richer descriptors.

Feature matrices are cached to ``data/processed/`` since building them touches
~1700 files.
"""
from __future__ import annotations

import warnings

import numpy as np

from src import data

_STATS = ("mean", "std", "min", "max", "median")


def mfcc_stats(m: np.ndarray) -> np.ndarray:
    """Summarize one ``(n_frames, 17)`` MFCC array → ``(5*17 + 1,)`` vector:
    per-coefficient mean/std/min/max/median, plus ``log1p(n_frames)`` (a proxy
    for clip length / activity).

    Robust to non-finite entries: a few cepstra files contain ``-inf``/``nan``
    (silent frames → log of zero energy). These are masked with nan-aware
    reductions and any all-non-finite result is floored to 0.
    """
    m = np.where(np.isfinite(m), m, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-nan slices
        stats = [
            np.nanmean(m, 0),
            np.nanstd(m, 0),
            np.nanmin(m, 0),
            np.nanmax(m, 0),
            np.nanmedian(m, 0),
        ]
    v = np.nan_to_num(np.concatenate(stats), nan=0.0, posinf=0.0, neginf=0.0)
    return np.append(v, np.log1p(m.shape[0]))


def feature_names() -> list[str]:
    names = [f"mfcc{c}_{s}" for s in _STATS for c in range(data.N_MFCC)]
    return names + ["log_nframes"]


def feature_matrix(split: str, cache: bool = True) -> np.ndarray:
    """Build (or load cached) the aggregated-MFCC feature matrix for a split.

    ``split`` ∈ {"train", "test"} → array of shape ``(N_TRAIN|N_TEST, 86)``,
    row *i* ↔ file *i+1*.
    """
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train'/'test', got {split!r}")
    cache_path = data.PROCESSED / f"{split}_mfcc_feats.npy"
    if cache and cache_path.exists():
        return np.load(cache_path)

    n = data.N_TRAIN if split == "train" else data.N_TEST
    path_fn = data.train_mfcc_path if split == "train" else data.test_mfcc_path
    X = np.stack([mfcc_stats(data.load_mfcc(path_fn(i))) for i in range(1, n + 1)])

    if cache:
        data.PROCESSED.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, X)
    return X


def frame_counts(split: str) -> np.ndarray:
    """Number of MFCC frames per file (a proxy for clip duration / activity)."""
    n = data.N_TRAIN if split == "train" else data.N_TEST
    path_fn = data.train_mfcc_path if split == "train" else data.test_mfcc_path
    return np.array([data.load_mfcc(path_fn(i)).shape[0] for i in range(1, n + 1)])
