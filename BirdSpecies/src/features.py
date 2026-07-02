"""Fixed-length per-file features for the classical models.

Three feature levels (``feats`` argument), each a fixed vector per clip:

- ``base`` (86) — per-coefficient mean/std/min/max/median of the provided MFCC
  cepstra + ``log1p(n_frames)``. (Phase 3 baseline.)
- ``ext``  (154) — ``base`` + MFCC **delta** (Δ over frames) mean/std +
  10th/90th **percentiles**. Captures temporal change / distribution shape that
  plain moments discard. No extra dependency (derived from the given MFCC).
- ``full`` (182) — ``ext`` + **librosa** spectral/mel descriptors from the raw
  WAV (centroid, bandwidth, rolloff, flatness, ZCR, RMS mean/std + 16 log-mel
  band means). Needs the ``audio`` extra (``uv sync --extra audio``).

Matrices are cached to ``data/processed/`` since building touches ~1700 files
(and ``full`` also decodes the WAVs).
"""
from __future__ import annotations

import warnings

import numpy as np

from src import data

_STATS = ("mean", "std", "min", "max", "median")
_N_AUDIO_MELS = 16
_SPECTRAL = ("centroid", "bandwidth", "rolloff", "flatness", "zcr", "rms")


def _nan(m: np.ndarray) -> np.ndarray:
    """Replace non-finite entries (silent-frame -inf/nan) with nan for masking."""
    return np.where(np.isfinite(m), m, np.nan)


def _clean(v: np.ndarray) -> np.ndarray:
    return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)


# ── MFCC-derived blocks ──────────────────────────────────────────────
def _mfcc_moments(m: np.ndarray) -> np.ndarray:
    m = _nan(m)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        v = np.concatenate([np.nanmean(m, 0), np.nanstd(m, 0), np.nanmin(m, 0),
                            np.nanmax(m, 0), np.nanmedian(m, 0)])
    return _clean(v)  # 85


def _mfcc_delta(m: np.ndarray) -> np.ndarray:
    m = _nan(m)
    d = np.diff(m, axis=0) if m.shape[0] > 1 else np.zeros((1, data.N_MFCC))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        v = np.concatenate([np.nanmean(d, 0), np.nanstd(d, 0)])
    return _clean(v)  # 34


def _mfcc_pctl(m: np.ndarray) -> np.ndarray:
    m = _nan(m)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        v = np.concatenate([np.nanpercentile(m, 10, axis=0), np.nanpercentile(m, 90, axis=0)])
    return _clean(v)  # 34


def mfcc_stats(m: np.ndarray) -> np.ndarray:
    """``base`` vector (86): moments + log1p(n_frames)."""
    return np.append(_mfcc_moments(m), np.log1p(m.shape[0]))


# ── raw-audio (librosa) block ────────────────────────────────────────
def audio_feats(path) -> np.ndarray:
    """Spectral + log-mel descriptors from a WAV (12 + 16 = 28)."""
    import librosa

    y, sr = librosa.load(path, sr=None, mono=True)
    if y.size < 2:
        return np.zeros(2 * len(_SPECTRAL) + _N_AUDIO_MELS)
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=512)) + 1e-10
    series = {
        "centroid": librosa.feature.spectral_centroid(S=S, sr=sr),
        "bandwidth": librosa.feature.spectral_bandwidth(S=S, sr=sr),
        "rolloff": librosa.feature.spectral_rolloff(S=S, sr=sr),
        "flatness": librosa.feature.spectral_flatness(S=S),
        "zcr": librosa.feature.zero_crossing_rate(y),
        "rms": librosa.feature.rms(S=S, frame_length=1024),
    }
    parts = []
    for k in _SPECTRAL:
        x = series[k]
        parts += [float(x.mean()), float(x.std())]
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=_N_AUDIO_MELS)
    parts += librosa.power_to_db(mel + 1e-10).mean(axis=1).tolist()
    return _clean(np.asarray(parts, dtype=float))


# ── assembly ─────────────────────────────────────────────────────────
def _vector(file_id: int, split: str, feats: str) -> np.ndarray:
    mfcc_path = data.train_mfcc_path if split == "train" else data.test_mfcc_path
    m = data.load_mfcc(mfcc_path(file_id))
    blocks = [_mfcc_moments(m)]
    if feats in ("ext", "full"):
        blocks += [_mfcc_delta(m), _mfcc_pctl(m)]
    if feats == "full":
        wav_path = data.train_wav_path if split == "train" else data.test_wav_path
        blocks.append(audio_feats(wav_path(file_id)))
    blocks.append(np.array([np.log1p(m.shape[0])]))
    return np.concatenate(blocks)


def feature_names(feats: str = "base") -> list[str]:
    names = [f"mfcc{c}_{s}" for s in _STATS for c in range(data.N_MFCC)]
    if feats in ("ext", "full"):
        names += [f"dmfcc{c}_mean" for c in range(data.N_MFCC)]
        names += [f"dmfcc{c}_std" for c in range(data.N_MFCC)]
        names += [f"mfcc{c}_p10" for c in range(data.N_MFCC)]
        names += [f"mfcc{c}_p90" for c in range(data.N_MFCC)]
    if feats == "full":
        for k in _SPECTRAL:
            names += [f"{k}_mean", f"{k}_std"]
        names += [f"logmel{b}_mean" for b in range(_N_AUDIO_MELS)]
    return names + ["log_nframes"]


def feature_matrix(split: str, feats: str = "base", cache: bool = True) -> np.ndarray:
    """Build (or load cached) feature matrix. ``split`` ∈ {train,test},
    ``feats`` ∈ {base,ext,full}. Row *i* ↔ file *i+1*."""
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train'/'test', got {split!r}")
    if feats not in ("base", "ext", "full"):
        raise ValueError(f"feats must be base/ext/full, got {feats!r}")
    tag = "mfcc" if feats == "base" else feats  # keep base cache name back-compatible
    cache_path = data.PROCESSED / f"{split}_{tag}_feats.npy"
    if cache and cache_path.exists():
        return np.load(cache_path)

    n = data.N_TRAIN if split == "train" else data.N_TEST
    X = np.stack([_vector(i, split, feats) for i in range(1, n + 1)])
    if cache:
        data.PROCESSED.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, X)
    return X


def frame_counts(split: str) -> np.ndarray:
    """Number of MFCC frames per file (proxy for clip duration/activity)."""
    n = data.N_TRAIN if split == "train" else data.N_TEST
    path_fn = data.train_mfcc_path if split == "train" else data.test_mfcc_path
    return np.array([data.load_mfcc(path_fn(i)).shape[0] for i in range(1, n + 1)])
