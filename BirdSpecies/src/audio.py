"""WAV → log-mel spectrogram cache for the CNN track (Phase 5).

Decodes each clip to a fixed-size log-mel spectrogram (``N_MELS × T_FRAMES``) and
caches the stacked array per split to ``data/processed/`` (gitignored). Needs the
``audio`` extra (``uv sync --extra audio``).

Clips are ≤5 s; we resample to 22.05 kHz (Nyquist 11 kHz covers most bird energy,
halving compute) and pad/crop to a common frame length.
"""
from __future__ import annotations

import numpy as np

from src import data

SR = 22_050
N_MELS = 128
N_FFT = 1024
HOP = 512
T_FRAMES = 216  # ~5 s at SR/HOP


def log_mel(path) -> np.ndarray:
    """One clip → ``(N_MELS, T_FRAMES)`` float32 log-mel spectrogram."""
    import librosa

    y, _ = librosa.load(path, sr=SR, mono=True)
    if y.size < N_FFT:
        y = np.pad(y, (0, N_FFT - y.size))
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, power=2.0
    )
    lm = librosa.power_to_db(mel + 1e-10).astype(np.float32)
    if lm.shape[1] < T_FRAMES:
        lm = np.pad(lm, ((0, 0), (0, T_FRAMES - lm.shape[1])), mode="edge")
    else:
        lm = lm[:, :T_FRAMES]
    return lm


def mel_cache(split: str, cache: bool = True) -> np.ndarray:
    """Stacked log-mels for a split: ``(N, N_MELS, T_FRAMES)`` float32."""
    if split not in ("train", "test"):
        raise ValueError(split)
    cache_path = data.PROCESSED / f"{split}_logmel.npy"
    if cache and cache_path.exists():
        return np.load(cache_path)
    n = data.N_TRAIN if split == "train" else data.N_TEST
    path_fn = data.train_wav_path if split == "train" else data.test_wav_path
    X = np.stack([log_mel(path_fn(i)) for i in range(1, n + 1)])
    if cache:
        data.PROCESSED.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, X)
    return X
