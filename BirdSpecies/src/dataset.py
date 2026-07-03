"""Torch multi-label dataset over cached log-mel spectrograms (Phase 5 CNN).

Each item is a single-channel spectrogram (per-spectrogram standardized) and a
87-dim multi-hot target. SpecAugment (freq/time masking) is applied on the
training split only — cheap regularization that matters given the tiny dataset.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


def _standardize(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-6)


def spec_augment(x: np.ndarray, n_freq: int = 2, n_time: int = 2,
                 max_f: int = 16, max_t: int = 24, rng=np.random) -> np.ndarray:
    x = x.copy()
    n_mels, n_frames = x.shape
    for _ in range(n_freq):
        f = rng.randint(0, max_f + 1)
        f0 = rng.randint(0, max(1, n_mels - f))
        x[f0 : f0 + f, :] = 0.0
    for _ in range(n_time):
        t = rng.randint(0, max_t + 1)
        t0 = rng.randint(0, max(1, n_frames - t))
        x[:, t0 : t0 + t] = 0.0
    return x


class MelDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray | None = None, augment: bool = False):
        self.X = X
        self.Y = Y
        self.augment = augment

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int):
        x = _standardize(self.X[i])
        if self.augment:
            x = spec_augment(x)
        xt = torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0).float()  # (1, H, W)
        if self.Y is None:
            return xt
        return xt, torch.from_numpy(self.Y[i]).float()
