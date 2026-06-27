"""Torch dataset + transforms for the JPEG-decoded flower images.

Reads rows from the ``metadata_{res}.parquet`` index (see ``src/convert.py``). Transforms
are built with ``timm.data.create_transform`` so augmentation / normalization match the
pretrained backbone.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src import data


class FlowerDataset(Dataset):
    """Yields ``(image_tensor, label, id)``. Unlabeled rows use ``label = -1``."""

    def __init__(self, df: pd.DataFrame, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.paths = self.df["path"].tolist()
        self.ids = self.df["id"].tolist()
        if "label" in self.df.columns:
            labels = self.df["label"].to_numpy()
            self.labels = [int(v) if pd.notna(v) else -1 for v in labels]
        else:
            self.labels = [-1] * len(self.df)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        img = Image.open(data.ROOT / self.paths[i]).convert("RGB")
        return self.transform(img), self.labels[i], self.ids[i]


def build_transforms(res: int, model_cfg: dict, *, train: bool, auto_augment: str | None):
    """Build a timm transform pipeline sized to ``res`` using the model's data config.

    ``model_cfg`` comes from ``timm.data.resolve_data_config({}, model=model)``.
    """
    from timm.data import create_transform

    return create_transform(
        input_size=(3, res, res),
        is_training=train,
        mean=model_cfg["mean"],
        std=model_cfg["std"],
        interpolation=model_cfg.get("interpolation", "bicubic"),
        crop_pct=model_cfg.get("crop_pct", 0.95) if not train else None,
        auto_augment=auto_augment if train else None,
        re_prob=0.25 if train else 0.0,          # random erasing
        hflip=0.5 if train else 0.0,
    )


def class_counts(df: pd.DataFrame) -> torch.Tensor:
    """Per-class sample counts (length ``N_CLASSES``) over labeled rows."""
    counts = torch.zeros(data.N_CLASSES, dtype=torch.long)
    for lbl in df["label"].dropna().astype(int):
        counts[int(lbl)] += 1
    return counts


def sample_weights(df: pd.DataFrame) -> torch.Tensor:
    """Inverse-frequency weight per row, for ``WeightedRandomSampler`` (macro-F1 aid)."""
    counts = class_counts(df).clamp(min=1)
    inv = 1.0 / counts.float()
    labels = df["label"].astype(int).to_numpy()
    return torch.tensor([inv[l] for l in labels], dtype=torch.double)
