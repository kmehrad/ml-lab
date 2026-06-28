"""Leakage-safe, stateless feature engineering applied identically to train, test and original data.

The raw data is clean (no missing values, no sentinels), so feature work is about giving the models
informative transforms grounded in astronomy:
- **Color indices** (differences of magnitude bands) carry the SED shape; we use all pairwise colors.
- **redshift transforms** — redshift is the dominant separator (STAR≈0, GALAXY≈0.5, QSO≈1.9).
- **redshift × color interactions** — QSOs are blue (u−g ≲ 1) at low z but red (u−g ≳ 2) at z>2.5
  (Lyman-α enters the u band), so the color/class relationship depends on redshift.
- **Magnitude aggregates** across ugriz (overall brightness / spread).
Categorical columns are returned as pandas `category` dtype so LightGBM/CatBoost use them natively.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from . import data as D

BANDS = ["u", "g", "r", "i", "z"]
# All pairwise color differences (band order is blue->red, so a-b is the standard sign).
COLOR_PAIRS = list(itertools.combinations(BANDS, 2))  # 10 pairs
COLOR_COLS = [f"{a}_{b}" for a, b in COLOR_PAIRS]

# redshift x color interactions (physically meaningful; see module docstring).
RZ_COLOR_COLS = [f"rz_{c}" for c in COLOR_COLS]

REDSHIFT_COLS = ["redshift_log1p", "redshift_is_neg", "redshift_bin"]
MAG_AGG_COLS = ["mag_mean", "mag_std", "mag_min", "mag_max", "mag_range"]

# Redshift regime edges separating STAR (~0) / GALAXY (~0.5) / QSO (high), incl. the z>2.5 QSO turn.
REDSHIFT_BIN_EDGES = [-np.inf, 0.0, 0.1, 0.3, 0.7, 1.2, 2.5, np.inf]

NUMERIC_FEATURES = (
    D.NUMERIC_COLS + COLOR_COLS + RZ_COLOR_COLS + REDSHIFT_COLS + MAG_AGG_COLS
)
CATEGORICAL_FEATURES = D.CATEGORICAL_COLS
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a feature matrix (ALL_FEATURES columns) from a raw train/test/original frame."""
    out = df[D.NUMERIC_COLS].copy()

    for a, b in COLOR_PAIRS:
        out[f"{a}_{b}"] = df[a] - df[b]

    # redshift can be marginally negative (~ -0.01); log1p is well-defined for > -1.
    rz = df["redshift"]
    out["redshift_log1p"] = np.log1p(rz.clip(lower=-0.999))
    out["redshift_is_neg"] = (rz < 0).astype("int8")
    out["redshift_bin"] = pd.cut(rz, bins=REDSHIFT_BIN_EDGES, labels=False).astype("int16")

    for c in COLOR_COLS:
        out[f"rz_{c}"] = rz * out[c]

    mags = df[BANDS]
    out["mag_mean"] = mags.mean(axis=1)
    out["mag_std"] = mags.std(axis=1)
    out["mag_min"] = mags.min(axis=1)
    out["mag_max"] = mags.max(axis=1)
    out["mag_range"] = out["mag_max"] - out["mag_min"]

    for c in D.CATEGORICAL_COLS:
        out[c] = df[c].astype("category")

    return out[ALL_FEATURES]
