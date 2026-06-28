"""Leakage-safe categorical encoders, fit on a training fold and applied to val/test.

GBDTs use the raw categoricals natively, but frequency and out-of-fold target encodings add extra
signal. Both are implemented as fit/transform so they can be fit **inside** the CV loop on the
train-fold only (never on validation rows) — see usage in `src/train.py`.

Encoded columns are numeric and meant to be concatenated alongside the native categorical columns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as D

# Columns to encode: the two categoricals plus their interaction.
INTERACTION_COL = "spectral_type__galaxy_population"


def _with_interaction(X: pd.DataFrame) -> pd.DataFrame:
    out = X[D.CATEGORICAL_COLS].astype(str).copy()
    out[INTERACTION_COL] = out[D.CATEGORICAL_COLS[0]] + "|" + out[D.CATEGORICAL_COLS[1]]
    return out


class CategoryEncoders:
    """Frequency encoding + smoothed multiclass target (mean) encoding.

    fit(X, y) learns maps from the training fold; transform(X) returns a numeric DataFrame with
    `freq_{col}` and `te_{col}_{CLASS}` columns. Unseen categories fall back to the global prior.
    """

    def __init__(self, smoothing: float = 50.0):
        self.smoothing = smoothing
        self.cols: list[str] = []
        self.freq_maps: dict[str, dict] = {}
        self.te_maps: dict[str, pd.DataFrame] = {}
        self.priors: np.ndarray | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "CategoryEncoders":
        Xc = _with_interaction(X)
        self.cols = list(Xc.columns)
        n = len(Xc)
        self.priors = np.bincount(y, minlength=len(D.CLASSES)) / n

        y_oh = pd.get_dummies(pd.Series(y), dtype="float64")
        y_oh = y_oh.reindex(columns=range(len(D.CLASSES)), fill_value=0.0)

        for col in self.cols:
            s = Xc[col]
            self.freq_maps[col] = (s.value_counts() / n).to_dict()
            # smoothed per-class mean: (sum_c + smoothing*prior_c) / (count + smoothing)
            grp = y_oh.groupby(s.to_numpy())
            counts = grp.size()
            sums = grp.sum()
            sm = self.smoothing
            te = (sums + sm * self.priors) / (counts.to_numpy()[:, None] + sm)
            te.columns = [f"te_{col}_{D.CLASSES[c]}" for c in range(len(D.CLASSES))]
            self.te_maps[col] = te
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        Xc = _with_interaction(X)
        out = pd.DataFrame(index=X.index)
        for col in self.cols:
            s = Xc[col]
            out[f"freq_{col}"] = s.map(self.freq_maps[col]).fillna(0.0).to_numpy()
            te = self.te_maps[col]
            mapped = te.reindex(s.to_numpy())
            for c in range(len(D.CLASSES)):
                cn = f"te_{col}_{D.CLASSES[c]}"
                vals = mapped[cn].to_numpy()
                out[cn] = np.where(np.isnan(vals), self.priors[c], vals)
        return out

    def fit_transform(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        return self.fit(X, y).transform(X)
