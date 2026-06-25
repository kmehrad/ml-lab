"""Smoothed group target encoding, computed inside each CV fold (leakage-free).

The competition ships `gte_*` columns but they appear to be fit on the full train set
(leaky) and we dropped them. Here we recompute target encodings, fitting only on each
fold's training rows and transforming valid + test.

IMPORTANT: encodings keyed on `building_id` are excluded — train/test buildings are
disjoint, so such encodings never generalize (always fall back to the global prior under
GroupKFold). We only encode groupings that transfer across buildings.
"""
from __future__ import annotations

import pandas as pd

# Cross-building generalizable groupings (single + interactions). No building_id.
GROUPINGS = [
    ("primary_use",), ("site_id",), ("hour",), ("weekday",), ("month",),
    ("primary_use", "hour"), ("site_id", "hour"), ("primary_use", "weekday"),
    ("primary_use", "month"), ("site_id", "month"), ("hour", "weekday"),
]


def te_columns() -> list[str]:
    return ["te_" + "_".join(g) for g in GROUPINGS]


class TargetEncoder:
    """Additive-smoothing mean target encoder: (n*mean + m*prior)/(n+m)."""

    def __init__(self, groupings=GROUPINGS, smoothing: float = 20.0):
        self.groupings = groupings
        self.smoothing = smoothing

    def fit(self, df: pd.DataFrame, y):
        self.prior_ = float(pd.Series(y).mean())
        tmp = df[[c for g in self.groupings for c in g]].copy()
        tmp = tmp.loc[:, ~tmp.columns.duplicated()]
        tmp["__y"] = pd.Series(y).to_numpy()
        self.maps_ = {}
        for g in self.groupings:
            agg = tmp.groupby(list(g), observed=True)["__y"].agg(["mean", "count"])
            self.maps_[g] = (agg["count"] * agg["mean"] + self.smoothing * self.prior_) / (
                agg["count"] + self.smoothing
            )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for g in self.groupings:
            name = "te_" + "_".join(g)
            m = self.maps_[g]
            if len(g) == 1:
                vals = df[g[0]].map(m)
            else:
                vals = pd.Index(pd.MultiIndex.from_frame(df[list(g)])).map(m)
            out[name] = pd.Series(vals, index=df.index).astype("float32").fillna(self.prior_)
        return pd.DataFrame(out, index=df.index)
