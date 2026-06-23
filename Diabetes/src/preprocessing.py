"""Model-ready preprocessing pipelines.

After feature engineering (ordinal encoding + clinical composites), the only
remaining non-numeric columns are the three nominal categoricals, which are
one-hot encoded. Two flavours are offered:

* ``"tree"`` — feature engineering + one-hot of nominal categoricals; everything
  else passes through. Consumed directly by LightGBM/XGBoost/CatBoost/HistGB.
* ``"linear"`` — the same, plus ``StandardScaler`` on all non-one-hot columns,
  yielding a dense float matrix for LogisticRegression.

The data has no missing values (see ``reports/EDA_FINDINGS.md``), so neither
pipeline imputes. Both are fit on the training fold only inside cross-validation.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import NOMINAL_FEATURES
from .features import DiabetesFeatureEngineer

PreprocessorKind = str  # "tree" | "linear"


def build_preprocessor(
    kind: PreprocessorKind = "tree",
    *,
    add_engineered: bool = True,
) -> Pipeline:
    """Build a fit/transform pipeline for the requested model family."""
    if kind not in {"tree", "linear"}:
        raise ValueError(f"kind must be 'tree' or 'linear', got {kind!r}")

    onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    remainder = StandardScaler() if kind == "linear" else "passthrough"
    encoder = ColumnTransformer(
        transformers=[("nominal", onehot, list(NOMINAL_FEATURES))],
        remainder=remainder,
        verbose_feature_names_out=False,
    )
    encoder.set_output(transform="pandas")
    return Pipeline(
        steps=[
            ("features", DiabetesFeatureEngineer(add_engineered=add_engineered)),
            ("encode", encoder),
        ]
    )


def prepare_tree_features(frame: pd.DataFrame, *, add_engineered: bool = True) -> pd.DataFrame:
    """One-shot helper returning a tree-ready (all-numeric) frame.

    For leakage-free cross-validation, prefer :func:`build_preprocessor` and fit
    it on the training fold only.
    """
    return build_preprocessor("tree", add_engineered=add_engineered).fit_transform(frame)
