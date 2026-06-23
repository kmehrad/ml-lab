"""Model-ready preprocessing pipelines.

Two encodings are offered because the EDA argues for gradient-boosted trees as
the baseline while a linear model is a useful sanity check:

* ``"tree"`` — feature engineering plus native ``category`` dtypes. LightGBM,
  CatBoost, and XGBoost consume this directly; no scaling or one-hot needed.
* ``"linear"`` — feature engineering, standardized numerics, and one-hot encoded
  (low-cardinality) categoricals, yielding a dense float matrix.

The data has no missing values, no skew, and no outliers (see
``reports/EDA_FINDINGS.md``), so neither pipeline imputes, transforms, or clips.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL_FEATURES
from .features import IrrigationFeatureEngineer

PreprocessorKind = str  # "tree" | "linear"


def build_preprocessor(
    kind: PreprocessorKind = "tree",
    *,
    add_engineered: bool = True,
) -> Pipeline:
    """Build a fit/transform pipeline for the requested model family."""
    if kind == "tree":
        return Pipeline(
            steps=[
                (
                    "features",
                    IrrigationFeatureEngineer(
                        add_engineered=add_engineered,
                        categorical_as_category=True,
                    ),
                )
            ]
        )
    if kind == "linear":
        encoder = ColumnTransformer(
            transformers=[
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    list(CATEGORICAL_FEATURES),
                ),
            ],
            remainder=StandardScaler(),
            verbose_feature_names_out=False,
        )
        encoder.set_output(transform="pandas")
        return Pipeline(
            steps=[
                (
                    "features",
                    IrrigationFeatureEngineer(
                        add_engineered=add_engineered,
                        categorical_as_category=False,
                    ),
                ),
                ("encode", encoder),
            ]
        )
    raise ValueError(f"kind must be 'tree' or 'linear', got {kind!r}")


def prepare_tree_features(
    frame: pd.DataFrame,
    *,
    add_engineered: bool = True,
) -> pd.DataFrame:
    """One-shot helper returning a tree-ready frame (categoricals as ``category``).

    For leakage-free cross-validation, prefer :func:`build_preprocessor` and fit
    it on the training fold only.
    """
    return build_preprocessor("tree", add_engineered=add_engineered).fit_transform(frame)
