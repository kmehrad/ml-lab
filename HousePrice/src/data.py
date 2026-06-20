"""Data loading and cleaning shared by both model families.

The cleaning here is deterministic and row-independent (NA -> "None"/0 per the
Ames data dictionary, type fixes), so it is safe to apply to train and test
alike without leakage. Statistical imputation that *does* depend on the data
distribution (medians, modes) is also done here but only ever fit on the
combined-but-deterministic logic — for the few genuinely missing-at-random
columns we use simple global/group statistics, whose tiny leakage is negligible
(1-4 missing cells, almost all in test).
"""
from __future__ import annotations

import pandas as pd

from . import config


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the raw train/test CSVs as downloaded from Kaggle."""
    train = pd.read_csv(config.TRAIN_CSV)
    test = pd.read_csv(config.TEST_CSV)
    return train, test


# Numeric-looking columns that are really categorical codes (no ordinal meaning).
_AS_CATEGORICAL = ["MSSubClass", "MoSold", "YrSold"]


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values and fix mistyped columns.

    Returns a new frame; the input is not mutated. Works on either train
    (with SalePrice) or test (without).
    """
    df = df.copy()

    # 1. Semantic missing: NA means the feature is absent, per the data dict.
    for col in config.NA_MEANS_NONE:
        if col in df.columns:
            df[col] = df[col].astype("object").fillna("None")
    for col in config.NA_MEANS_ZERO:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # 2. LotFrontage: strongly tied to the neighborhood -> impute by its median,
    #    falling back to the global median for any neighborhood seen only with NA.
    if "LotFrontage" in df.columns:
        by_nbhd = df.groupby("Neighborhood")["LotFrontage"].transform("median")
        df["LotFrontage"] = df["LotFrontage"].fillna(by_nbhd)
        df["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].median())

    # 3. Mistyped numeric codes -> strings so they are treated as categories.
    for col in _AS_CATEGORICAL:
        if col in df.columns:
            df[col] = df[col].astype("Int64").astype("string")

    # 4. Whatever genuine missing-at-random remains: mode for text, median for
    #    numbers. These are 1-4 cells per column (see EDA), overwhelmingly in test.
    cat_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in cat_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode(dropna=True).iloc[0])

    num_cols = df.select_dtypes(include="number").columns
    for col in num_cols:
        if col in (config.ID_COL, config.TARGET):
            continue
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    return df


def drop_outliers(train: pd.DataFrame) -> pd.DataFrame:
    """Drop the two documented GrLivArea outliers (De Cock, 2011).

    Ids 524 and 1299 are >4500 sqft, OverallQual 10 homes that sold far below
    market. They distort every regression. TRAIN ONLY — never touch test rows.
    """
    mask = (train["GrLivArea"] > 4000) & (train[config.TARGET] < 300000)
    return train.loc[~mask].reset_index(drop=True)
