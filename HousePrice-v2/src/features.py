"""Preprocessing and feature engineering for House Prices.

Three layers are provided:

1. :func:`engineer_features` - dataset-specific, domain-driven transforms
   (fixing "NA means none" columns, combining areas, age features, etc.).
2. :func:`build_preprocessor` - a generic sklearn ``ColumnTransformer`` that
   imputes and encodes whatever numeric/categorical columns remain. Use with
   :func:`engineer_features` inside a ``Pipeline`` so the same steps run on
   train, CV folds, and test.
3. :func:`preprocess` - an all-in-one, pandas-based pipeline (driven by the
   EDA findings) that takes raw train/test frames and returns model-ready
   ``(X_train, X_test, y, test_ids)``. Train and test are concatenated for
   consistent imputation/encoding; ``SalePrice`` is removed first so it never
   leaks into the combined frame.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import ID_COL, TARGET

# Columns where a missing value genuinely means "feature absent" (no basement,
# no garage, etc.) rather than unknown. They should be filled with "None".
NONE_CATEGORICAL = [
    "Alley", "MasVnrType", "BsmtQual", "BsmtCond", "BsmtExposure",
    "BsmtFinType1", "BsmtFinType2", "FireplaceQu", "GarageType",
    "GarageFinish", "GarageQual", "GarageCond", "PoolQC", "Fence",
    "MiscFeature",
]

# Numeric columns where NaN means 0 (no garage -> 0 cars, etc.).
ZERO_NUMERIC = [
    "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
    "BsmtFullBath", "BsmtHalfBath", "GarageCars", "GarageArea", "GarageYrBlt",
]

# MSSubClass is encoded as an int but is really a categorical code.
FORCE_CATEGORICAL = ["MSSubClass", "MoSold", "YrSold"]

# Quality/condition grades are ordinal (Po < Fa < TA < Gd < Ex); the EDA showed
# a clean monotonic price gradient, so encode them as integers rather than
# one-hot. "None" (feature absent) maps to 0.
QUALITY_MAP = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
ORDINAL_QUALITY_COLS = [
    "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
    "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC",
]

# Categorical columns to target-encode (kept raw through preprocess() so a
# fold-aware encoder can fit them inside the model pipeline -> no leakage).
TARGET_ENCODE_COLS = ["Neighborhood"]


def engineer_features(
    df: pd.DataFrame, *, ordinal: bool = True, interactions: bool = True
) -> pd.DataFrame:
    """Apply domain-specific feature engineering. Returns a new frame.

    ``ordinal`` collapses quality grades to integers (vs leaving them as
    one-hot categoricals); ``interactions`` adds the quality*size products.
    Both default on, but hurt the linear models on this dataset -- set them
    off to recover the stronger pure-one-hot representation.
    """
    df = df.copy()

    # "NA means none" handling.
    for col in NONE_CATEGORICAL:
        if col in df.columns:
            df[col] = df[col].astype("object").fillna("None")
    for col in ZERO_NUMERIC:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Treat numeric codes as categories.
    for col in FORCE_CATEGORICAL:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # Ordinal-encode quality grades (Po..Ex -> 1..5, None -> 0). Anything not in
    # the map (e.g. a stray NaN) becomes NaN and is median-filled downstream.
    if ordinal:
        for col in ORDINAL_QUALITY_COLS:
            if col in df.columns:
                df[col] = df[col].map(QUALITY_MAP).astype(float)

    # --- Engineered features ---
    # Aggregate areas / counts.
    if {"TotalBsmtSF", "1stFlrSF", "2ndFlrSF"}.issubset(df.columns):
        df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]

    if {"FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"}.issubset(df.columns):
        df["TotalBathrooms"] = (
            df["FullBath"] + 0.5 * df["HalfBath"]
            + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"]
        )

    # Interaction features: quality x size is the dominant nonlinearity.
    if interactions and "OverallQual" in df.columns:
        df["Qual_sq"] = df["OverallQual"] ** 2
        if "GrLivArea" in df.columns:
            df["Qual_x_GrLivArea"] = df["OverallQual"] * df["GrLivArea"]
        if "TotalSF" in df.columns:
            df["Qual_x_TotalSF"] = df["OverallQual"] * df["TotalSF"]
        if "TotalBathrooms" in df.columns:
            df["GrLivArea_x_Baths"] = df["GrLivArea"] * df["TotalBathrooms"]

    porch_cols = ["OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"]
    if set(porch_cols).issubset(df.columns):
        df["TotalPorchSF"] = df[porch_cols].sum(axis=1)

    # Ages (YrSold is coerced to str above, so cast back to int here). Guard
    # GarageAge: a missing GarageYrBlt was filled with 0 (no garage) and must
    # not become a ~2000-year age -> set those to 0.
    if {"YrSold", "YearBuilt"}.issubset(df.columns):
        yr_sold = df["YrSold"].astype(int)
        df["HouseAge"] = yr_sold - df["YearBuilt"]
        if "YearRemodAdd" in df.columns:
            df["RemodAge"] = yr_sold - df["YearRemodAdd"]
        if "GarageYrBlt" in df.columns:
            df["GarageAge"] = np.where(
                df["GarageYrBlt"] > 0, yr_sold - df["GarageYrBlt"], 0
            )

    # Has-X indicator flags.
    if "GarageArea" in df.columns:
        df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
    if "TotalBsmtSF" in df.columns:
        df["HasBasement"] = (df["TotalBsmtSF"] > 0).astype(int)
    if "Fireplaces" in df.columns:
        df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)
    if "PoolArea" in df.columns:
        df["HasPool"] = (df["PoolArea"] > 0).astype(int)
    if "TotalPorchSF" in df.columns:
        df["HasPorch"] = (df["TotalPorchSF"] > 0).astype(int)
    if {"YearBuilt", "YearRemodAdd"}.issubset(df.columns):
        df["IsRemodeled"] = (df["YearBuilt"] != df["YearRemodAdd"]).astype(int)

    return df


def split_column_types(df: pd.DataFrame):
    """Return ``(numeric_cols, categorical_cols)`` for the given frame."""
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = df.select_dtypes(exclude=[np.number]).columns.tolist()
    return numeric, categorical


def build_preprocessor(df: pd.DataFrame, scale: bool = False) -> ColumnTransformer:
    """Build a ColumnTransformer that imputes + encodes ``df``'s columns.

    Pass ``scale=True`` for linear models that need standardized numeric
    features; tree models can leave it ``False``.
    """
    numeric, categorical = split_column_types(df)

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps)

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical),
    ])


# ---------------------------------------------------------------------------
# All-in-one preprocessing (driven by EDA findings). See notebooks/01_eda.ipynb.
# ---------------------------------------------------------------------------

# Outliers: two homes with huge living area but a low price (Ids 524 & 1299).
# Expressed as a rule rather than hard-coded Ids so it stays reproducible.
OUTLIER_RULE = {"col": "GrLivArea", "area_gt": 4000, "price_lt": 300000}

# Default skewness above which a (non-negative) numeric feature is log1p'd.
SKEW_THRESHOLD = 0.75

# How to fill categorical columns that are NOT in NONE_CATEGORICAL (e.g.
# Electrical, MSZoning in the test set) -- "mode" or a literal label.
CAT_FILL = "mode"


def drop_training_outliers(train_df: pd.DataFrame) -> pd.DataFrame:
    """Drop the known GrLivArea outliers from a *training* frame.

    No-op if the required columns are absent (e.g. on the test set).
    """
    r = OUTLIER_RULE
    if r["col"] not in train_df.columns or TARGET not in train_df.columns:
        return train_df
    mask = (train_df[r["col"]] > r["area_gt"]) & (train_df[TARGET] < r["price_lt"])
    return train_df.loc[~mask].copy()


def _fill_missing(df: pd.DataFrame, cat_fill: str = CAT_FILL) -> pd.DataFrame:
    """Impute any values still missing after :func:`engineer_features`.

    Numeric -> median, categorical -> mode (or a literal label). Statistics
    are computed on whatever frame is passed in (the combined train+test set
    when called from :func:`preprocess`), giving consistent fills across both.
    """
    df = df.copy()
    numeric, categorical = split_column_types(df)

    for col in numeric:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    for col in categorical:
        if df[col].isna().any():
            fill = df[col].mode(dropna=True).iloc[0] if cat_fill == "mode" else cat_fill
            df[col] = df[col].fillna(fill)

    return df


def _log_skewed(df: pd.DataFrame, threshold: float = SKEW_THRESHOLD):
    """log1p-transform highly skewed, non-negative numeric features.

    Skips binary/indicator columns (<=2 unique values). Returns
    ``(df, transformed_columns)``.
    """
    df = df.copy()
    numeric, _ = split_column_types(df)
    skew = df[numeric].skew().abs()
    # Leave ordinal quality grades as clean integers (don't squash with log1p).
    cols = [
        c for c in numeric
        if c not in ORDINAL_QUALITY_COLS
        and skew[c] > threshold and df[c].min() >= 0 and df[c].nunique() > 2
    ]
    df[cols] = np.log1p(df[cols])
    return df, cols


def preprocess(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    drop_outliers: bool = True,
    log_skew: bool = True,
    encode: bool = True,
    cat_fill: str = CAT_FILL,
    skew_threshold: float = SKEW_THRESHOLD,
    target_encode_cols: list[str] | None = None,
    ordinal: bool = True,
    interactions: bool = True,
):
    """Run the full preprocessing pipeline on raw train/test frames.

    Steps (all flags default to the EDA-recommended behaviour):
      1. Drop known training outliers (``drop_outliers``).
      2. Extract the target ``y = log1p(SalePrice)`` and the test ``Id``s.
      3. Concatenate train + test *features* (no ``SalePrice`` -> no leak) so
         imputation and one-hot encoding are consistent across both.
      4. Domain feature engineering + "None"/0 fills (:func:`engineer_features`).
      5. Impute remaining numeric (median) and categorical (mode/label) values.
      6. log1p the most skewed numeric features (``log_skew``).
      7. One-hot encode categoricals (``encode``); the two frames are aligned
         to identical columns. Columns in ``target_encode_cols`` are left raw
         for a fold-aware target encoder downstream.
      8. Split back into train/test and return.

    Returns
    -------
    X_train, X_test : pd.DataFrame
        Processed feature matrices with identical columns.
    y : pd.Series
        ``log1p(SalePrice)`` aligned to ``X_train``.
    test_ids : pd.Series
        The test-set ``Id`` column, for building a submission.
    """
    train_df = train_df.copy()
    if drop_outliers:
        train_df = drop_training_outliers(train_df)

    y = np.log1p(train_df[TARGET]).reset_index(drop=True)
    test_ids = test_df[ID_COL].reset_index(drop=True)

    X_train = train_df.drop(columns=[c for c in (TARGET, ID_COL) if c in train_df.columns])
    X_test = test_df.drop(columns=[c for c in (TARGET, ID_COL) if c in test_df.columns])
    n_train = len(X_train)

    # Combine for consistent preprocessing (features only -> SalePrice cannot leak).
    combined = pd.concat([X_train, X_test], axis=0, ignore_index=True)

    combined = engineer_features(combined, ordinal=ordinal, interactions=interactions)
    combined = _fill_missing(combined, cat_fill=cat_fill)

    if log_skew:
        combined, _ = _log_skewed(combined, threshold=skew_threshold)

    if encode:
        # Hold out target-encode columns so a fold-aware encoder can fit them
        # inside the model pipeline (one-hot the rest). They stay as raw
        # categoricals in the returned frames.
        te_cols = [c for c in (target_encode_cols or []) if c in combined.columns]
        encode_cols = [c for c in combined.select_dtypes(exclude=[np.number]).columns
                       if c not in te_cols]
        combined = pd.get_dummies(combined, columns=encode_cols, dummy_na=False)

    X_train_out = combined.iloc[:n_train].reset_index(drop=True)
    X_test_out = combined.iloc[n_train:].reset_index(drop=True)
    return X_train_out, X_test_out, y, test_ids
