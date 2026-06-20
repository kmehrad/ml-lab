"""Feature engineering and per-model-family matrix assembly.

Two representations are produced from the same cleaned data, because the model
families want opposite things:

* **Tree matrix** (LightGBM / XGBoost): raw numerics + ordinals as ranked ints +
  nominals as pandas ``category`` dtype. Trees are invariant to monotone
  transforms and ingest categories natively, so scaling / one-hot would only add
  width and noise.
* **Linear matrix** (ElasticNet): skew-corrected + standardized numerics +
  one-hot nominals. Linear models need comparable scales and symmetric features.

Both consume the cleaned frames from :func:`src.data.basic_clean`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from . import config

# --- Ordinal encodings (rankings verified against data_description.txt) -----

# Ex > Gd > TA > Fa > Po > None(absent). None == 0 so "no feature" < worst grade.
_QUALITY = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
_QUALITY_COLS = [
    "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
    "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC",
]

ORDINAL_MAPS: dict[str, dict[str, int]] = {col: _QUALITY for col in _QUALITY_COLS}
ORDINAL_MAPS.update({
    "BsmtExposure": {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4},
    "BsmtFinType1": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "BsmtFinType2": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "GarageFinish": {"None": 0, "Unf": 1, "RFn": 2, "Fin": 3},
    "Functional": {"Sal": 0, "Sev": 1, "Maj2": 2, "Maj1": 3, "Mod": 4, "Min2": 5, "Min1": 6, "Typ": 7},
    "LotShape": {"IR3": 0, "IR2": 1, "IR1": 2, "Reg": 3},
    "LandSlope": {"Sev": 0, "Mod": 1, "Gtl": 2},
    "PavedDrive": {"N": 0, "P": 1, "Y": 2},
    "CentralAir": {"N": 0, "Y": 1},  # binary
})

# Columns dropped for being near-constant (no signal).
DROP_COLS = ["Utilities"]


# --- Feature engineering ----------------------------------------------------

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add size/age aggregate features. Pure: returns a new frame."""
    df = df.copy()

    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    df["TotalBath"] = (df["FullBath"] + 0.5 * df["HalfBath"]
                       + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"])
    df["TotalPorchSF"] = (df["OpenPorchSF"] + df["EnclosedPorch"]
                          + df["3SsnPorch"] + df["ScreenPorch"] + df["WoodDeckSF"])

    # YrSold is nominal as a standalone column, but the *age* derived from it is
    # the real signal, so use it numerically here.
    yr_sold = df["YrSold"].astype(int)
    df["HouseAge"] = yr_sold - df["YearBuilt"]
    df["SinceRemod"] = yr_sold - df["YearRemodAdd"]
    df["IsRemodeled"] = (df["YearBuilt"] != df["YearRemodAdd"]).astype(int)

    df["HasPool"] = (df["PoolArea"] > 0).astype(int)
    df["Has2ndFloor"] = (df["2ndFlrSF"] > 0).astype(int)
    df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
    df["HasBsmt"] = (df["TotalBsmtSF"] > 0).astype(int)
    df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)
    df["HasPorch"] = (df["TotalPorchSF"] > 0).astype(int)

    return df


def encode_ordinals(df: pd.DataFrame) -> pd.DataFrame:
    """Map ordinal/binary categoricals to ranked integers. Returns a new frame."""
    df = df.copy()
    for col, mapping in ORDINAL_MAPS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).astype("int64")
    return df


def _prep(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Shared steps: engineer features, encode ordinals, drop near-constant cols,
    split off the log target. Returns feature frames (no Id) and y."""
    y = np.log1p(train[config.TARGET]).rename(config.TARGET)

    def transform(df: pd.DataFrame) -> pd.DataFrame:
        df = add_features(df)
        df = encode_ordinals(df)
        df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
        df = df.drop(columns=[config.ID_COL])
        if config.TARGET in df.columns:
            df = df.drop(columns=[config.TARGET])
        return df

    return transform(train), transform(test), y


def _nominal_cols(df: pd.DataFrame) -> list[str]:
    """Remaining non-numeric columns after ordinal encoding == nominal ones."""
    return df.select_dtypes(exclude="number").columns.tolist()


def make_tree_matrix(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Matrix for LightGBM/XGBoost: raw numerics, ranked-int ordinals, nominals as
    ``category`` dtype with levels aligned across train and test."""
    Xtr, Xte, y = _prep(train, test)
    nominal = _nominal_cols(Xtr)

    # Fit category levels on the union so train and test share identical codes.
    for col in nominal:
        levels = pd.Index(
            pd.concat([Xtr[col], Xte[col]], ignore_index=True).astype("string").unique()
        )
        dtype = pd.CategoricalDtype(categories=levels.dropna())
        Xtr[col] = Xtr[col].astype("string").astype(dtype)
        Xte[col] = Xte[col].astype("string").astype(dtype)

    return Xtr, Xte, y


def make_linear_matrix(
    train: pd.DataFrame, test: pd.DataFrame, skew_thresh: float = 0.75,
    scale: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Matrix for ElasticNet: log1p on skewed numerics, one-hot nominals (aligned).

    When ``scale=True`` (default) a ``StandardScaler`` is fit on train and applied
    to both sets. Pass ``scale=False`` to get the unscaled one-hot matrix so the
    estimator can own scaling inside a CV-fold ``Pipeline`` (no leakage).
    """
    Xtr, Xte, y = _prep(train, test)
    nominal = _nominal_cols(Xtr)
    numeric = [c for c in Xtr.columns if c not in nominal]

    # 1. Reduce right-skew on numeric features (skew measured on train).
    skewed = [c for c in numeric if abs(Xtr[c].skew()) > skew_thresh]
    for col in skewed:
        Xtr[col] = np.log1p(Xtr[col])
        Xte[col] = np.log1p(Xte[col])

    # 2. One-hot the nominals; align test to train's columns.
    Xtr = pd.get_dummies(Xtr, columns=nominal, drop_first=True)
    Xte = pd.get_dummies(Xte, columns=nominal, drop_first=True)
    Xtr, Xte = Xtr.align(Xte, join="left", axis=1, fill_value=0)
    Xte = Xte[Xtr.columns]  # identical order

    if not scale:
        return Xtr, Xte, y

    # 3. Standardize everything (fit on train only).
    scaler = StandardScaler()
    Xtr_s = pd.DataFrame(scaler.fit_transform(Xtr), columns=Xtr.columns, index=Xtr.index)
    Xte_s = pd.DataFrame(scaler.transform(Xte), columns=Xtr.columns, index=Xte.index)

    return Xtr_s, Xte_s, y
