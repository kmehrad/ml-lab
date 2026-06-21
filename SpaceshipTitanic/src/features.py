"""Preprocessing and feature engineering for Spaceship Titanic.

The pipeline turns the raw passenger records into model-ready features in two
stages:

1. :func:`engineer_features` — pure-pandas derivation of new columns and
   finding-based missing-value fixes. It also **drops identifier columns**
   (``PassengerId``, ``Name``, the raw ``Cabin`` string, and the travel
   ``Group`` id) so they are never used directly as features. Information they
   carry is kept only through engineered, non-identifying summaries
   (``GroupSize``, ``IsAlone``, ``Deck``, ``CabinNum``, ``Side``).
2. :func:`build_preprocessor` — a scikit-learn ``ColumnTransformer`` that
   imputes the remaining missing values and encodes each feature according to
   its type (see "Encoding strategy" below).

Decisions are grounded in ``notebooks/01_eda.ipynb``:

- **Cabin has real predictive power** and is *not* removed. Transport rate
  swings strongly by deck (B≈0.73, C≈0.68 vs E≈0.36, T≈0.20) and side
  (S≈0.56 vs P≈0.45). The raw "deck/num/side" string is high-cardinality and
  useless as-is, so it is split into ``Deck`` / ``CabinNum`` / ``Side``.
- **Spend columns** are right-skewed and a missing value means "spent nothing"
  (every CryoSleep passenger has zero spend), so missing spends are filled with
  ``0`` and the columns are ``log1p``-transformed.
- **Train/test share the same distribution**, so a fitted preprocessor
  transfers cleanly to the test set.

Encoding strategy
-----------------
- **One-hot** — low-cardinality nominal features:
  ``HomePlanet``, ``Destination``, ``Side``, ``CryoSleep``, ``VIP``.
- **Target encoding** — higher-cardinality nominal feature ``Deck`` (8 levels +
  ``Unknown``); sklearn's :class:`~sklearn.preprocessing.TargetEncoder`
  cross-fits internally to avoid leakage.
- **Ordinal** — supported via ``ordinal_features`` but empty by default: the
  dataset has no naturally ordered categoricals (deck order is not monotonic in
  the target, so ``Deck`` is target-encoded instead). Pass e.g.
  ``ordinal_features={"Deck": [...]}`` to opt in.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    TargetEncoder,
)
from sklearn.impute import SimpleImputer

from .data import ID_COL, TARGET

# Raw columns that are pure identifiers — never used as model inputs.
ID_LIKE_COLS = [ID_COL, "Name", "Cabin", "Group"]

SPEND_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

# Categorical columns that are genuinely missing (no "None" meaning) and are
# given an explicit ``Unknown`` category during engineering.
_UNKNOWN_FILL_COLS = ["HomePlanet", "Destination", "Deck", "Side"]

# --- Feature groups consumed by :func:`build_preprocessor` -------------------
NUMERIC_FEATURES = ["Age", "CabinNum", "GroupSize", "IsAlone"]
SPEND_FEATURES = SPEND_COLS + ["TotalSpend"]          # log1p-transformed
ONEHOT_FEATURES = ["HomePlanet", "Destination", "Side", "CryoSleep", "VIP"]
TARGET_FEATURES = ["Deck"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive features and apply finding-based missing-value fixes.

    Returns a new DataFrame with identifier columns removed. Works on both the
    training and test sets (it never references the target).
    """
    df = df.copy()

    # --- Cabin -> Deck / CabinNum / Side -------------------------------------
    cabin = df["Cabin"].str.split("/", expand=True)
    df["Deck"] = cabin[0]
    df["CabinNum"] = pd.to_numeric(cabin[1], errors="coerce")
    df["Side"] = cabin[2]

    # --- PassengerId ("gggg_pp") -> travel group size (non-identifying) -------
    df["Group"] = df[ID_COL].str.split("_").str[0]
    df["GroupSize"] = df["Group"].map(df["Group"].value_counts())
    df["IsAlone"] = (df["GroupSize"] == 1).astype(int)

    # --- Spend columns: missing means 0 ("spent nothing"); add total ---------
    df[SPEND_COLS] = df[SPEND_COLS].fillna(0.0)
    df["TotalSpend"] = df[SPEND_COLS].sum(axis=1)

    # --- Boolean flags -------------------------------------------------------
    # A passenger who spent anything cannot have been in CryoSleep.
    cryo = df["CryoSleep"].where(~(df["CryoSleep"].isna() & (df["TotalSpend"] > 0)), False)
    df["CryoSleep"] = cryo.fillna(False).astype(bool)
    df["VIP"] = df["VIP"].fillna(False).astype(bool)

    # --- Genuinely-missing categoricals get an explicit category -------------
    for col in _UNKNOWN_FILL_COLS:
        df[col] = df[col].fillna("Unknown")

    return df.drop(columns=ID_LIKE_COLS)


def build_preprocessor(
    *,
    scale_numeric: bool = False,
    numeric_features: list[str] | None = None,
    spend_features: list[str] | None = None,
    onehot_features: list[str] | None = None,
    target_features: list[str] | None = None,
    ordinal_features: dict[str, list] | None = None,
) -> ColumnTransformer:
    """Build the feature-encoding ``ColumnTransformer``.

    Expects a DataFrame already passed through :func:`engineer_features`.

    Parameters
    ----------
    scale_numeric:
        Standard-scale the numeric/spend features (useful for linear models;
        unnecessary for tree ensembles).
    numeric_features, spend_features, onehot_features, target_features:
        Override the default feature lists.
    ordinal_features:
        Mapping of ``column -> ordered category list`` for ordinal encoding.
        Empty by default (no naturally ordinal columns in this dataset).
    """
    numeric_features = numeric_features if numeric_features is not None else NUMERIC_FEATURES
    spend_features = spend_features if spend_features is not None else SPEND_FEATURES
    onehot_features = onehot_features if onehot_features is not None else ONEHOT_FEATURES
    target_features = target_features if target_features is not None else TARGET_FEATURES
    ordinal_features = ordinal_features or {}

    def with_scaler(*steps):
        steps = list(steps)
        if scale_numeric:
            steps.append(("scaler", StandardScaler()))
        return Pipeline(steps)

    # Numeric: median-impute (CabinNum/Age may be NaN), optionally scale.
    numeric_pipe = with_scaler(("impute", SimpleImputer(strategy="median")))

    # Spend: already 0-filled, so just log1p (then optional scale).
    spend_pipe = with_scaler(
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
    )

    # One-hot for low-cardinality nominal features.
    onehot_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    # Target encoding for higher-cardinality nominal features (e.g. Deck).
    # Pass a seeded CV generator so the internal cross-fitting (and thus the
    # overall CV score) is reproducible across runs.
    target_pipe = Pipeline([
        ("encode", TargetEncoder(
            target_type="binary",
            cv=KFold(n_splits=5, shuffle=True, random_state=42),
        )),
    ])

    transformers = [
        ("numeric", numeric_pipe, numeric_features),
        ("spend", spend_pipe, spend_features),
        ("onehot", onehot_pipe, onehot_features),
        ("target", target_pipe, target_features),
    ]

    for col, categories in ordinal_features.items():
        ord_pipe = Pipeline([
            ("encode", OrdinalEncoder(
                categories=[categories],
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )),
        ])
        transformers.append((f"ordinal_{col}", ord_pipe, [col]))

    pre = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
    # Emit a named DataFrame so downstream estimators keep feature names.
    return pre.set_output(transform="pandas")


def split_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split an engineered *training* frame into features ``X`` and target ``y``."""
    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET])
    return X, y
