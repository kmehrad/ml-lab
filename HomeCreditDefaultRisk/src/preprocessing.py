"""Preprocessing and application-level feature engineering.

The pipeline is designed for ``application_train.csv`` and
``application_test.csv``. It keeps missing-value semantics explicit, adds
domain ratios, log-transforms non-negative skewed features, and applies
cardinality-aware categorical encoding.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, TargetEncoder

TARGET_COLUMN = "TARGET"
ID_COLUMN = "SK_ID_CURR"
EMPLOYMENT_SENTINEL = 365243
UNKNOWN_CATEGORY = "__UNKNOWN__"
MISSING_CATEGORY = "__MISSING__"

# The EDA found these to be the only application categories with more than
# 10 distinct values. TargetEncoder.fit_transform uses internal cross-fitting,
# which prevents a row's target from directly encoding its own category.
HIGH_CARDINALITY_CATEGORICALS = (
    "ORGANIZATION_TYPE",
    "OCCUPATION_TYPE",
)

CATEGORICAL_SENTINELS = {
    "CODE_GENDER": {"XNA": UNKNOWN_CATEGORY},
    "NAME_FAMILY_STATUS": {"Unknown": UNKNOWN_CATEGORY},
    "ORGANIZATION_TYPE": {"XNA": UNKNOWN_CATEGORY},
}


def _safe_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """Divide two series while treating zero and infinite results as missing."""
    result = numerator.div(denominator.replace(0, np.nan))
    return result.replace([np.inf, -np.inf], np.nan)


class ApplicationFeatureEngineer(BaseEstimator, TransformerMixin):
    """Clean application semantics and add deterministic derived features.

    Log-transform candidates are learned from the training data. Eligible
    columns must be non-negative, have more than 10 distinct values, and have
    absolute skewness above ``skew_threshold``. The raw feature is retained and
    a ``LOG1P_`` feature is appended.
    """

    def __init__(self, skew_threshold: float = 1.0) -> None:
        self.skew_threshold = skew_threshold

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> "ApplicationFeatureEngineer":
        engineered = self._add_semantic_features(self._validate_input(X))
        numeric = engineered.select_dtypes(include=np.number)

        excluded = {ID_COLUMN}
        self.log_columns_ = []
        for column in numeric.columns:
            values = numeric[column].replace([np.inf, -np.inf], np.nan).dropna()
            if (
                column not in excluded
                and values.nunique() > 10
                and not values.empty
                and values.min() >= 0
                and abs(values.skew()) > self.skew_threshold
            ):
                self.log_columns_.append(column)

        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "log_columns_"):
            raise RuntimeError("ApplicationFeatureEngineer must be fitted first.")

        engineered = self._add_semantic_features(self._validate_input(X))
        for column in self.log_columns_:
            if column in engineered:
                engineered[f"LOG1P_{column}"] = np.log1p(
                    engineered[column].clip(lower=0)
                )
        return engineered.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def _validate_input(X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("ApplicationFeatureEngineer requires a pandas DataFrame.")
        return X.copy()

    @staticmethod
    def _add_semantic_features(X: pd.DataFrame) -> pd.DataFrame:
        # DAYS_EMPLOYED=365243 is a documented anomaly, not a plausible duration.
        if "DAYS_EMPLOYED" in X:
            anomaly = X["DAYS_EMPLOYED"].eq(EMPLOYMENT_SENTINEL)
            X["DAYS_EMPLOYED_ANOMALY"] = anomaly.astype("int8")
            X.loc[anomaly, "DAYS_EMPLOYED"] = np.nan

        # A car age is structurally not applicable when the applicant owns no car.
        if {"FLAG_OWN_CAR", "OWN_CAR_AGE"}.issubset(X.columns):
            no_car = X["FLAG_OWN_CAR"].eq("N")
            X["OWN_CAR_AGE_NOT_APPLICABLE"] = no_car.astype("int8")
            X.loc[no_car, "OWN_CAR_AGE"] = 0.0

        # Preserve explicit unknown sentinels as categories distinct from NaN.
        for column, replacements in CATEGORICAL_SENTINELS.items():
            if column in X:
                X[column] = X[column].replace(replacements)

        # Human-readable time features; retain raw relative-day columns too.
        day_features = {
            "DAYS_BIRTH": "AGE_YEARS",
            "DAYS_EMPLOYED": "EMPLOYMENT_YEARS",
            "DAYS_REGISTRATION": "REGISTRATION_YEARS",
            "DAYS_ID_PUBLISH": "ID_PUBLISH_YEARS",
            "DAYS_LAST_PHONE_CHANGE": "PHONE_CHANGE_YEARS",
        }
        for source, destination in day_features.items():
            if source in X:
                X[destination] = (-X[source] / 365.25).clip(lower=0)

        ratios = {
            "CREDIT_INCOME_RATIO": ("AMT_CREDIT", "AMT_INCOME_TOTAL"),
            "ANNUITY_INCOME_RATIO": ("AMT_ANNUITY", "AMT_INCOME_TOTAL"),
            "CREDIT_ANNUITY_RATIO": ("AMT_CREDIT", "AMT_ANNUITY"),
            "GOODS_CREDIT_RATIO": ("AMT_GOODS_PRICE", "AMT_CREDIT"),
            "INCOME_PER_PERSON": ("AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS"),
        }
        for destination, (numerator, denominator) in ratios.items():
            if {numerator, denominator}.issubset(X.columns):
                X[destination] = _safe_ratio(X[numerator], X[denominator])

        if {"DAYS_EMPLOYED", "DAYS_BIRTH"}.issubset(X.columns):
            X["EMPLOYED_AGE_RATIO"] = _safe_ratio(
                X["DAYS_EMPLOYED"].abs(),
                X["DAYS_BIRTH"].abs(),
            )

        if {"CNT_CHILDREN", "CNT_FAM_MEMBERS"}.issubset(X.columns):
            X["CHILDREN_FAMILY_RATIO"] = _safe_ratio(
                X["CNT_CHILDREN"],
                X["CNT_FAM_MEMBERS"],
            )

        return X


def _low_cardinality_selector(X: pd.DataFrame) -> list[str]:
    categorical = X.select_dtypes(include=["object", "string", "category"]).columns
    return [
        column
        for column in categorical
        if column not in HIGH_CARDINALITY_CATEGORICALS
    ]


def _high_cardinality_selector(X: pd.DataFrame) -> list[str]:
    return [column for column in HIGH_CARDINALITY_CATEGORICALS if column in X]


def build_application_preprocessor(
    *,
    skew_threshold: float = 1.0,
    target_encoder_cv: int = 5,
    random_state: int = 42,
) -> Pipeline:
    """Build the complete application preprocessing pipeline.

    Missing-value policy:

    - no-car ``OWN_CAR_AGE`` becomes zero with an explicit applicability flag;
    - categorical NaN values become ``__MISSING__``;
    - explicit unknown sentinels remain ``__UNKNOWN__``;
    - other numeric values use median imputation with missingness indicators.

    Low-cardinality categories are one-hot encoded. The two high-cardinality
    application categories are target encoded with cross-fitting during
    training. Call ``fit_transform(X_train, y_train)`` or fit this preprocessor
    as part of a scikit-learn model pipeline to preserve leakage protection.
    """
    numeric_pipeline = Pipeline(
        steps=[
            (
                "impute",
                SimpleImputer(strategy="median", add_indicator=True),
            ),
        ]
    )
    low_cardinality_pipeline = Pipeline(
        steps=[
            (
                "impute",
                SimpleImputer(
                    strategy="constant",
                    fill_value=MISSING_CATEGORY,
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )
    high_cardinality_pipeline = Pipeline(
        steps=[
            (
                "impute",
                SimpleImputer(
                    strategy="constant",
                    fill_value=MISSING_CATEGORY,
                ),
            ),
            (
                "target_encode",
                TargetEncoder(
                    target_type="binary",
                    smooth="auto",
                    cv=StratifiedKFold(
                        n_splits=target_encoder_cv,
                        shuffle=True,
                        random_state=random_state,
                    ),
                ),
            ),
        ]
    )

    columns = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                make_column_selector(dtype_include=np.number),
            ),
            (
                "low_cardinality",
                low_cardinality_pipeline,
                _low_cardinality_selector,
            ),
            (
                "high_cardinality",
                high_cardinality_pipeline,
                _high_cardinality_selector,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    return Pipeline(
        steps=[
            (
                "feature_engineering",
                ApplicationFeatureEngineer(skew_threshold=skew_threshold),
            ),
            ("columns", columns),
        ]
    )


def split_application_target(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate ``TARGET`` and remove the non-predictive application ID."""
    if TARGET_COLUMN not in frame:
        raise ValueError(f"{TARGET_COLUMN!r} is missing from the training frame.")
    y = frame[TARGET_COLUMN].astype("int8")
    X = frame.drop(columns=[TARGET_COLUMN, ID_COLUMN], errors="ignore")
    return X, y


def prepare_application_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    skew_threshold: float = 1.0,
    target_encoder_cv: int = 5,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, pd.Series, Pipeline]:
    """Fit on training applications and transform train and test consistently."""
    X_train, y = split_application_target(train)
    X_test = test.drop(columns=[ID_COLUMN], errors="ignore")

    preprocessor = build_application_preprocessor(
        skew_threshold=skew_threshold,
        target_encoder_cv=target_encoder_cv,
        random_state=random_state,
    )
    train_features = preprocessor.fit_transform(X_train, y)
    test_features = preprocessor.transform(X_test)
    return train_features, test_features, y, preprocessor


def load_application_data(
    data_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the primary application train and test CSV files."""
    data_path = Path(data_dir)
    train = pd.read_csv(data_path / "application_train.csv")
    test = pd.read_csv(data_path / "application_test.csv")
    return train, test


def get_feature_names(preprocessor: Pipeline) -> Iterable[str]:
    """Return output names after the preprocessor has been fitted."""
    return preprocessor.named_steps["columns"].get_feature_names_out()
