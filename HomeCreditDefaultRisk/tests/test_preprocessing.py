"""Tests for application preprocessing and feature engineering."""

import numpy as np
import pandas as pd

from src.preprocessing import (
    ApplicationFeatureEngineer,
    prepare_application_features,
)


def _application_frame() -> pd.DataFrame:
    rows = 12
    return pd.DataFrame(
        {
            "SK_ID_CURR": np.arange(100001, 100001 + rows),
            "TARGET": [0, 1] * 6,
            "FLAG_OWN_CAR": ["N", "Y"] * 6,
            "OWN_CAR_AGE": [np.nan, 4, np.nan, 8, np.nan, 2] * 2,
            "DAYS_BIRTH": [-12000, -15000, -18000, -20000, -13000, -16000] * 2,
            "DAYS_EMPLOYED": [
                -1000,
                365243,
                -5000,
                -2000,
                365243,
                -4000,
            ]
            * 2,
            "DAYS_REGISTRATION": [-2000, -3000, -4000, -5000, -2500, -3500] * 2,
            "DAYS_ID_PUBLISH": [-1000, -2000, -3000, -4000, -1500, -2500] * 2,
            "DAYS_LAST_PHONE_CHANGE": [-100, -200, -300, -400, -150, -250] * 2,
            "AMT_INCOME_TOTAL": [
                10000,
                20000,
                40000,
                80000,
                160000,
                320000,
            ]
            * 2,
            "AMT_CREDIT": [5000, 15000, 30000, 60000, 120000, 240000] * 2,
            "AMT_ANNUITY": [500, 1500, 3000, 6000, 12000, 24000] * 2,
            "AMT_GOODS_PRICE": [4500, 14000, 28000, 55000, 110000, 220000] * 2,
            "SKEWED_AMOUNT": list(range(11)) + [10000],
            "CNT_FAM_MEMBERS": [1, 2, 3, 4, 2, 5] * 2,
            "CNT_CHILDREN": [0, 1, 2, 2, 0, 3] * 2,
            "EXT_SOURCE_1": [0.1, np.nan, 0.3, 0.4, np.nan, 0.6] * 2,
            "NAME_CONTRACT_TYPE": ["Cash loans", "Revolving loans"] * 6,
            "NAME_FAMILY_STATUS": [
                "Single",
                "Married",
                "Unknown",
                "Married",
                np.nan,
                "Single",
            ]
            * 2,
            "ORGANIZATION_TYPE": [f"Organization {i}" for i in range(rows)],
            "OCCUPATION_TYPE": [f"Occupation {i % 6}" for i in range(rows)],
        }
    )


def test_feature_engineering_handles_semantic_missing_values() -> None:
    frame = _application_frame().drop(columns="TARGET")
    engineer = ApplicationFeatureEngineer(skew_threshold=0.5)
    transformed = engineer.fit_transform(frame)

    no_car = transformed["FLAG_OWN_CAR"].eq("N")
    assert transformed.loc[no_car, "OWN_CAR_AGE"].eq(0).all()
    assert transformed.loc[no_car, "OWN_CAR_AGE_NOT_APPLICABLE"].eq(1).all()

    anomaly = frame["DAYS_EMPLOYED"].eq(365243)
    assert transformed.loc[anomaly, "DAYS_EMPLOYED"].isna().all()
    assert transformed.loc[anomaly, "DAYS_EMPLOYED_ANOMALY"].eq(1).all()
    assert "__UNKNOWN__" in transformed["NAME_FAMILY_STATUS"].values


def test_prepare_features_is_finite_and_schema_consistent() -> None:
    train = _application_frame()
    test = train.drop(columns="TARGET").copy()
    test.loc[0, "NAME_CONTRACT_TYPE"] = "Unseen contract"
    test.loc[1, "ORGANIZATION_TYPE"] = "Unseen organization"

    train_features, test_features, y, preprocessor = prepare_application_features(
        train,
        test,
        skew_threshold=0.5,
        target_encoder_cv=2,
    )

    assert train_features.shape[0] == len(train)
    assert test_features.shape[0] == len(test)
    assert train_features.shape[1] == test_features.shape[1]
    assert np.isfinite(train_features).all()
    assert np.isfinite(test_features).all()
    assert y.tolist() == train["TARGET"].tolist()

    names = preprocessor.named_steps["columns"].get_feature_names_out()
    assert len(names) == train_features.shape[1]
    assert any("ORGANIZATION_TYPE" in name for name in names)
    assert not any(
        "ORGANIZATION_TYPE_Organization" in name
        for name in names
    )


def test_log_features_are_learned_from_training_data() -> None:
    frame = _application_frame().drop(columns="TARGET")
    engineer = ApplicationFeatureEngineer(skew_threshold=0.5)
    transformed = engineer.fit_transform(frame)

    log_columns = [column for column in transformed if column.startswith("LOG1P_")]
    assert log_columns
    assert all(np.isfinite(transformed[column].dropna()).all() for column in log_columns)
