"""Train the validated blend and create a Kaggle submission."""

from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold

from .features import ID_COLUMN, load_or_build_historical_features
from .preprocessing import ApplicationFeatureEngineer, load_application_data
from .train import RANDOM_STATE

LIGHTGBM_WEIGHT = 0.35
CATBOOST_WEIGHT = 0.65
LIGHTGBM_ITERATIONS = (665, 739, 789)
CATBOOST_ITERATIONS = (1972, 1443, 1986)


def build_train_test_matrices(
    application_train: pd.DataFrame,
    application_test: pd.DataFrame,
    historical: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Apply identical feature engineering and historical joins."""
    y = application_train["TARGET"].astype("int8")
    train = application_train.drop(columns="TARGET")
    test = application_test.copy()

    engineer = ApplicationFeatureEngineer(skew_threshold=1.0)
    train = engineer.fit_transform(train)
    test = engineer.transform(test)

    train = train.merge(historical, on=ID_COLUMN, how="left")
    test = test.merge(historical, on=ID_COLUMN, how="left")
    train = train.drop(columns=ID_COLUMN)
    test = test.drop(columns=ID_COLUMN)
    return train, test, y


def prepare_lightgbm_frames(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Give train and test categorical columns one shared category vocabulary."""
    train = train.replace([np.inf, -np.inf], np.nan).copy()
    test = test.replace([np.inf, -np.inf], np.nan).copy()
    categorical = train.select_dtypes(
        include=["object", "string", "category"]
    ).columns
    for column in categorical:
        combined = pd.concat(
            [
                train[column].fillna("__MISSING__").astype(str),
                test[column].fillna("__MISSING__").astype(str),
            ],
            ignore_index=True,
        )
        categories = pd.Index(combined.unique())
        train[column] = pd.Categorical(
            train[column].fillna("__MISSING__").astype(str),
            categories=categories,
        )
        test[column] = pd.Categorical(
            test[column].fillna("__MISSING__").astype(str),
            categories=categories,
        )
    return train, test


def prepare_catboost_frames(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    train = train.replace([np.inf, -np.inf], np.nan).copy()
    test = test.replace([np.inf, -np.inf], np.nan).copy()
    categorical = train.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()
    for column in categorical:
        train[column] = train[column].fillna("__MISSING__").astype(str)
        test[column] = test[column].fillna("__MISSING__").astype(str)
    indices = [train.columns.get_loc(column) for column in categorical]
    return train, test, indices


def predict_blend(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: pd.Series,
    *,
    folds: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Average test predictions across the same folds used in validation."""
    cv = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    lightgbm_test = np.zeros(len(test), dtype=np.float64)
    catboost_test = np.zeros(len(test), dtype=np.float64)

    lgb_train, lgb_test = prepare_lightgbm_frames(train, test)
    cat_train, cat_test, cat_indices = prepare_catboost_frames(train, test)

    for fold, (train_index, _) in enumerate(cv.split(train, y), start=1):
        lgb_iterations = LIGHTGBM_ITERATIONS[fold - 1]
        lightgbm = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=lgb_iterations,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=40,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.05,
            reg_lambda=0.2,
            random_state=RANDOM_STATE + fold,
            n_jobs=-1,
            verbosity=-1,
        )
        lightgbm.fit(lgb_train.iloc[train_index], y.iloc[train_index])
        lightgbm_test += lightgbm.predict_proba(lgb_test)[:, 1] / folds
        print(f"LightGBM fold {fold}/{folds} complete", flush=True)

        cat_iterations = CATBOOST_ITERATIONS[fold - 1]
        catboost = CatBoostClassifier(
            iterations=cat_iterations,
            learning_rate=0.04,
            depth=7,
            loss_function="Logloss",
            eval_metric="AUC",
            l2_leaf_reg=5.0,
            random_seed=RANDOM_STATE + fold,
            thread_count=-1,
            verbose=False,
            allow_writing_files=False,
        )
        catboost.fit(
            cat_train.iloc[train_index],
            y.iloc[train_index],
            cat_features=cat_indices,
            verbose=False,
        )
        catboost_test += catboost.predict_proba(cat_test)[:, 1] / folds
        print(f"CatBoost fold {fold}/{folds} complete", flush=True)

    blend = LIGHTGBM_WEIGHT * lightgbm_test + CATBOOST_WEIGHT * catboost_test
    return blend, np.column_stack([lightgbm_test, catboost_test])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=Path("data/processed/historical_features.pkl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/exp-004_submission.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train, test = load_application_data(args.data_dir)
    historical = load_or_build_historical_features(args.data_dir, args.cache_path)
    X_train, X_test, y = build_train_test_matrices(train, test, historical)
    print(f"Train matrix: {X_train.shape}; test matrix: {X_test.shape}", flush=True)

    blend, components = predict_blend(X_train, X_test, y)
    submission = pd.DataFrame(
        {
            ID_COLUMN: test[ID_COLUMN],
            "TARGET": blend,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    np.save(args.output.with_suffix(".components.npy"), components)
    print(
        f"Wrote {args.output}: rows={len(submission)}, "
        f"min={blend.min():.6f}, max={blend.max():.6f}, mean={blend.mean():.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
