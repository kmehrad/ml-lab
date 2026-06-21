"""Cross-validated Home Credit baseline and boosted-tree experiments."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import ID_COLUMN, load_or_build_historical_features
from .preprocessing import (
    ApplicationFeatureEngineer,
    build_application_preprocessor,
    load_application_data,
    split_application_target,
)

RANDOM_STATE = 42


@dataclass
class ExperimentResult:
    experiment_id: str
    model: str
    features: str
    folds: int
    fold_auc: list[float]
    mean_auc: float
    std_auc: float
    oof_auc: float
    log_loss: float
    brier_score: float
    runtime_seconds: float
    notes: str


def _metrics(
    experiment_id: str,
    model: str,
    features: str,
    fold_auc: list[float],
    y: pd.Series,
    oof: np.ndarray,
    runtime_seconds: float,
    notes: str,
) -> ExperimentResult:
    return ExperimentResult(
        experiment_id=experiment_id,
        model=model,
        features=features,
        folds=len(fold_auc),
        fold_auc=[round(score, 6) for score in fold_auc],
        mean_auc=float(np.mean(fold_auc)),
        std_auc=float(np.std(fold_auc)),
        oof_auc=float(roc_auc_score(y, oof)),
        log_loss=float(log_loss(y, oof)),
        brier_score=float(brier_score_loss(y, oof)),
        runtime_seconds=float(runtime_seconds),
        notes=notes,
    )


def _folds(y: pd.Series, n_splits: int) -> StratifiedKFold:
    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )


def run_logistic_regression(
    application_train: pd.DataFrame,
    *,
    n_splits: int,
) -> tuple[ExperimentResult, np.ndarray]:
    """Run the application-only linear sanity baseline without CV leakage."""
    X, y = split_application_target(application_train)
    cv = _folds(y, n_splits)
    oof = np.zeros(len(y), dtype=np.float64)
    fold_auc: list[float] = []
    started = time.perf_counter()

    base_model = Pipeline(
        steps=[
            ("preprocess", build_application_preprocessor(target_encoder_cv=5)),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=0.1,
                    class_weight=None,
                    max_iter=200,
                    solver="lbfgs",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    for fold, (train_index, valid_index) in enumerate(cv.split(X, y), start=1):
        model = clone(base_model)
        model.fit(X.iloc[train_index], y.iloc[train_index])
        predictions = model.predict_proba(X.iloc[valid_index])[:, 1]
        oof[valid_index] = predictions
        score = roc_auc_score(y.iloc[valid_index], predictions)
        fold_auc.append(score)
        print(f"logistic fold {fold}: AUC={score:.6f}", flush=True)

    runtime = time.perf_counter() - started
    result = _metrics(
        "exp-001",
        "Logistic regression",
        "Application table; semantic cleaning, ratios, log1p, OHE, target encoding",
        fold_auc,
        y,
        oof,
        runtime,
        "Sanity baseline. Preprocessing is refit inside each outer fold.",
    )
    return result, oof


def build_tree_matrix(
    application_train: pd.DataFrame,
    historical: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Combine engineered application features with applicant-level history."""
    y = application_train["TARGET"].astype("int8")
    application = application_train.drop(columns="TARGET")
    engineer = ApplicationFeatureEngineer(skew_threshold=1.0)
    application = engineer.fit_transform(application)
    combined = application.merge(historical, on=ID_COLUMN, how="left")
    combined = combined.drop(columns=ID_COLUMN)
    return combined, y


def _prepare_lightgbm_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    categorical = prepared.select_dtypes(
        include=["object", "string", "category"]
    ).columns
    for column in categorical:
        prepared[column] = prepared[column].fillna("__MISSING__").astype("category")
    return prepared.replace([np.inf, -np.inf], np.nan)


def run_lightgbm(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_splits: int,
) -> tuple[ExperimentResult, np.ndarray]:
    prepared = _prepare_lightgbm_frame(X)
    cv = _folds(y, n_splits)
    oof = np.zeros(len(y), dtype=np.float64)
    fold_auc: list[float] = []
    best_iterations: list[int] = []
    started = time.perf_counter()

    for fold, (train_index, valid_index) in enumerate(
        cv.split(prepared, y),
        start=1,
    ):
        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=3000,
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
        model.fit(
            prepared.iloc[train_index],
            y.iloc[train_index],
            eval_set=[(prepared.iloc[valid_index], y.iloc[valid_index])],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(150, verbose=False)],
        )
        predictions = model.predict_proba(
            prepared.iloc[valid_index],
            num_iteration=model.best_iteration_,
        )[:, 1]
        oof[valid_index] = predictions
        score = roc_auc_score(y.iloc[valid_index], predictions)
        fold_auc.append(score)
        best_iterations.append(int(model.best_iteration_))
        print(
            f"lightgbm fold {fold}: AUC={score:.6f}, "
            f"best_iteration={model.best_iteration_}",
            flush=True,
        )

    runtime = time.perf_counter() - started
    result = _metrics(
        "exp-002",
        "LightGBM",
        "Engineered application features plus all historical aggregates",
        fold_auc,
        y,
        oof,
        runtime,
        f"Native categorical handling; best iterations={best_iterations}.",
    )
    return result, oof


def _prepare_catboost_frame(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, list[int]]:
    prepared = frame.replace([np.inf, -np.inf], np.nan).copy()
    categorical = prepared.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()
    for column in categorical:
        prepared[column] = prepared[column].fillna("__MISSING__").astype(str)
    categorical_indices = [prepared.columns.get_loc(column) for column in categorical]
    return prepared, categorical_indices


def run_catboost(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_splits: int,
) -> tuple[ExperimentResult, np.ndarray]:
    prepared, categorical_indices = _prepare_catboost_frame(X)
    cv = _folds(y, n_splits)
    oof = np.zeros(len(y), dtype=np.float64)
    fold_auc: list[float] = []
    best_iterations: list[int] = []
    started = time.perf_counter()

    for fold, (train_index, valid_index) in enumerate(
        cv.split(prepared, y),
        start=1,
    ):
        model = CatBoostClassifier(
            iterations=2500,
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
        model.fit(
            prepared.iloc[train_index],
            y.iloc[train_index],
            cat_features=categorical_indices,
            eval_set=(prepared.iloc[valid_index], y.iloc[valid_index]),
            early_stopping_rounds=150,
            verbose=False,
        )
        predictions = model.predict_proba(prepared.iloc[valid_index])[:, 1]
        oof[valid_index] = predictions
        score = roc_auc_score(y.iloc[valid_index], predictions)
        fold_auc.append(score)
        best_iterations.append(int(model.get_best_iteration()))
        print(
            f"catboost fold {fold}: AUC={score:.6f}, "
            f"best_iteration={model.get_best_iteration()}",
            flush=True,
        )

    runtime = time.perf_counter() - started
    result = _metrics(
        "exp-003",
        "CatBoost",
        "Engineered application features plus all historical aggregates",
        fold_auc,
        y,
        oof,
        runtime,
        f"Native categorical handling; best iterations={best_iterations}.",
    )
    return result, oof


def find_best_blend(
    y: pd.Series,
    lightgbm_oof: np.ndarray,
    catboost_oof: np.ndarray,
    *,
    n_splits: int,
) -> tuple[ExperimentResult, np.ndarray]:
    """Select a blend weight using OOF ROC AUC."""
    candidates = np.linspace(0.0, 1.0, 21)
    scores = {}
    for lightgbm_weight in candidates:
        blend = (
            lightgbm_weight * lightgbm_oof
            + (1.0 - lightgbm_weight) * catboost_oof
        )
        scores[float(lightgbm_weight)] = roc_auc_score(y, blend)

    best_weight = max(scores, key=scores.get)
    oof = best_weight * lightgbm_oof + (1.0 - best_weight) * catboost_oof
    fold_auc = []
    for _, valid_index in _folds(y, n_splits).split(np.zeros(len(y)), y):
        fold_auc.append(roc_auc_score(y.iloc[valid_index], oof[valid_index]))

    result = _metrics(
        "exp-004",
        "LightGBM/CatBoost blend",
        "OOF probability blend of exp-002 and exp-003",
        fold_auc,
        y,
        oof,
        0.0,
        f"OOF-selected LightGBM weight={best_weight:.2f}; "
        f"CatBoost weight={1.0 - best_weight:.2f}.",
    )
    return result, oof


def _save_results(
    results: list[ExperimentResult],
    predictions: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = [asdict(result) for result in results]
    (output_dir / "model_results.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "experiment_id": [result.experiment_id for result in results],
            "model": [result.model for result in results],
            "mean_auc": [result.mean_auc for result in results],
            "std_auc": [result.std_auc for result in results],
            "oof_auc": [result.oof_auc for result in results],
            "log_loss": [result.log_loss for result in results],
            "brier_score": [result.brier_score for result in results],
            "runtime_seconds": [result.runtime_seconds for result in results],
            "notes": [result.notes for result in results],
        }
    ).to_csv(output_dir / "model_results.csv", index=False)
    for experiment_id, oof in predictions.items():
        np.save(output_dir / f"{experiment_id}_oof.npy", oof)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=Path("data/processed/historical_features.pkl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/artifacts"),
    )
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["logistic", "lightgbm", "catboost"],
        default=["logistic", "lightgbm", "catboost"],
    )
    parser.add_argument("--force-features", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train, _ = load_application_data(args.data_dir)
    results: list[ExperimentResult] = []
    predictions: dict[str, np.ndarray] = {}

    if "logistic" in args.models:
        result, oof = run_logistic_regression(train, n_splits=args.folds)
        results.append(result)
        predictions[result.experiment_id] = oof

    tree_models = {"lightgbm", "catboost"}.intersection(args.models)
    if tree_models:
        historical = load_or_build_historical_features(
            args.data_dir,
            args.cache_path,
            force=args.force_features,
        )
        X_tree, y = build_tree_matrix(train, historical)
        print(f"tree matrix: {X_tree.shape}", flush=True)

        if "lightgbm" in args.models:
            result, oof = run_lightgbm(X_tree, y, n_splits=args.folds)
            results.append(result)
            predictions[result.experiment_id] = oof
        if "catboost" in args.models:
            result, oof = run_catboost(X_tree, y, n_splits=args.folds)
            results.append(result)
            predictions[result.experiment_id] = oof

        if {"exp-002", "exp-003"}.issubset(predictions):
            result, oof = find_best_blend(
                y,
                predictions["exp-002"],
                predictions["exp-003"],
                n_splits=args.folds,
            )
            results.append(result)
            predictions[result.experiment_id] = oof

    _save_results(results, predictions, args.output_dir)
    print(pd.DataFrame([asdict(result) for result in results]).to_string(index=False))


if __name__ == "__main__":
    main()
