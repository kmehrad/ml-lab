"""End-to-end training pipeline.

Run from the project root, e.g.::

    python -m src.train --model lightgbm --cv 5
    python -m src.train --model ridge --submit

The script:
  1. loads train/test,
  2. applies feature engineering,
  3. builds a sklearn Pipeline (preprocess + model),
  4. cross-validates in log space (== RMSLE),
  5. fits on all training data and (optionally) writes a submission.
"""
from __future__ import annotations

import argparse

import numpy as np
from sklearn.pipeline import Pipeline

from . import data as data_mod
from . import evaluate as eval_mod
from . import features as feat_mod
from . import models as models_mod


def build_pipeline(model_name: str, X_engineered, **model_kwargs) -> Pipeline:
    """Compose preprocessing + model into a single Pipeline."""
    scale = model_name in models_mod.NEEDS_SCALING
    preprocessor = feat_mod.build_preprocessor(X_engineered, scale=scale)
    model = models_mod.get_model(model_name, **model_kwargs)
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def run(model_name: str = "gradient_boosting", n_splits: int = 5, submit: bool = False):
    print(f"Loading data... (model={model_name})")
    train_df = data_mod.load_train()
    test_df = data_mod.load_test()

    X_raw, y = data_mod.split_features_target(train_df)
    test_ids = test_df[data_mod.ID_COL]
    X_test_raw = test_df.drop(columns=[data_mod.ID_COL])

    # Feature engineering (same transform for train and test).
    X = feat_mod.engineer_features(X_raw)
    X_test = feat_mod.engineer_features(X_test_raw)

    # Train on log1p(SalePrice); RMSE in this space == RMSLE.
    y_log = np.log1p(y)

    pipe = build_pipeline(model_name, X)

    print(f"Cross-validating with {n_splits}-fold KFold...")
    scores = eval_mod.cross_validate_rmse(pipe, X, y_log, n_splits=n_splits)
    summary = eval_mod.summarize_scores(scores)
    print(f"CV RMSLE: {summary['mean']:.5f} +/- {summary['std']:.5f}")
    print("  folds:", [f"{s:.5f}" for s in summary["folds"]])

    print("Fitting on full training set...")
    pipe.fit(X, y_log)

    if submit:
        preds_log = pipe.predict(X_test)
        preds = np.expm1(preds_log)
        filename = f"submission_{model_name}.csv"
        out_path = data_mod.make_submission(test_ids, preds, filename=filename)
        print(f"Wrote submission -> {out_path}")

    return pipe, summary


def main():
    parser = argparse.ArgumentParser(description="Train a House Prices model.")
    parser.add_argument(
        "--model", default="gradient_boosting",
        choices=models_mod.available_models(),
        help="Model to train (depends on installed optional deps).",
    )
    parser.add_argument("--cv", type=int, default=5, help="Number of CV folds.")
    parser.add_argument("--submit", action="store_true", help="Write a submission CSV.")
    args = parser.parse_args()

    run(model_name=args.model, n_splits=args.cv, submit=args.submit)


if __name__ == "__main__":
    main()
