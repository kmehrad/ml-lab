"""Training pipeline entry point for Spaceship Titanic.

Ties together :mod:`data`, :mod:`features`, :mod:`models`, and :mod:`evaluate`
to cross-validate a classifier, fit it on the full training set, predict on the
test set, and write a Kaggle submission to ``data/submissions``.

Usage::

    python -m src.train                 # default model (hgb)
    python -m src.train --model lgbm
    python -m src.train --model rf --n-splits 10
"""
from __future__ import annotations

import argparse

import pandas as pd

from . import data
from .evaluate import cross_validate, summarize_scores
from .features import engineer_train_test, split_X_y
from .models import available_models, get_model


def run(
    model_name: str = "hgb",
    n_splits: int = 5,
    output: str | None = None,
    model=None,
) -> pd.DataFrame:
    """Cross-validate, fit on all of train, predict on test, write a submission.

    Pass a pre-built ``model`` (e.g. an Optuna-tuned pipeline) to override the
    default :func:`models.get_model` estimator. Returns the submission DataFrame.
    """
    # --- Load & engineer (jointly, so group/family stats span both sets) -----
    raw_train = data.load_train()
    raw_test = data.load_test()
    test_ids = raw_test[data.ID_COL]  # keep before engineering drops identifiers

    train_eng, X_test = engineer_train_test(raw_train, raw_test)
    X, y = split_X_y(train_eng)

    model = model if model is not None else get_model(model_name)

    # --- Cross-validate -------------------------------------------------------
    scores = cross_validate(model, X, y, n_splits=n_splits)
    print(f"Model: {model_name}")
    print(f"CV accuracy: {summarize_scores(scores)}")

    # --- Fit on full train, predict on test ----------------------------------
    model.fit(X, y)
    preds = model.predict(X_test).astype(bool)  # Kaggle expects True/False

    submission = pd.DataFrame({data.ID_COL: test_ids, data.TARGET: preds})

    # --- Write submission ----------------------------------------------------
    data.SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = output or (data.SUBMISSIONS_DIR / f"submission_{model_name}.csv")
    submission.to_csv(out_path, index=False)
    print(f"Wrote {len(submission)} predictions to {out_path}")

    return submission


def main() -> None:
    """Parse CLI arguments and run the training pipeline."""
    parser = argparse.ArgumentParser(description="Train a Spaceship Titanic model and write a submission.")
    parser.add_argument(
        "--model", default="hgb", choices=available_models(),
        help="Which classifier to use (default: hgb).",
    )
    parser.add_argument("--n-splits", type=int, default=5, help="CV folds (default: 5).")
    parser.add_argument("--output", default=None, help="Submission CSV path (default: data/submissions/submission_<model>.csv).")
    args = parser.parse_args()

    run(model_name=args.model, n_splits=args.n_splits, output=args.output)


if __name__ == "__main__":
    main()
