"""Benchmark models across three tiers with 5-fold CV.

All models are evaluated with the same KFold (``n_splits=5, shuffle=True,
random_state=42``) on the ``log1p(SalePrice)`` target, so the reported RMSE is
the competition RMSLE. Run from the project root::

    python -m src.benchmark

The averaging ensemble is scored on *out-of-fold* predictions (each base model
is refit per fold) so its CV number is comparable to the individual models.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from . import data as data_mod
from . import evaluate as eval_mod
from . import features as feat_mod
from . import models as models_mod

warnings.filterwarnings("ignore")  # silence LightGBM/sklearn convergence chatter

N_SPLITS = 5
SEED = 42

TIER1 = ["ridge", "lasso", "elasticnet", "random_forest", "gradient_boosting"]
TIER2 = ["xgboost", "lightgbm", "catboost"]
# Tier 3: strong individual models + the averaging ensemble.
TIER3_MODELS = ["ridge", "lasso", "gradient_boosting", "xgboost", "lightgbm", "catboost"]
ENSEMBLE_WEIGHTS = {"ridge": 0.25, "xgboost": 0.25, "lightgbm": 0.25, "catboost": 0.25}


def _evaluate_single(name: str, X, y) -> dict:
    """5-fold CV RMSE for one model; skips cleanly if its lib is missing."""
    if name not in models_mod.available_models():
        return {"model": name, "rmse": np.nan, "std": np.nan, "note": "not installed"}
    scores = eval_mod.cross_validate_rmse(
        models_mod.get_model(name), X, y, n_splits=N_SPLITS, seed=SEED
    )
    return {"model": name, "rmse": scores.mean(), "std": scores.std(), "note": ""}


def _evaluate_ensemble(X, y) -> dict:
    """OOF-averaged ensemble RMSE using ENSEMBLE_WEIGHTS."""
    available = {n: w for n, w in ENSEMBLE_WEIGHTS.items()
                 if n in models_mod.available_models()}
    blended = np.zeros(len(y), dtype=float)
    for name, weight in available.items():
        oof = eval_mod.oof_predictions(
            lambda n=name: models_mod.get_model(n), X, y, n_splits=N_SPLITS, seed=SEED
        )
        blended += weight * oof
    # Renormalise in case a member was unavailable.
    blended /= sum(available.values())
    rmse = eval_mod.rmse(np.asarray(y), blended)
    members = "+".join(available)
    return {"model": "avg_ensemble", "rmse": rmse, "std": np.nan, "note": members}


def _print_table(title: str, rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    df = df.sort_values("rmse", na_position="last").reset_index(drop=True)
    print(f"\n=== {title} ===")
    for _, r in df.iterrows():
        rmse = "   n/a " if np.isnan(r["rmse"]) else f"{r['rmse']:.5f}"
        std = "" if np.isnan(r["std"]) else f" +/- {r['std']:.5f}"
        note = f"   [{r['note']}]" if r["note"] else ""
        print(f"  {r['model']:<18} RMSE {rmse}{std}{note}")
    return df


def main():
    print("Loading + preprocessing (encode=False -> matrix shared by all models)...")
    train_df, test_df = data_mod.load_train(), data_mod.load_test()
    # encode=False keeps categoricals as strings; we one-hot once here so the
    # exact same numeric matrix feeds every model (incl. linear ones).
    X, _, y, _ = feat_mod.preprocess(train_df, test_df, encode=True)

    print(f"X: {X.shape}   target: log1p(SalePrice)   CV: {N_SPLITS}-fold, seed={SEED}")

    all_rows = []
    for title, names in [("Tier 1 - classic sklearn", TIER1),
                         ("Tier 2 - gradient boosting", TIER2)]:
        rows = [_evaluate_single(n, X, y) for n in names]
        _print_table(title, rows)
        all_rows.extend(rows)

    tier3_rows = [_evaluate_single(n, X, y) for n in TIER3_MODELS]
    tier3_rows.append(_evaluate_ensemble(X, y))
    _print_table("Tier 3 - best models + averaging ensemble", tier3_rows)

    return all_rows


if __name__ == "__main__":
    main()
