"""Stacking / blending ensemble for House Prices.

Pipeline per base model::

    ColumnTransformer(TargetEncoder(Neighborhood) + passthrough)
      -> [StandardScaler for linear models]
      -> estimator

The ``TargetEncoder`` (sklearn, internal cross-fitting) lives *inside* the
pipeline, so under k-fold CV it is refit on each training fold only -- the
validation fold's target never influences the encoding (no leakage).

We compute out-of-fold (OOF) predictions for every base model on the same
folds, then combine them two ways:

* **NNLS blend** - non-negative least squares weights (``scipy.optimize.nnls``),
  the principled version of the hand-picked ``0.25`` average.
* **Ridge meta-learner** - classic stacking.

Both combiners are themselves cross-validated on the OOF matrix so the reported
blend score isn't optimistic. Run::

    python -m src.ensemble            # report CV
    python -m src.ensemble --submit   # report CV + submit best blend
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder

from . import data as data_mod
from . import evaluate as eval_mod
from . import features as feat_mod
from . import models as models_mod

warnings.filterwarnings("ignore")

N_SPLITS = 5
SEED = 42
BASE_MODELS = ["elasticnet", "lasso", "ridge", "gradient_boosting",
               "xgboost", "lightgbm", "catboost"]


def build_base_pipeline(name: str, te_cols: list[str]) -> Pipeline:
    """Wrap a base model with fold-aware target encoding (+ scaling if linear).

    With an empty ``te_cols`` the encoder is skipped (features are already fully
    numeric, e.g. one-hot ``Neighborhood``).
    """
    steps = []
    if te_cols:
        encoder = ColumnTransformer(
            [("te", TargetEncoder(target_type="continuous", random_state=SEED), te_cols)],
            remainder="passthrough",
        )
        steps.append(("encode", encoder))
    if name in models_mod.NEEDS_SCALING:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", models_mod.get_model(name)))
    return Pipeline(steps)


def oof_and_test(X, y, X_test, te_cols, names=BASE_MODELS, n_splits=N_SPLITS, seed=SEED):
    """Return (oof_df, test_df) of per-model predictions.

    ``oof_df[name]`` are leak-free OOF predictions on the train rows; the test
    predictions average the per-fold fitted models for stability.
    """
    cv = eval_mod.make_kfold(n_splits=n_splits, seed=seed)
    available = [n for n in names if n in models_mod.available_models()]
    oof = pd.DataFrame(index=range(len(X)), columns=available, dtype=float)
    test = pd.DataFrame(index=range(len(X_test)), columns=available, dtype=float)

    y = np.asarray(y)
    for name in available:
        oof_col = np.zeros(len(X))
        test_col = np.zeros(len(X_test))
        for tr_idx, val_idx in cv.split(X):
            pipe = build_base_pipeline(name, te_cols)
            pipe.fit(X.iloc[tr_idx], y[tr_idx])
            oof_col[val_idx] = pipe.predict(X.iloc[val_idx])
            test_col += pipe.predict(X_test) / n_splits
        oof[name] = oof_col
        test[name] = test_col
        print(f"  {name:<18} OOF RMSE {eval_mod.rmse(y, oof_col):.5f}")
    return oof, test


def nnls_weights(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Non-negative least squares weights, normalised to sum to 1."""
    w, _ = nnls(P, y)
    return w / w.sum() if w.sum() > 0 else np.full(P.shape[1], 1 / P.shape[1])


def cv_blend_score(oof: pd.DataFrame, y, method="nnls", n_splits=N_SPLITS, seed=SEED):
    """Honestly CV the combiner on the OOF matrix (weights refit per fold)."""
    P = oof.values
    y = np.asarray(y)
    cv = eval_mod.make_kfold(n_splits=n_splits, seed=seed)
    preds = np.zeros(len(y))
    for tr_idx, val_idx in cv.split(P):
        if method == "nnls":
            w = nnls_weights(P[tr_idx], y[tr_idx])
            preds[val_idx] = P[val_idx] @ w
        else:  # ridge meta-learner
            meta = RidgeCV(alphas=np.logspace(-3, 2, 20))
            meta.fit(P[tr_idx], y[tr_idx])
            preds[val_idx] = meta.predict(P[val_idx])
    return eval_mod.rmse(y, preds)


def run(submit: bool = False):
    print("Preprocessing (Neighborhood held raw for in-pipeline target encoding)...")
    train_df, test_df = data_mod.load_train(), data_mod.load_test()
    te_cols = feat_mod.TARGET_ENCODE_COLS
    X, X_test, y, test_ids = feat_mod.preprocess(
        train_df, test_df, target_encode_cols=te_cols
    )
    print(f"X: {X.shape}   target-encoded: {te_cols}   CV: {N_SPLITS}-fold seed={SEED}\n")

    print("Base model OOF scores:")
    oof, test = oof_and_test(X, y, X_test, te_cols)

    print("\nBlend CV scores (combiner refit per fold -> honest):")
    nnls_cv = cv_blend_score(oof, y, method="nnls")
    ridge_cv = cv_blend_score(oof, y, method="ridge")
    avg_cv = eval_mod.rmse(np.asarray(y), oof.mean(axis=1).values)
    print(f"  simple average     RMSE {avg_cv:.5f}")
    print(f"  NNLS blend         RMSE {nnls_cv:.5f}")
    print(f"  Ridge meta-stack   RMSE {ridge_cv:.5f}")

    # Final weights from full OOF (for the submission) + show them.
    w = nnls_weights(oof.values, np.asarray(y))
    print("\nNNLS weights:")
    for name, weight in sorted(zip(oof.columns, w), key=lambda t: -t[1]):
        print(f"  {name:<18} {weight:.3f}")

    best_name, best_cv = min(
        [("nnls", nnls_cv), ("ridge", ridge_cv), ("average", avg_cv)],
        key=lambda t: t[1],
    )
    print(f"\nBest combiner: {best_name}  (CV RMSE {best_cv:.5f})")

    if submit:
        if best_name == "nnls":
            test_pred_log = test.values @ w
        elif best_name == "ridge":
            meta = RidgeCV(alphas=np.logspace(-3, 2, 20)).fit(oof.values, np.asarray(y))
            test_pred_log = meta.predict(test.values)
        else:
            test_pred_log = test.mean(axis=1).values
        preds = np.expm1(test_pred_log)
        fname = f"submission_stack_{best_name}.csv"
        path = data_mod.make_submission(test_ids, preds, filename=fname)
        print(f"Wrote {path}  (pred range ${preds.min():,.0f}-${preds.max():,.0f})")
        return path, best_cv
    return None, best_cv


def main():
    ap = argparse.ArgumentParser(description="Stacking/blending ensemble.")
    ap.add_argument("--submit", action="store_true", help="Write the best blend submission.")
    args = ap.parse_args()
    run(submit=args.submit)


if __name__ == "__main__":
    main()
