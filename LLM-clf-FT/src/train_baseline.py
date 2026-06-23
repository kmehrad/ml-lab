"""Phase 1 baseline: TF-IDF/SVD + engineered features -> LightGBM (3-class).

Uses the shared stratified folds (data/processed/folds.parquet) so the reported
out-of-fold log loss is comparable to later models. The feature builder is fit
unsupervised on the full train text (no label leakage), then each fold trains a
LightGBM with early stopping. Test predictions average the fold models.

Run:  python -m src.train_baseline
Outputs:
  data/processed/oof_baseline.npy        OOF probabilities (n, 3)
  data/submissions/submission_baseline.csv
"""
from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np
from sklearn.metrics import log_loss

from .data import PROC, SUB, TARGETS, SEED, load_test, load_train
from .features import FeatureBuilder

PARAMS = dict(
    objective="multiclass",
    num_class=3,
    metric="multi_logloss",
    learning_rate=0.03,
    num_leaves=63,
    feature_fraction=0.7,
    bagging_fraction=0.8,
    bagging_freq=1,
    min_child_samples=50,
    lambda_l2=1.0,
    seed=SEED,
    num_threads=0,
    verbose=-1,
)
NUM_ROUNDS = 3000
EARLY_STOP = 100


def main() -> None:
    t0 = time.time()
    train = load_train(with_folds=True)
    test = load_test()
    y = train[TARGETS].values.argmax(1)

    fb = FeatureBuilder(seed=SEED)
    print("building features ...")
    X = fb.fit_transform(train).values.astype(np.float32)
    X_test = fb.transform(test).values.astype(np.float32)
    print(f"  X={X.shape}  X_test={X_test.shape}  ({time.time()-t0:.0f}s)")

    folds = train["fold"].values
    oof = np.zeros((len(train), 3), dtype=float)
    test_pred = np.zeros((len(test), 3), dtype=float)
    n_folds = int(folds.max()) + 1

    for f in range(n_folds):
        tr, va = folds != f, folds == f
        dtr = lgb.Dataset(X[tr], y[tr])
        dva = lgb.Dataset(X[va], y[va])
        model = lgb.train(
            PARAMS, dtr, num_boost_round=NUM_ROUNDS, valid_sets=[dva],
            callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                       lgb.log_evaluation(0)],
        )
        oof[va] = model.predict(X[va], num_iteration=model.best_iteration)
        test_pred += model.predict(X_test, num_iteration=model.best_iteration) / n_folds
        ll = log_loss(y[va], oof[va], labels=[0, 1, 2])
        print(f"  fold {f}: best_iter={model.best_iteration:4d}  logloss={ll:.5f}")

    oof_ll = log_loss(y, oof, labels=[0, 1, 2])
    print(f"\nOOF log loss: {oof_ll:.5f}   (prior baseline 1.09723)")

    PROC.mkdir(parents=True, exist_ok=True)
    SUB.mkdir(parents=True, exist_ok=True)
    np.save(PROC / "oof_baseline.npy", oof)

    sub = test[["id"]].copy()
    sub[TARGETS] = test_pred
    sub.to_csv(SUB / "submission_baseline.csv", index=False)
    print(f"wrote {SUB / 'submission_baseline.csv'}  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
