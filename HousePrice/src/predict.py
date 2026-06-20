"""Generate the Kaggle submission from saved models.

Run from the project root (after ``python -m src.train``):

    python -m src.predict

Rebuilds the test matrices exactly as in training, blends the three models with the
saved NNLS weights, inverts the log target, and writes a validated submission CSV.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from . import config
from .data import basic_clean, drop_outliers, load_raw
from .features import make_linear_matrix, make_tree_matrix


def main() -> None:
    train_raw, test_raw = load_raw()
    # Build matrices on the same cleaned train/test so one-hot columns and category
    # levels line up with what the models were trained on.
    train = drop_outliers(basic_clean(train_raw))
    test = basic_clean(test_raw)

    _, Xte_tree, _ = make_tree_matrix(train, test)
    _, Xte_lin, _ = make_linear_matrix(train, test, scale=False)

    enet = joblib.load(config.MODELS / "enet.joblib")
    lgbm = joblib.load(config.MODELS / "lgbm.joblib")
    xgb = joblib.load(config.MODELS / "xgb.joblib")
    blend = joblib.load(config.MODELS / "blend.joblib")

    preds = {
        "enet": enet.predict(Xte_lin),
        "lgbm": lgbm.predict(Xte_tree),
        "xgb": xgb.predict(Xte_tree),
    }
    log_pred = sum(w * preds[n] for n, w in zip(blend["order"], blend["weights"]))
    sale_price = np.clip(np.expm1(log_pred), 0, None)

    submission = pd.DataFrame({
        config.ID_COL: test_raw[config.ID_COL].values,
        config.TARGET: sale_price,
    })

    _validate(submission)

    out = config.SUBMISSIONS / "submission.csv"
    submission.to_csv(out, index=False)
    print(f"wrote {out}  ({len(submission)} rows)")
    print(f"  blend weights: " + ", ".join(
        f"{n}={w:.3f}" for n, w in zip(blend["order"], blend["weights"])))
    print(f"  SalePrice  min {sale_price.min():,.0f}  "
          f"median {np.median(sale_price):,.0f}  max {sale_price.max():,.0f}")
    print(f"  (training blend CV RMSE: {blend['blend_rmse']:.5f})")


def _validate(submission: pd.DataFrame) -> None:
    sample = pd.read_csv(config.DATA_RAW / "sample_submission.csv")
    assert len(submission) == len(sample), (
        f"row count {len(submission)} != expected {len(sample)}")
    assert list(submission[config.ID_COL]) == list(sample[config.ID_COL]), \
        "Id values/order do not match sample_submission"
    assert submission[config.TARGET].notna().all(), "NaN predictions"
    assert (submission[config.TARGET] >= 0).all(), "negative predictions"


if __name__ == "__main__":
    main()
