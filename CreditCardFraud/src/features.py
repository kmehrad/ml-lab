"""Light, leakage-free feature engineering for the fraud dataset.

The bulk of the signal lives in the anonymised PCA components ``V1``..``V28``,
which are already centred and decorrelated, so they pass through untouched. The
only raw columns with interpretable structure are ``Amount`` (heavy-tailed, many
zeros) and ``Time`` (seconds since the first transaction, spanning ~2 days):

* ``amount_log`` — ``log1p(Amount)`` tames the heavy tail for linear models.
* ``amount_zero`` — flag for the (genuinely present) zero-amount transactions.
* ``hour`` — hour-of-day ``(Time / 3600) % 24`` exposes any diurnal fraud rhythm
  without leaking absolute time.

The transformer is **stateless and row-wise** (``fit`` is a no-op), so fitting it
inside cross-validation cannot leak information between folds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

ENGINEERED_FEATURES: tuple[str, ...] = ("amount_log", "amount_zero", "hour")


class FraudFeatureEngineer(BaseEstimator, TransformerMixin):
    """Append a few interpretable features derived from ``Amount`` and ``Time``."""

    def __init__(self, add_engineered: bool = True) -> None:
        self.add_engineered = add_engineered

    def fit(self, X: pd.DataFrame, y=None) -> "FraudFeatureEngineer":  # noqa: N803
        # Stateless, but expose a fitted marker so sklearn's check_is_fitted
        # (used by Pipeline.transform) recognises a fitted instance.
        self.n_features_in_ = X.shape[1]
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        out = X.copy()
        if not self.add_engineered:
            return out
        out["amount_log"] = np.log1p(out["Amount"])
        out["amount_zero"] = (out["Amount"] == 0).astype("int8")
        out["hour"] = (out["Time"] / 3600.0) % 24.0
        return out
