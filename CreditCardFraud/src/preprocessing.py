"""Model-ready preprocessing pipelines.

After feature engineering every column is numeric, so the pipelines differ only
in whether they scale:

* ``"tree"``   — feature engineering only; values pass straight through. Consumed
  by LightGBM/XGBoost/CatBoost/HistGB, which are scale-invariant.
* ``"linear"`` — feature engineering plus ``RobustScaler`` on the raw-scale
  columns (``Time``, ``Amount`` and the engineered ones). ``V1``..``V28`` are
  already centred/unit-scale from the original PCA, so re-scaling them adds
  nothing; ``RobustScaler`` (median/IQR) is used because ``Amount`` is heavy-tailed
  with outliers. Yields a dense float matrix for LogisticRegression.

The data has no missing values (see ``reports/EDA_FINDINGS.md``), so neither
pipeline imputes. Both are fit on the training fold only inside cross-validation.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from .features import FraudFeatureEngineer

PreprocessorKind = str  # "tree" | "linear"

# Raw-scale columns worth scaling for linear models (V1..V28 already ~unit-scale).
SCALE_COLUMNS: tuple[str, ...] = ("Time", "Amount", "amount_log", "hour")


def build_preprocessor(
    kind: PreprocessorKind = "tree",
    *,
    add_engineered: bool = True,
) -> Pipeline:
    """Build a fit/transform pipeline for the requested model family."""
    if kind not in {"tree", "linear"}:
        raise ValueError(f"kind must be 'tree' or 'linear', got {kind!r}")

    steps = [("features", FraudFeatureEngineer(add_engineered=add_engineered))]
    if kind == "linear":
        scale_cols = [c for c in SCALE_COLUMNS if add_engineered or c in ("Time", "Amount")]
        scaler = ColumnTransformer(
            transformers=[("scale", RobustScaler(), scale_cols)],
            remainder="passthrough",
            verbose_feature_names_out=False,
        )
        scaler.set_output(transform="pandas")
        steps.append(("scale", scaler))
    return Pipeline(steps=steps)
