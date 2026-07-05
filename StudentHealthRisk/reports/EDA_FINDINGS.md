# EDA findings — StudentHealthRisk (PS S6E7)

_To be filled in by Step 3 (`notebooks/eda_build.py`)._

## Confirmed at setup (from raw CSVs)
- **Target `health_condition`**, 3 classes, heavily imbalanced: `at-risk` 592,561 (85.9%),
  `unhealthy` 57,724 (8.4%), `fit` 39,803 (5.8%).
- 690,088 train / 295,753 test rows. 7 numeric + 6 categorical (3-level) features.
- **Missing values in nearly every column** (~449k cells total); no missing target.
- Metric: **Balanced Accuracy Score** (mean per-class recall) → decision-sensitive.

## Open questions for EDA
- Per-class feature signal (which features separate `fit`/`unhealthy` from `at-risk`?).
- Missingness patterns — is missingness informative (correlated with the target)?
- Train/test drift (KS on numeric, share-delta on categorical) → confirms StratifiedKFold is safe.
- Label-conflict ceiling from duplicate feature rows.
