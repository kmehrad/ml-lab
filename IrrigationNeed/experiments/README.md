# Experiment Log

Use one row per meaningful experiment. Keep generated artifacts in `artifacts/` and
trained estimators in `../models/`.

Metric is **balanced accuracy** (OOF). All runs: 19 raw + 7 engineered features,
5-fold stratified CV, seed 42, balanced class weights, full 630k train rows.

| ID | Date | Model | Validation | OOF balanced acc | Notes |
|---|---|---|---|---:|---|
| exp-001 | 2026-06-22 | Logistic regression | 5-fold stratified | 0.85861 | Baseline floor; over-favours `High`, weak on `Medium` |
| exp-002 | 2026-06-22 | LightGBM | 5-fold stratified | 0.96758 | Native categoricals, `class_weight="balanced"` |
| exp-003 | 2026-06-22 | XGBoost | 5-fold stratified | 0.96939 | `hist` + `enable_categorical`, balanced sample weights |
| exp-004 | 2026-06-22 | CatBoost | 5-fold stratified | 0.96779 | `auto_class_weights="Balanced"`; ~10x slower, no gain |
| exp-005 | 2026-06-22 | HistGradientBoosting | 5-fold stratified | 0.96954 | Best so far; best score-for-time with XGBoost |

Reproduce with `python -m src.train --model all`.

Suggested artifact naming:

```text
artifacts/exp-001_metrics.json
artifacts/exp-001_oof_predictions.csv
../models/exp-001_model.joblib
../outputs/exp-001_submission.csv
```

Record the metric, fold scheme, seed, and feature set for every modeling run. Validate
each generated submission against `sample_submission.csv` columns and row count before
uploading.
