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

**Submissions:**

| Model | OOF balanced acc | Public LB | Private LB |
|---|---:|---:|---:|
| exp-005 HistGradientBoosting (`src.submit`) | 0.96954 | 0.96699 | 0.96936 |
| exp-006 Stacked ensemble + threshold tuning (`src.ensemble`) | 0.97132 | 0.96952 | 0.97166 |
| exp-007 Hierarchical + multi-FE ensemble (`src.hierarchical`) | 0.97228 | 0.96990 | 0.97205 |

**exp-006 notes** — `src/ensemble.py`: calibrated base learners (LGBM/XGB/HistGB/CatBoost
+ one-vs-rest LGBM), equal blend and logistic stack, per-class weight optimization for
balanced accuracy. Ruled out as routes past the ~0.97 ceiling: more model capacity
(early-stopped deep LGBM ties the small one), one-vs-rest, regression-to-latent (worse),
external/original data (incompatible schema), and local determinism (1-NN only 0.63).
The ~0.980 leaderboard cluster (rank 500 of 4315 = 0.98006) appears to need a
publicly-shared data insight not reproducible from the competition files alone.

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
