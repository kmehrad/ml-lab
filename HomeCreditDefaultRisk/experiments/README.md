# Experiment Log

Use one row per meaningful experiment. Keep generated artifacts in `artifacts/` and
trained estimators in `../models/`.

| ID | Date | Features | Model | Validation | OOF ROC AUC | Notes |
|---|---|---|---|---|---:|---|
| exp-001 | 2026-06-21 | Application preprocessing | Logistic regression | 3-fold stratified CV, seed 42 | 0.751271 | Sanity baseline |
| exp-002 | 2026-06-21 | Application + 262 historical aggregates | LightGBM | 3-fold stratified CV, seed 42 | 0.788298 | Mean best iteration 731 |
| exp-003 | 2026-06-21 | Application + 262 historical aggregates | CatBoost | 3-fold stratified CV, seed 42 | 0.790853 | Stronger but substantially slower |
| exp-004 | 2026-06-21 | exp-002/exp-003 OOF predictions | Probability blend | Same OOF folds | 0.792249 | 35% LightGBM, 65% CatBoost |

Suggested artifact naming:

```text
artifacts/exp-001_metrics.json
artifacts/exp-001_oof_predictions.csv
../models/exp-001_model.joblib
../outputs/exp-001_submission.csv
```

Machine-readable metrics and OOF arrays are generated under `artifacts/` by:

```bash
python -m src.train --folds 3
```

The blend weight is selected from increments of 0.05 using OOF ROC AUC. This
selection can make the reported blend score mildly optimistic; confirm the
weight with repeated folds before using it as a final estimate.
