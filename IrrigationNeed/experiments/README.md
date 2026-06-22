# Experiment Log

Use one row per meaningful experiment. Keep generated artifacts in `artifacts/` and
trained estimators in `../models/`.

| ID | Date | Features | Model | Validation | OOF Metric | Notes |
|---|---|---|---|---|---:|---|
| _none yet_ | | | | | | |

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
