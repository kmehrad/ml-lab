# Experiments — BirdSpecies (MLSP 2013 Birds)

Run log. Each `src/train_clf.py` / `src/train_cnn.py` run appends a row and
writes artifacts to `experiments/artifacts/` (`{model}_oof.npy`,
`{model}_test.npy`, `{model}_metrics.json`), aligned via the shared 5-fold
`MultilabelStratifiedKFold` (seed 42).

Metric = **pooled AUC** on out-of-fold predictions (one ROC-AUC over all
`file × class` pairs). Show OOF AUC and get approval before any LB submission.

| ID | Date | Model | Features | OOF AUC | Public LB | Private LB | Notes |
|----|------|-------|----------|--------:|----------:|-----------:|-------|
| logreg | 2026-07-01 | logreg | agg-MFCC(86) | 0.89515 |  |  | ranknorm=0.87843, macro=0.8778, 2s |
| rf | 2026-07-01 | rf | agg-MFCC(86) | 0.92981 |  |  | ranknorm=0.91335, macro=0.9130, 216s |
| lgbm | 2026-07-01 | lgbm | agg-MFCC(86) | 0.88545 |  |  | ranknorm=0.87974, macro=0.8781, 351s |
