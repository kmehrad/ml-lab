# Experiments — BirdSpecies (MLSP 2013 Birds)

Run log. Each `src/train_clf.py` / `src/train_cnn.py` run appends a row and
writes artifacts to `experiments/artifacts/` (`{model}_oof.npy`,
`{model}_test.npy`, `{model}_metrics.json`), aligned via the shared 5-fold
`MultilabelStratifiedKFold` (seed 42).

Metric = **flattened-pair AUC** on out-of-fold predictions (pooled over all
`recording × species` pairs). Show OOF AUC and get approval before any LB
submission.

| ID | Date | Model | Features | OOF AUC | Public LB | Private LB | Notes |
|----|------|-------|----------|--------:|----------:|-----------:|-------|
| _(pending)_ | | | | | | | |
