# Experiments — BirdSpecies (NIPS4B 2013 Bird Challenge)

Run log. Each `src/train_clf.py` / `src/train_cnn.py` run appends a row and
writes artifacts to `experiments/artifacts/` (`{model}_oof.npy`,
`{model}_test.npy`, `{model}_metrics.json`), aligned via the shared 5-fold
`MultilabelStratifiedKFold` (seed 42).

Metric = **pooled AUC** on out-of-fold predictions (one ROC-AUC over all
`file × class` pairs). Show OOF AUC and get approval before any LB submission.

| ID | Date | Model | Features | OOF AUC | Public LB | Private LB | Notes |
|----|------|-------|----------|--------:|----------:|-----------:|-------|
| logreg | 2026-07-01 | logreg | agg-MFCC(86) | 0.89515 |  |  | ranknorm=0.87843, macro=0.8778, 2s |
| rf | 2026-07-01 | rf | agg-MFCC(86) | 0.92981 | 0.89424 | 0.88889 | ranknorm=0.91335, macro=0.9130, 216s. Sub 54251123. |
| lgbm | 2026-07-01 | lgbm | agg-MFCC(86) | 0.88545 |  |  | ranknorm=0.87974, macro=0.8781, 351s |
| rf_ext | 2026-07-01 | rf | ext(154) | 0.92983 | 0.89843 | 0.89122 | ranknorm=0.91255, macro=0.9111, 218s. Sub 54749853. |
| rf_full | 2026-07-01 | rf | full(182) | 0.93732 | 0.90799 | 0.89786 | ranknorm=0.92021, macro=0.9190, 216s. Sub 54254895. Best. |
| logreg_ext | 2026-07-01 | logreg | ext(154) | 0.88503 |  |  | ranknorm=0.87339, macro=0.8742, 2s |
| logreg_full | 2026-07-01 | logreg | full(182) | 0.89243 |  |  | ranknorm=0.87991, macro=0.8814, 2s |
| cnn | 2026-07-02 | cnn-efficientnet_b0 | logmel | 0.90990 |  |  | macro=0.8953, 18ep/mps, 4084s |
| blend | 2026-07-02 | rank-avg rf_full+rf+cnn | mixed | 0.94099 | 0.91598 | 0.90821 | Best. Global rank-average. Sub 54284029. |
