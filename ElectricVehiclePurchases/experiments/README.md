# Experiment log

Record date, code/configuration, feature set, fold scores, overall OOF ROC-AUC, runtime, and
keep/discard decision for every full run. Do not select a model using public LB score alone.

| Run | Model/features | OOF AUC | Fold range | Decision |
|---|---|---:|---:|---|
| S6E9-01 | Logistic regression, schema-driven one-hot/scaling | 0.938094 | 0.936669–0.939071 | Baseline |
| S6E9-02 | LightGBM, native categoricals | 0.941814 | 0.940684–0.942904 | Keep |
| S6E9-03 | XGBoost, native categoricals | 0.941982 | 0.940815–0.943026 | Keep |
| S6E9-04 | CatBoost, native categoricals | 0.941563 | 0.940349–0.942576 | Discard alone |
| S6E9-05 | Equal LightGBM/XGBoost probabilities | **0.942028** | — | Submit |
| S6E9-06 | Equal LightGBM/XGBoost/CatBoost probabilities | 0.941985 | — | Discard |

All full CV runs: 5 shuffled stratified folds, seed 42, all 668,665 rows, base feature set.
