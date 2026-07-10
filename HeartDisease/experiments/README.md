# Experiment log — HeartDisease (PS S6E2)

Run log, most recent first. Metric is OOF **ROC AUC** (5-fold `StratifiedKFold(42)`) unless
noted. Log null results too.

## Baseline GBDTs (raw 13 features, full 630k train)

| model | OOF AUC | fold mean +/- std | best_iter (typical) | wall time |
|---|---|---|---|---|
| CatBoost | **0.95547** (best single) | 0.95547 +/- 0.00046 | ~2000-2900 (hit 3000 cap on some folds) | 990s |
| XGBoost | 0.95529 | 0.95530 +/- 0.00043 | ~750-870 | 80s |
| LightGBM | 0.95524 | 0.95524 +/- 0.00043 | ~530-640 | 83s |

Blends (equal-weight probability average):

| blend | OOF AUC | delta vs best base |
|---|---|---|
| lgbm + xgb | 0.95530 | +0.00001 vs xgb |
| xgb + cat | 0.95544 | -0.00003 vs cat |
| lgbm + cat | 0.95542 | -0.00005 vs cat |
| lgbm + xgb + cat | 0.95541 | -0.00006 vs cat |

**Verdict: no blend beats solo CatBoost beyond the fold-noise floor (~0.0004-0.0005 std).**
All three models and every blend combination land within ~0.0002 of each other — i.e.
essentially a 3-way tie, consistent with the EDA read that GBDTs are already close to the
public-LB ceiling (~0.9553-0.9554) on this high-signal dataset. CatBoost edges out on OOF
but is ~12x slower than LGBM/XGB for a gain at the noise floor.
