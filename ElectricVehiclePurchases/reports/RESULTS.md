# Results

## Five-fold OOF comparison

| Candidate | OOF ROC-AUC |
|---|---:|
| Logistic regression | 0.938094 |
| LightGBM | 0.941814 |
| XGBoost | 0.941982 |
| CatBoost | 0.941563 |
| LightGBM + XGBoost equal blend | **0.942028** |
| LightGBM + XGBoost + CatBoost equal blend | 0.941985 |

All models used the 13 base features and the same five shuffled stratified folds (seed 42). The
two-model blend was chosen by OOF score, not public LB. The gain over XGBoost alone is small
(0.000046), so avoid interpreting it as a large effect.

## Kaggle

Validated 286,571-row `id,Will_Buy_EV` probability file and uploaded the two-model blend on
September 15, 2026. Kaggle submission reference: `56252477`; status **complete** and public
ROC-AUC **0.94182**. The private score is not yet available.
