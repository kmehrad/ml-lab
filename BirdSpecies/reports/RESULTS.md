# Results — BirdSpecies (NIPS4B 2013 Bird Challenge)

Metric: **pooled AUC** (one ROC-AUC over all `file × class` pairs). OOF = 5-fold
multilabel-stratified (seed 42). Features = 86-dim aggregated MFCC
(`src/features.py`). Full run log: `experiments/README.md`.

## Summary

| Model | Features | OOF pooled AUC | OOF macro AUC | Public LB | Notes |
|-------|----------|---------------:|--------------:|----------:|-------|
| **RandomForest** | agg-MFCC(86) | **0.92981** | 0.9130 | _(pending)_ | 400 trees, balanced. Best baseline. |
| LogReg | agg-MFCC(86) | 0.89515 | 0.8778 | | StandardScaler + balanced, C=1. |
| LightGBM | agg-MFCC(86) | 0.88545 | 0.8781 | | 300 trees, lr 0.03; overfits few positives. |

## Notes

- Submissions are gated: report OOF pooled AUC → get approval → upload.
- Adopt a blend only if it beats the best single member by more than fold noise.

## Key findings

1. **RandomForest is the clear baseline leader (0.930).** On 687 clips with
   7–20 positives/class, RF's bagging generalizes better than boosting
   (LightGBM 0.885) or a linear model (LogReg 0.895) — consistent with the
   historical challenge winner's use of per-class random forests.
2. **Rank-normalizing per-class scores *hurts* pooled AUC** (RF 0.930 → 0.913;
   same for all models). The raw one-vs-rest probabilities already pool well
   across classes, so we keep raw scores and drop cross-class calibration.
3. **Pooled > macro AUC** for every model (RF 0.930 vs 0.913): the 14.6% empty
   clips give clean cross-class negatives that the pooled metric rewards.
4. **Reminder (from EDA):** measured train/test covariate shift means these OOF
   numbers are likely **optimistic vs the leaderboard** — treat as an upper bound.
