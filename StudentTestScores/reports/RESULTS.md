# Results — StudentTestScores (Kaggle PS S6E1)

Metric: **RMSE** (lower is better). CV: `KFold(5, shuffle, random_state=42)`. Naive floor (predict the
mean) = **18.92**. Full per-run log in `experiments/README.md`.

## Baselines (5-fold OOF RMSE, all 630k rows, 11 raw features)

| Model | OOF RMSE | Fold ± std | Time | Notes |
|---|---:|---|---:|---|
| XGBoost | **8.75063** | ±0.01336 | 89s | Best single. |
| LightGBM | 8.75118 | ±0.01324 | 189s | ≈ tied (Δ0.0006 < fold std). |
| CatBoost | 8.76904 | ±0.01274 | 894s | Weakest + slowest; hit the 3000-iter cap (under-trained). |
| **Blend avg(lgbm, xgb)** | **8.74171** | — | — | **Best. −0.0089 vs xgb.** Candidate submission. |
| Blend avg(lgbm, xgb, cat) | 8.74216 | — | — | Rejected — cat drags it (+0.0004). |

## Read
- Folds are extremely tight (±0.013) and there is no train/test drift, so CV should track the LB well.
- XGBoost ≈ LightGBM (within fold std). The **equal-weight 2-GBDT blend is an above-noise win** (−0.0089)
  and is the current best; CatBoost adds no diversity and is dropped.
- Candidate submission: `outputs/blend_submission.csv` (270,000 rows, `id,exam_score`, clipped to
  [19.6, 100], mean 62.52). **Not uploaded — pending CV review/approval.**

## Next levers (each gated on OOF RMSE)
- Give CatBoost more trees / higher lr so it early-stops, then re-test whether it adds blend diversity.
- Light per-model tuning (depth, lr, leaves) on xgb/lgbm.
- `ratios` feature group (`study_hours × class_attendance`, `study_hours / sleep_hours`) — plausible on
  this high-signal data, but trees may already capture it; adopt only if OOF improves.
- Seed-averaging the GBDTs for a small stability gain.
