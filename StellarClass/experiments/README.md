# Experiment log

OOF metric = **Balanced Accuracy** (5-fold StratifiedKFold, seed 42). "corrected" = after per-class
probability multiplier search on OOF (`src.metrics.tune_multipliers`). Full breakdown in
`reports/RESULTS.md`.

| id | date | model | OOF bal-acc (raw) | OOF bal-acc (corrected) | LB | notes |
|----|------|-------|-------------------|-------------------------|----|-------|
| 1 | 2026-06-27 | lgbm | 0.96364 | 0.96550 | — | LightGBM, balanced weights, 3000 trees / lr 0.03 |
| 2 | 2026-06-27 | xgb | 0.96329 | 0.96510 | — | XGBoost hist, native categoricals; ~10 min/fold |
| 3 | 2026-06-27 | cat | 0.96240 | 0.96246 | — | CatBoost, auto balanced weights |
| 4 | 2026-06-27 | nn (MLP) | 0.95518 | 0.95518 | — | GPU RTX 3090 Ti; class-weighted CE (self-balanced) |
| 5 | 2026-06-27 | **blend lgbm+xgb** | 0.96393 | **0.96578** | — | **best**; mult=(1.0, 2.0, 2.52); excludes cat+nn (dilute) |

Not submitted to Kaggle — pending review.
