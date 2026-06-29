# Experiment log

OOF metric = **Balanced Accuracy** (5-fold StratifiedKFold, seed 42). "corrected" = after per-class
probability multiplier search on OOF (`src.metrics.tune_multipliers`). Full breakdown in
`reports/RESULTS.md`.

| id | date | model | OOF bal-acc (raw) | OOF bal-acc (corrected) | LB | notes |
|----|------|-------|-------------------|-------------------------|----|-------|
| 1 | 2026-06-27 | lgbm | 0.96364 | 0.96550 | 0.96659 | LightGBM, balanced weights, 3000 trees / lr 0.03 (submitted) |
| 2 | 2026-06-27 | xgb | 0.96329 | 0.96510 | — | XGBoost hist, native categoricals; ~10 min/fold |
| 3 | 2026-06-27 | cat | 0.96240 | 0.96246 | — | CatBoost, auto balanced weights |
| 4 | 2026-06-27 | nn (MLP) | 0.95518 | 0.95518 | — | GPU RTX 3090 Ti; class-weighted CE (self-balanced) |
| 5 | 2026-06-27 | **blend lgbm+xgb** | 0.96393 | **0.96578** | — | **best**; mult=(1.0, 2.0, 2.52); excludes cat+nn (dilute) |

Not submitted to Kaggle — pending review.

## Round 2 — improvement attempts (all neutral; details in reports/RESULTS.md)
| id | date | experiment | OOF corrected | verdict |
|----|------|-----------|--------------:|---------|
| 6 | 2026-06-28 | lgbm + expanded features | 0.96538 | neutral |
| 7 | 2026-06-28 | lgbm + encoders (freq + OOF target) | 0.96533 | neutral |
| 8 | 2026-06-28 | lgbm + original SDSS17 data | 0.96536 | neutral |
| 9 | 2026-06-28 | lgbm + all combined | 0.96532 | neutral |
| 10 | 2026-06-28 | lgbm Optuna-tuned (raw objective) | 0.96427 | worse (calibration vs correction) |
| 11 | 2026-06-28 | mean-blend lgbm+xgb (expanded) | **0.96555** | best R2 ensemble |
| 12 | 2026-06-28 | stack logreg lgbm+xgb | 0.96549 | ≈ blend |
| 13 | 2026-06-28 | stack logreg lgbm+xgb+cat+nn | 0.96481 | worse |

Conclusion: plateau ~0.9655–0.9658 OOF; no genuine gain over Round 1. `id` exploit ruled out.
