# Results — StudentTestScores (Kaggle PS S6E1)

Metric: **RMSE** (lower is better). CV: `KFold(5, shuffle, random_state=42)`. Naive floor (predict the
mean) = **18.92**. Full per-run log in `experiments/README.md`.

## Baselines (5-fold OOF RMSE, all 630k rows, 11 raw features)

| Model | OOF RMSE | Fold ± std | Time | Notes |
|---|---:|---|---:|---|
| XGBoost | **8.75063** | ±0.01336 | 89s | Best single. |
| LightGBM | 8.75118 | ±0.01324 | 189s | ≈ tied (Δ0.0006 < fold std). |
| CatBoost | 8.76904 | ±0.01274 | 894s | Weakest + slowest; hit the 3000-iter cap (under-trained). |
| **Blend avg(lgbm, xgb)** | **8.74171** | — | — | **Best. −0.0089 vs xgb. SUBMITTED → public LB 8.70275 / private 8.73109.** |
| Blend avg(lgbm, xgb, cat) | 8.74216 | — | — | Rejected — cat drags it (+0.0004). |

**Submitted (exp-004):** avg(lgbm, xgb) → **public LB 8.70275 / private 8.73109** (OOF 8.74171). LB is a
touch *better* than OOF (public −0.039), so CV↔LB tracks well on this clean data. Top public LB ~8.57 →
~0.13 headroom for the next levers below.

## Read
- Folds are extremely tight (±0.013) and there is no train/test drift, so CV should track the LB well.
- XGBoost ≈ LightGBM (within fold std). The **equal-weight 2-GBDT blend is an above-noise win** (−0.0089)
  and is the current best; CatBoost adds no diversity and is dropped.
- Candidate submission: `outputs/blend_submission.csv` (270,000 rows, `id,exam_score`, clipped to
  [19.6, 100], mean 62.52). **Not uploaded — pending CV review/approval.**

## Quick GBDT improvement round (2026-07-08) — null result
Tried every lever from the list below on the remote RTX 3090 Ti box (`scripts/remote_run.sh`) plus
local Mac CPU. Full numbers in `experiments/README.md` (exp-006..016). Summary:

| Lever | Best variant OOF | vs. its baseline | Verdict |
|---|---:|---:|---|
| CatBoost, 8000 trees (fixes early-stop) | 8.77680 | worse (was 8.76904) | reject |
| XGBoost depth 7/8 | 8.75361 / 8.75483 | worse (was 8.75063) | reject |
| XGBoost lr 0.02/0.05 | 8.74953 / 8.75404 | +0.0011 / worse | within noise |
| XGBoost base+ratios | 8.75559 | worse | reject |
| LightGBM depth 7/8 | 8.75077 / 8.75318 | ~tied / worse (was 8.75118) | within noise |
| LightGBM lr 0.02/0.05 | 8.74891 / 8.75565 | +0.0023 / worse | within noise |
| Reblend lgbm(lr02) + xgb(lr02) | **8.74112** | −0.00059 vs submitted blend (8.74171) | **noise, not adopted** |

**No lever cleared the noise bar** (fold std ≈ ±0.013; the accepted exp-004 blend gain was −0.0089, an
order of magnitude larger than anything found here). Depth increases and the `ratios` group actively
hurt — this high-signal synthetic data is already well-fit by the vanilla-default GBDTs. **Decision: no
resubmission.** The exp-004 blend (public LB 8.70275 / private 8.73109) remains the best and only
submission. Closing the remaining ~0.13 RMSE gap to the top of the leaderboard (~8.57) would need a
fundamentally different lever — a non-tree model for blend diversity, stacking, or a much larger
hyperparameter search — not incremental GBDT tuning.

## Next levers (previously proposed — now resolved, see above)
- ~~Give CatBoost more trees / higher lr so it early-stops~~ — tried, worse.
- ~~Light per-model tuning (depth, lr, leaves) on xgb/lgbm~~ — tried, within noise or worse.
- ~~`ratios` feature group~~ — tried, worse.
- Seed-averaging the GBDTs — not tried this round (skipped once steps 1–3 showed no OOF room per the
  approved plan's decision gate); low priority given the lr-tuned variants were themselves noise-level.
