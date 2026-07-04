# Experiment log — OptimalFertilizer (Kaggle PS S5E6)

Metric: **MAP@3**. CV: **StratifiedKFold(n_splits=5, shuffle=True, random_state=42)** on the target.
Submit only after CV review + approval. Naive floor (predict 3 most-frequent classes
`14-35-14 10-26-26 17-17-17`) = **0.27849** OOF.

| ID | Date | Model | OOF MAP@3 | Public LB | Fold mean ± std | Notes |
|----|------|-------|----------:|----------:|-----------------|-------|
| naive | 2026-07-03 | Most-frequent-3 | 0.27849 | — | — | Floor. Constant prior ranking; per-fold train frequencies. |
| exp-001 | 2026-07-03 | LightGBM | 0.34149 | — | 0.34149 ± 0.00034 | Baseline, 8 raw features, native categoricals. 2000 trees lr 0.03 num_leaves 63, early stop → best_iter ~1800 (converged). 1406s. |
| exp-002 | 2026-07-03 | XGBoost | **0.34199** | — | 0.34199 ± 0.00057 | Baseline, same features. hist, enable_categorical, depth 6, lr 0.03. **Best single model** (≈ tied with lgbm, Δ+0.0005 < fold std). best_iter=1999 (hit cap — did not converge). 646s. |
| exp-003 | 2026-07-03 | CatBoost | 0.32091 | — | 0.32091 ± 0.00109 | Baseline, MultiClass, depth 6, lr 0.03. Weakest (−0.021 vs xgb) and **very slow (7728s ≈ 129 min)**. best_iter=1999 (hit cap). |

## Notes
- **All three clear the naive floor comfortably** (best +0.064). OOF ~0.34 matches the EDA prediction
  for a low-signal dataset; folds are extremely tight (±0.0003–0.001), so CV is stable and sub-0.001
  differences are noise.
- **XGBoost ≈ LightGBM** (0.34199 vs 0.34149, Δ0.0005 within fold std) → effectively tied for best.
- **CatBoost lags** (0.32091) and is ~5–12× slower than the others. Default depth 6 / lr 0.03 under-fits
  here; it hit the 2000-iteration cap without early-stopping. It may still add ensemble diversity
  (Step 5 will confirm whether it helps or drags the blend), but its cost/benefit is poor — candidate to
  drop or to re-tune (higher lr / fewer iters) if it doesn't lift the blend.
- **xgb and cat both hit best_iter=1999** (early stopping on multi-logloss never triggered) → more trees
  could add a hair, but this is a Step-4 tuning question, not a baseline change.
- Next (Step 4): feature engineering (Soil×Crop combo first — top EDA signal; NPK ratios/sums/bins),
  the original-dataset augmentation test, and light per-model tuning — each gated on OOF MAP@3.
