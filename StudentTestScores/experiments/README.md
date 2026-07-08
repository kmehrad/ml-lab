# Experiment log — StudentTestScores (Kaggle PS S6E1)

Metric: **RMSE** (lower is better). CV: **KFold(n_splits=5, shuffle=True, random_state=42)**.
Submit only after CV review + approval. Naive floor (predict the train mean ≈ 62.5) = train std ≈
**18.92** RMSE. Public LB tops out ~8.57.

| ID | Date | Model | OOF RMSE | Public LB | Fold mean ± std | Notes |
|----|------|-------|---------:|----------:|-----------------|-------|
| naive | 2026-07-07 | Predict-mean | 18.9169 | — | — | Floor = train-target std. |
| exp-001 | 2026-07-07 | XGBoost | **8.75063** | — | 8.75062 ± 0.01336 | Baseline, 11 raw features, native categoricals. hist, depth 6, lr 0.03, 3000 trees, early-stop → best_iter ~1900–2200. **Best single.** 89s. |
| exp-002 | 2026-07-07 | LightGBM | 8.75118 | — | 8.75117 ± 0.01324 | Baseline, same features. num_leaves 63, lr 0.03, early-stop best_iter ~1800–2100. ≈ tied with xgb (Δ+0.0006 < fold std). 189s. |
| exp-003 | 2026-07-07 | CatBoost | 8.76904 | — | 8.76903 ± 0.01274 | Baseline, RMSE loss, depth 6, lr 0.03. **Hit the 3000-iter cap on every fold (no early stop) → under-trained**, weakest (+0.018 vs xgb) and slowest (894s). Candidate to drop or give more trees. |
| exp-004 | 2026-07-07 | Blend avg(lgbm, xgb) | **8.74171** | — | — | **Best. −0.00892 vs xgb** (systematic across full OOF; well beyond the blend's own noise). Equal-weight mean of the two GBDT prediction vectors. Saved as `blend`. |
| exp-005 | 2026-07-07 | Blend avg(lgbm, xgb, cat) | 8.74216 | — | — | **Rejected (+0.00045 vs 2-model blend).** Weak, under-trained CatBoost adds no useful diversity and drags the average a hair. Confirms dropping cat. |
