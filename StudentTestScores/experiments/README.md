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
| exp-004 | 2026-07-07 | Blend avg(lgbm, xgb) | **8.74171** | **8.70275** | — | **Best. −0.00892 vs xgb** (systematic across full OOF; well beyond the blend's own noise). Equal-weight mean of the two GBDT prediction vectors. Saved as `blend`. **SUBMITTED: public LB 8.70275 / private 8.73109** — LB *better* than OOF (public −0.039, private −0.011); clean data, CV↔LB tracks well. Top LB ~8.57 → ~0.13 headroom. |
| exp-005 | 2026-07-07 | Blend avg(lgbm, xgb, cat) | 8.74216 | — | — | **Rejected (+0.00045 vs 2-model blend).** Weak, under-trained CatBoost adds no useful diversity and drags the average a hair. Confirms dropping cat. |
| exp-006 | 2026-07-08 | CatBoost, 8000 trees, GPU | 8.77680 | — | 8.77679 ± 0.01287 | **Rejected.** Fixed the under-training (`best_iter` now 6079–6758, no longer hits the cap) but OOF is *worse* than the capped 3000-tree run (8.76904) — the extra trees overfit past the earlier plateau. |
| exp-007 | 2026-07-08 | XGBoost depth=7 | 8.75361 | — | 8.75360 ± 0.01342 | **Rejected.** Worse than depth=6 baseline. |
| exp-008 | 2026-07-08 | XGBoost depth=8 | 8.75483 | — | 8.75482 ± 0.01264 | **Rejected.** Worse. |
| exp-009 | 2026-07-08 | XGBoost lr=0.02 | 8.74953 | — | 8.74952 ± 0.01325 | +0.0011 vs xgb baseline — within fold noise (±0.013). |
| exp-010 | 2026-07-08 | XGBoost lr=0.05 | 8.75404 | — | 8.75403 ± 0.01341 | **Rejected.** Worse. |
| exp-011 | 2026-07-08 | XGBoost base+ratios | 8.75559 | — | 8.75558 ± 0.01271 | **Rejected.** Worse than base-only xgb — confirms trees already capture `study_hours×class_attendance` etc.; the extra columns just add overfit surface. |
| exp-012 | 2026-07-08 | LightGBM depth=7 | 8.75077 | — | 8.75077 ± 0.01310 | ≈ tied with baseline (Δ−0.0004) — within noise. |
| exp-013 | 2026-07-08 | LightGBM depth=8 | 8.75318 | — | 8.75317 ± 0.01322 | **Rejected.** Worse. |
| exp-014 | 2026-07-08 | LightGBM lr=0.02 | 8.74891 | — | 8.74890 ± 0.01313 | +0.0023 vs lgbm baseline — within fold noise. |
| exp-015 | 2026-07-08 | LightGBM lr=0.05 | 8.75565 | — | 8.75564 ± 0.01292 | **Rejected.** Worse. |
| exp-016 | 2026-07-08 | Blend avg(lgbm lr02, xgb lr02) | 8.74112 | — | — | **Not adopted.** −0.00059 vs the submitted blend (8.74171) — an order of magnitude below the ~0.0089 gain that made exp-004 a real signal; treated as noise, not a genuine improvement. |

**Conclusion of this round (2026-07-08):** every quick GBDT lever (more CatBoost trees, depth/lr tuning on xgb+lgbm, the `ratios` feature group) either hurt OOF or landed inside fold noise. No candidate beat the submitted exp-004 blend by a margin distinguishable from noise, so **no resubmission** — public LB stays at 8.70275. Closing the remaining ~0.13 gap to the top of the leaderboard likely needs a fundamentally different lever (new model family, stacking, or a much larger hyperparameter search) rather than incremental GBDT tuning.

| exp-017 | 2026-07-08 | PyTorch MLP (`src/models_nn.py`), 1 seed | 8.88036 | — | — | New model family for blend diversity (one-hot cats, standardized numerics, MSE loss). Standalone accuracy ~0.13 worse than the GBDTs — not close to competitive alone. |
| exp-018 | 2026-07-08 | MLP, wider/deeper (512,256,128), 100k sample | 8.94619 | — | — | Same or worse vs the smaller net (8.92766 on the same 100k sample) — **not a capacity problem**; extra width/depth doesn't help. |
| exp-019 | 2026-07-08 | MLP, 3-seed average | **8.86402** | — | — | Small gain from averaging (−0.016 vs 1-seed) but still far from GBDT accuracy. |
| exp-020 | 2026-07-08 | Weighted blend lgbm+xgb+MLP(3-seed), best weight found | 8.74193 | — | — | **Rejected.** Even at the empirically best mixing weight (nn≈0.03, found by grid search over OOF), still slightly worse than the plain lgbm+xgb blend (8.74171). MLP residuals correlate ρ≈0.985–0.986 with the GBDT residuals (vs ρ≈0.996 between lgbm and xgb themselves) — not decorrelated enough to earn any blend weight. |

**Why the NN did not help:** the MLP's prediction errors are almost as correlated with the GBDTs' errors (ρ≈0.985) as the two GBDTs are with each other (ρ≈0.996), despite being a structurally different model (smooth global function vs axis-aligned splits). This is the signature of an **irreducible-noise floor**: once a model captures the real, learnable signal in `study_hours`/`class_attendance`/etc., what is left over is mostly the same per-row noise for every model family, which is why blending different architectures adds almost nothing here — there is very little "different-but-correct" signal left to diversify on. This is consistent with the EDA's read of this dataset as clean and high-signal but still noisy around a mean structure.
