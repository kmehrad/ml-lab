# Experiment log — OptimalFertilizer (Kaggle PS S5E6)

Metric: **MAP@3**. CV: **StratifiedKFold(n_splits=5, shuffle=True, random_state=42)** on the target.
Submit only after CV review + approval. Naive floor (predict 3 most-frequent classes
`14-35-14 10-26-26 17-17-17`) = **0.27849** OOF.

| ID | Date | Model | OOF MAP@3 | Public LB | Fold mean ± std | Notes |
|----|------|-------|----------:|----------:|-----------------|-------|
| naive | 2026-07-03 | Most-frequent-3 | 0.27849 | — | — | Floor. Constant prior ranking; per-fold train frequencies. |
| exp-001 | 2026-07-03 | LightGBM | 0.34149 | — | 0.34149 ± 0.00034 | Baseline, 8 raw features, native categoricals. 2000 trees lr 0.03 num_leaves 63, early stop → best_iter ~1800 (converged). 1406s. |
| exp-002 | 2026-07-03 | XGBoost | **0.34199** | **0.34415** | 0.34199 ± 0.00057 | Baseline, same features. hist, enable_categorical, depth 6, lr 0.03. **Best single model** (≈ tied with lgbm, Δ+0.0005 < fold std). best_iter=1999 (hit cap — did not converge). 646s. **Submitted: public LB 0.34415 (private 0.34486) — LB +0.0022 over OOF, CV tracks LB.** |
| exp-003 | 2026-07-03 | CatBoost | 0.32091 | — | 0.32091 ± 0.00109 | Baseline, MultiClass, depth 6, lr 0.03. Weakest (−0.021 vs xgb) and **very slow (7728s ≈ 129 min)**. best_iter=1999 (hit cap). **Dropped** from the ensemble (user decision): weakest + slowest. |
| exp-004 | 2026-07-03 | LightGBM + Soil×Crop | 0.33282 | — | 0.33282 ± 0.00054 | **Rejected (−0.0087).** Explicit 55-level Soil×Crop categorical. LGBM already gets the interaction from the two raw categoricals; the combo only adds a high-cardinality overfit handle (best_iter 1800→~1220). |
| exp-005 | 2026-07-03 | LightGBM + Soil×Crop + NPK | 0.32739 | — | 0.32739 ± 0.00078 | **Rejected (−0.0141).** Added NPK sum/diffs/ratios — deterministic functions of existing features, no new info, more overfit surface (best_iter→~890). |
| exp-006 | 2026-07-03 | LightGBM + all FE (soilcross+npk+env) | 0.32487 | — | 0.32487 ± 0.00079 | **Rejected (−0.0166).** Env interaction products on top. Monotonic degradation confirms FE is net-negative on this near-noise data. |
| exp-007 | 2026-07-03 | LightGBM + original-data augmentation | 0.34122 | — | 0.34122 ± 0.00057 | **Rejected (−0.00027, within fold noise).** Appended the 99 UCI seed rows to each train fold. Neutral, as EDA predicted (0.013% of train). best_iter ~1800 (unchanged vs baseline). |
| exp-008 | 2026-07-03 | Blend proba-avg (lgbm+xgb) | **0.34255** | 0.34402 | — | Best OOF (+0.00056 vs xgb ≈ fold std → borderline). Equal-weight probability average, saved as `blend`. **Submitted: public LB 0.34402 (private 0.34550).** Public −0.00013 vs xgb (wash), private **+0.00064** vs xgb — both within noise; blend ≈ xgb, edges it on private. |
| exp-009 | 2026-07-03 | Blend rank-avg (lgbm+xgb) | 0.33986 | — | — | **Rejected (−0.00213 vs xgb).** Per-column rank-normalising flattens each class to a uniform marginal and breaks the within-row calibration MAP@3 needs. Rank-average is the wrong scheme for this metric. |
| exp-010 | 2026-07-03 | Blend proba-avg (lgbm+xgb+cat) | 0.34084 | — | — | **Rejected (−0.00115 vs xgb).** Weak CatBoost (0.32091) drags the average; it adds no useful diversity. Confirms the Step-3 decision to drop it. |
| exp-011 | 2026-07-04 | XGB + cuML combination TE (GPU) | **0.35080** | — | 0.35080 ± 0.00086 | **Step 6 — first above-noise gain (+0.0088 vs blend 0.34255).** cuML `TargetEncoder` (k-fold OOF, smooth=0) on all order≤2 combos × 7 binary classes + original-data-as-columns, hstacked with raw numerics → 510 features. XGB depth6 lr0.03, best_iter ~1600 (uses trees). 726s on RTX 3090 Ti. Saved `xgb_te2_cuml`. **NB the initial hand-rolled LOO encoding failed (best_iter=0, dragged to floor); cuML's k-fold OOF fixed it — see memory.** Order saturates by 2 (order-4 ≈ 0.3515 on 1 fold). |
| exp-012 | 2026-07-04 | XGB TE depth-4 | 0.35118 | — | 0.35118 ± 0.00102 | Same cuML TE features, depth **4** (sweep: 4 > 6 > 8), trees 5000, best_iter ~3000. +0.0004 vs depth-6 (≈ noise). Categoricals dropped (sweep: no help). Saved `xgb_te2_d4`. |
| exp-013 | 2026-07-04 | NN (MLP) on TE | 0.35000 | — | 0.35000 ± 0.00061 | PyTorch MLP [512,256,128] on the same TE features, CE loss, early-stop ~17 epochs. ≈ XGB but a **different model class** (diversity). 115s. Saved `nn_nn_te2`. |
| exp-014 | 2026-07-04 | Hill-climb ensemble (xgb_d4 + NN) | 0.35318 | — | — | Greedy proba-average = 0.5·xgb_te2_d4 + 0.5·NN (single-seed each); dropped the redundant depth-6 XGB. NN diversity drives +0.0020 over best single. |
| exp-015 | 2026-07-04 | NN (MLP) × 10-seed average | **0.35715** | — | 0.35002 ± 0.00076 (per-seed) | **Big seed-averaging win: +0.00715 vs single-seed NN (0.35000).** MAP@3 is highly seed-sensitive; averaging 10 NN probability matrices before top-3 lifts hugely. Now best single model. 1113s. Saved `nn_te2_s10`. |
| exp-016 | 2026-07-04 | XGB TE depth-4 × 3-seed average | 0.35511 | — | 0.35140 ± 0.00076 (per-seed) | +0.0039 vs single-seed (0.35118). XGB less seed-sensitive than the NN but still lifts. Saved `xgb_te2_d4_s3`. |
| exp-017 | 2026-07-04 | **Hill-climb ensemble (NN_s10 + XGB_s3)** | **0.35775** | **0.35776** | — | **SUBMITTED — best result. Public 0.35776 / private 0.35770.** Greedy proba-average = 0.667·nn_te2_s10 + 0.333·xgb_te2_d4_s3. **CV↔LB gap ≈ 0** (OOF 0.35775 → public 0.35776). **+0.01374 public / +0.01220 private vs the prior blend (0.34402/0.34550).** Saved `ensemble`. |

### Push toward 0.36 (branch `experiment/fertilizer-ensemble-036`) — plateaued ~0.358, not submitted
Added 2 more seed-averaged NN architectures for ensemble diversity: **nn_te2b_s10** (wider 1024-512-256,
dropout 0.4) OOF 0.35729; **nn_te2c_s10** (deeper 512-512-256-128, lr 8e-4) OOF 0.35688. Hill-climb over
all 3 NN archs + 3 XGBs → **0.35793** (+0.00018 vs the submitted 0.35775 — within noise, not resubmitted).
**Level-2 stacking (`src/stack.py`, MLP meta-learner) HURT: 0.35643** — base models are all on the same
order-2 TE, too correlated for a nonlinear stacker to help. Conclusion: single models cap ~0.357 and the
ensemble caps ~0.358 (per-column TE signal is weak); reaching 0.36 would need many more seeds (winner used
~100 NNs) and/or a decorrelated feature set. Diminishing returns — stopped here.

## Step 5 verdict (blend)
- **Probability-average > rank-average for MAP@3.** Contrary to the sibling rank-average preference:
  MAP@3 ranks the 7 classes *within each row*, and per-column rank-normalising across the dataset
  destroys the calibration that ordering needs (rank-avg 0.33986 vs proba-avg 0.34255). `blend.py`
  defaults to `proba`; `rank` kept behind a flag.
- **Best blend = proba-avg(lgbm, xgb) = 0.34255**, only **+0.00056 over xgb (0.34199) ≈ one fold std**
  → a borderline, at-noise gain, not a clear improvement. CatBoost drags any blend it joins (−).
- **Submitted (user approved):** blend public LB **0.34402** / private **0.34550** vs xgb public 0.34415 /
  private 0.34486. Public is a wash (−0.00013), private favours the blend (+0.00064) — both within noise.
  Confirms the OOF read: the blend is ≈ xgb, not an above-noise win, but it does edge xgb on the private
  split. **Blend is the strongest submission by private LB.**

## Step 4 verdict (features + augmentation)
- **All engineered features hurt** — monotonic degradation (baseline 0.34149 → +all-FE 0.32487). On this
  near-noise dataset (EDA: MI ~0, features independent), extra features only add overfit surface: `best_iter`
  falls as features are added. **Keep the raw 8 features.** Matches the sibling StellarClass lesson
  (FE neutral/negative on GBDT-ceiling synthetic data).
- **Original-data augmentation is neutral** (−0.00027, within noise). Rejected.
- **CV↔LB confirmed:** xgb OOF 0.34199 → public LB 0.34415 (+0.0022). StratifiedKFold is trustworthy for
  iteration; small OOF gains should transfer.

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
