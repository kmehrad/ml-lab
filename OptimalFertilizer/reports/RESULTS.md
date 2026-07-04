# Results — OptimalFertilizer (Kaggle PS S5E6)

Consolidated results and decisions. Metric **MAP@3**; CV is
`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on the target. The blow-by-blow run
log with every rejected experiment lives in [`../experiments/README.md`](../experiments/README.md);
this file is the summary and the "why".

## Headline

| Model | OOF MAP@3 | Public LB | Private LB | Status |
|-------|----------:|----------:|-----------:|--------|
| Naive floor (most-frequent 3) | 0.27849 | — | — | reference |
| LightGBM baseline | 0.34149 | — | — | base learner |
| **XGBoost baseline** | 0.34199 | **0.34415** | 0.34486 | submitted |
| CatBoost baseline | 0.32091 | — | — | dropped (weak + 129 min) |
| **Blend — proba-avg(lgbm, xgb)** | **0.34255** | 0.34402 | **0.34550** | submitted · best private LB |

**Best submission by private LB: the blend (0.34550).** On public LB the blend and xgb are a wash
(0.34402 vs 0.34415, Δ−0.00013). All gains past the ~0.342 baseline are within the ±0.0005 fold-noise
floor — consistent with the dataset's known low-signal character.

## What worked

- **Two tree GBDTs on the raw 8 features.** XGBoost ≈ LightGBM (0.34199 vs 0.34149, Δ within fold std) —
  effectively tied for best single model. Native categorical handling for `Soil Type` / `Crop Type`; no
  encoding needed.
- **Equal-weight probability-average blend** of lgbm + xgb — best OOF (0.34255) and best private LB
  (0.34550). The lift is small (+0.00056 OOF vs xgb ≈ one fold std), i.e. borderline, but it is the
  strongest submission on the private split.
- **CV↔LB tracks.** xgb OOF 0.34199 → public 0.34415 (+0.0022); blend OOF 0.34255 → public 0.34402.
  StratifiedKFold(5) is trustworthy for iteration — no drift (see EDA).

## What didn't (and why we're sure)

- **All engineered features hurt** — monotonic degradation as features were added
  (baseline 0.34149 → +Soil×Crop 0.33282 → +NPK 0.32739 → +env 0.32487), with `best_iter` falling each
  time. On near-noise data (EDA: MI ≈ 0, features independent) extra features only add overfit surface.
  **Kept the raw 8 features.**
- **Rank-average blending hurt** (0.33986 vs proba-avg 0.34255). MAP@3 ranks the 7 classes *within each
  row*; per-column rank-normalising across the dataset flattens every class to a uniform marginal and
  destroys the calibration that ordering needs. **Probability-average is the correct blend here** —
  contrary to the sibling-project rank-average default (which assumed binary/low-class ranking).
- **CatBoost** — weakest single model (0.32091, under-fits at depth 6) *and* 5–12× slower (129 min). It
  drags any blend it joins (proba-avg with cat: 0.34084, −0.00115). **Dropped** from the ensemble.
- **Original-dataset augmentation** (99 UCI seed rows appended per train fold) — neutral (−0.00027,
  within noise), exactly as EDA predicted (0.013% of train). **Rejected.**

## Decisions log

| Decision | Verdict | Basis |
|----------|---------|-------|
| Feature engineering (Soil×Crop, NPK, env) | Rejected | Monotonic OOF degradation |
| Original-data augmentation | Rejected | Neutral (−0.00027, within noise) |
| CatBoost in ensemble | Dropped | Weakest + slowest; drags the blend |
| Blend scheme | proba-average (not rank-average) | rank-avg −0.0021; proba-avg is calibration-preserving for MAP@3 |
| Submissions | xgb baseline, then proba-avg blend | Both approved after OOF review |

## Reproduce

```bash
uv run pytest                                        # metric + submission-format tests
uv run python -m src.train --model lgbm              # -> lgbm_oof/_test.npy
uv run python -m src.train --model xgb               # -> xgb_oof/_test.npy
uv run python -m src.blend --models lgbm xgb         # proba-avg -> blend_oof/_test.npy (OOF 0.34255)
uv run python -m src.submit --model blend            # build + validate outputs/blend_submission.csv
```

## Open lever (not yet tried)

Both xgb and lgbm hit the 2000-tree cap without early-stopping triggering, so they did not fully
converge. More trees / a smaller learning rate is the one untried knob that could yield a genuine (if
small) lift, which would then flow into the blend. Everything else is at the noise floor.
