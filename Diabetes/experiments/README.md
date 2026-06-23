# Experiment Log

ROC-AUC on 5-fold out-of-fold predictions (`seed=42`). Higher is better. Only
adopt a change if its OOF AUC beats the current best by more than `auc_fold_std`.

## Model comparison

| Date | Model | OOF AUC | Fold mean ± std | Avg precision | Notes |
|------|-------|--------:|-----------------|--------------:|-------|
| 2026-06-22 | **lgbm** | **0.72664** | 0.72665 ± 0.00079 | 0.81168 | best single; 1000 trees, lr 0.03 |
| 2026-06-22 | xgb | 0.72504 | 0.72505 ± 0.00074 | 0.81087 | |
| 2026-06-22 | histgb | 0.72329 | 0.72330 ± 0.00085 | 0.80936 | |
| 2026-06-22 | catboost | 0.72140 | 0.72141 ± 0.00082 | 0.80855 | |
| 2026-06-22 | baseline (logreg) | 0.69452 | 0.69452 ± 0.00086 | 0.78983 | scaled features |
| 2026-06-22 | blend | 0.72664 | — | — | SLSQP put 100% on lgbm; models rank-corr ~0.99, no lift |

**Finding:** the four GBMs are highly correlated (rank-corr ~0.99) and LightGBM
dominates, so rank-average blending does not beat it. Submit LightGBM directly;
revisit blending only after the models are made more diverse (different feature
sets / objectives / seeds).

Data: 700k train rows, target `diagnosed_diabetes` (~62% positive). 15 numeric +
3 binary + 3 ordinal + 3 nominal features.

## Artifact naming
- `experiments/artifacts/{model}_oof.npy` — out-of-fold positive-class probabilities.
- `experiments/artifacts/y_true.npy` — aligned 0/1 labels (base data, no augmentation).
- `experiments/artifacts/model_results.{json,csv}` — metrics per model.
- `experiments/artifacts/blend_weights.json` — chosen blend members + weights.
- `outputs/{model}_submission.csv` — final probability submission.

## Run history
- 2026-06-22 — `src.train --model all --folds 5` on full 700k. Best lgbm OOF AUC
  0.72664. Blend = lgbm (no lift; rank-corr ~0.99).
- 2026-06-22 — Submitted `lgbm_submission.csv`. **Public LB 0.69687 / private
  0.69384** vs OOF 0.72664. Large CV→LB gap (~0.03).
- 2026-06-23 — Diagnostics: adversarial AUC (train vs test) **0.634** (moderate
  shift); adversarial sample-weighting gives **no lift**; a "test-like" holdout
  scores *higher* than random CV (0.743 vs 0.727), so the LB gap is **not**
  covariate shift → concept shift / intrinsic test difficulty. **Local CV does
  not predict the LB.** Top of leaderboard is only **0.70504**.
- 2026-06-23 — Regularized GBM blend (lgbmr+lgbmr2+xgb+histgb, rank-avg, OOF
  0.72544). **Public LB 0.69704 / private 0.69351.** Marginal gain (models
  rank-corr 0.99).
- 2026-06-23 — Found the **source dataset** (`alamshihab075/health-and-lifestyle-
  data-for-diabetes-prediction`, 97k rows, identical schema). 1st-place writeup
  used original-data concat + target encoding + hill-climbing ensemble (private
  0.70514). Augmented ensemble (original in each training fold only): OOF drops
  ~0.001 per model (expected) but...
- 2026-06-23 — **10-model combo blend** (non-aug + original-augmented GBMs,
  rank-avg, OOF 0.72452). **Public LB 0.69812 / private 0.69460** — biggest jump
  yet (+0.0011 over the prior blend). The **original-data concat was the lever**.
- 2026-06-23 — Target encoding (6 categoricals + 4 interactions, smoothed,
  CV-safe). TE models stay **0.998 corr** with base (low-cardinality cats already
  captured) → no decorrelation. 16-set blend (base+aug+TE+TE-aug GBMs):
  **Public LB 0.69814 / private 0.69477** — flat. **GBM blending has plateaued**
  (all models 0.995+ corr). Next lever = decorrelated model *classes* (DART, MLP).
- 2026-06-23 — Diverse classes: DART-LGBM OOF ~0.7248 but still **0.995 corr**
  with the GBM blend (no help). MLP OOF ~0.695 and logreg ~0.695 are the only
  **decorrelated** models (~0.91 corr) but individually weak, so they lower the
  blended OOF. Only a *small* weight on a decorrelated model carries new signal.
- 2026-06-23 — **Stopped at best public 0.69814 / private 0.69477** (diversity
  blend; top of LB 0.70504). Plateau confirmed: feature set + tree models cap out
  near 0.698; the remaining gap needs the full 1st-place recipe (more diverse
  model classes + hill-climbing over many models). Productionized the winning
  recipe into `src/`: regularised configs, leakage-safe `--use-original`
  augmentation, bagged test predictions, equal-weight rank-average blend.

## Final state
- Best submission: equal-weight rank-average of regularised GBMs (lgbm/lgbm2/xgb/
  histgb) across base + original-augmented pools.
- Reproduce: `train --model all` (+ `--use-original`) → `blend` → `submit --model blend`.
- Biggest lever: concatenating the source dataset into each training fold.
