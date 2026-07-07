# Results — StudentHealthRisk (PS S6E7)

- **Task:** 3-class classification of `health_condition` (imbalanced 86/8/6). **Metric:** Balanced
  Accuracy Score (mean per-class recall). Decision-sensitive → predict probabilities, tune per-class
  decision weights on OOF (`metric.tune_weights`), apply to test.
- **Best public LB:** **0.94981** — diverse hillclimb ensemble **xgb + TabM + FTT** (⅓ each), OOF 0.94994.
  The genuine model-family diversity (GBDT + two decorrelated tabular NNs) is the first improvement that
  transferred to the LB (+0.00028 over the 3-GBDT blend; OOF→LB gap −0.00013). See the TabM/FTT phase below.

## Improvement phase (target 0.95, top LB cluster ~0.951) — no real gain found
Systematically tested six levers. All new-signal levers failed on OOF; the one that looked promising
regressed on the LB:
- ⚠️ **Metric-aligned tree count** — looked like a win (OOF 0.94979 → **0.94998**, peak ~800 trees), but
  **the LB regressed to 0.94938 < 0.94953**: the OOF gain was overfit noise that did not transfer.
- ❌ **Original-data augmentation** — real seed rows are complete, synthetic data is missing-heavy →
  off-distribution (−0.0002..−0.0005 on all GBDTs).
- ❌ **Combination target encoding** — 3-level cats + binned numerics already captured by trees.
- ❌ **Decision-rule refinement** — coordinate-ascent already global-optimal.
- ❌ **Diverse ensembling** — TE/NN learners add no decorrelation; hillclimb ignores them.
- ❌ **Pseudo-labeling** — confident self-labels reinforce known patterns, add noise.

**GBDT-only conclusion:** the label is near-deterministic in a few raw features (0 label conflicts), so
GBDTs saturate at **~0.9495** and no sub-0.001 *GBDT* OOF move transferred. The 0.951 cluster's extra
signal is model diversity — see below.

## NN-diversity phase — the breakthrough (LB 0.94953 → 0.94981)
Mining the public notebooks (LLM-assisted discovery) showed the 0.951 cluster = GBDTs + strong tabular
NNs + stacking. Built a pytabkit NN zoo (`src/models_realmlp.py --arch ...`):
- **RealMLP 0.94785** — too weak; every ensemble ignored it. **Rejected.**
- **TabM 0.94905** and **FTT (FT-Transformer) 0.94911** — near-GBDT solo, *different families*, decorrelated
  (each fixes ~4–5% of xgb's errors). Both get **selected** by the hillclimb.
- **Best: hillclimb xgb + TabM + FTT (⅓ each) → OOF 0.94994 → LB 0.94981** (+0.00028, OOF→LB gap −0.00013).
Lessons: NN strength matters (RealMLP failed, TabM/FTT worked); metric-aligned early stopping and
oversampling *hurt* (they trade the calibration our decision tuning needs); genuine cross-family diversity
transfers where single-model OOF fiddling did not. Remaining gap to the 0.951 cluster (~+0.0013) is more/
stronger diverse learners + stacking — diminishing returns per model.

## Base learners (5-fold OOF, balanced accuracy, tuned)

| Model | OOF tuned | Fold ±std (raw) | Notes |
|-------|-----------|-----------------|-------|
| xgboost (GPU, balanced sample-wt) | 0.94972 | ±0.00151 | best single |
| lightgbm (baseline) | 0.94956 | ±0.00147 | reference |
| catboost (GPU, auto-balanced) | 0.94928 | ±0.00119 | raw≈tuned |
| MLP (class-weighted, seeds=5) | 0.94805 | — | diverse but lower |
| **blend lgbm+xgb+cat (equal wt)** | **0.94979** | — | +0.00007 vs best single |
| hillclimb (xgb+cat 50/50) | 0.94986 | — | +0.00014 vs best single (OOF-overfit) |

## Key conclusions
1. **The signal is strong and saturated.** A single feature (`stress_level`) already gives 0.754; the
   full models plateau at **~0.9497**. EDA found **0 label conflicts**, so the target is nearly a
   deterministic function of a few features — which is why all models agree and ensembling/FE add
   nothing above the ±0.0015 fold-noise floor.
2. **Decision tuning is the real lever, not model choice.** Balanced accuracy is decision-sensitive;
   tuning per-class weights on OOF adds +0.0035 over raw argmax. Every model/blend is scored on the
   tuned decision.
3. **Feature engineering (ratio, missflag) does not help** — raw features already capture the signal.
4. **Submission candidate:** equal-weight 3-GBDT blend (0.94979). Ties the OOF-overfit hillclimb within
   noise but should generalize better.

## Reproduce
```bash
uv run python -m src.train --model lgbm                       # baseline (CPU)
scripts/remote_run.sh run python -m src.train --model xgb --device cuda   # GPU box
scripts/remote_run.sh run python -m src.train --model cat --device cuda
scripts/remote_run.sh run python -m src.models_nn --seeds 5 --tag nn_s5
scripts/remote_run.sh pull
uv run python -m src.blend --models lgbm xgb cat --tag blend_gbdt
uv run python -m src.submit --model blend_gbdt                # build + validate (add --submit to upload)
```

See [`../experiments/README.md`](../experiments/README.md) for the full per-run log.
