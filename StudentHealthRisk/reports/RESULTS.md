# Results — StudentHealthRisk (PS S6E7)

- **Task:** 3-class classification of `health_condition` (imbalanced 86/8/6). **Metric:** Balanced
  Accuracy Score (mean per-class recall). Decision-sensitive → predict probabilities, tune per-class
  decision weights on OOF (`metric.tune_weights`), apply to test.
- **Best OOF (tuned):** **0.94979** — equal-weight 3-GBDT blend (lgbm + xgb + cat).
- **Best public LB:** _pending submission approval._

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
