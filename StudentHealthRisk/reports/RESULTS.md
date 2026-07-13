# Results — StudentHealthRisk (PS S6E7)

- **Task:** 3-class classification of `health_condition` (imbalanced 86/8/6). **Metric:** Balanced
  Accuracy Score (mean per-class recall). Decision-sensitive → predict probabilities, tune per-class
  decision weights on OOF (`metric.tune_weights`), apply to test.
- **Best OOF (tuned):** **0.94994** — hillclimb xgb+tabm+ftt (⅓ each, cross-family GBDT+NN).
- **Public LB:** **0.94981** (current best submission; CV↔LB gap −0.00013, transferred).

## Base learners (5-fold OOF, balanced accuracy, tuned)

| Model | OOF tuned | Fold ±std (raw) | Notes |
|-------|-----------|-----------------|-------|
| xgboost (GPU, balanced sample-wt) | 0.94972 | ±0.00151 | best single |
| lightgbm (baseline) | 0.94956 | ±0.00147 | reference |
| catboost (GPU, auto-balanced) | 0.94928 | ±0.00119 | raw≈tuned |
| MLP (class-weighted, seeds=5) | 0.94805 | — | diverse but lower |
| TabM (pytabkit) | 0.94905 | ±0.0017 (raw) | near-GBDT strength, decorrelated |
| FTT (pytabkit) | 0.94911 | ±0.0036 (raw) | near-GBDT strength, decorrelated |
| blend lgbm+xgb+cat (equal wt) | 0.94979 | — | +0.00007 vs best single; **LB 0.94953** |
| xgb+lgbm @800 trees (metric-aligned ES) | 0.94998 | — | highest single-family OOF; **LB 0.94938 — REGRESSED**, noise |
| hillclimb xgb+cat 50/50 (single-family) | 0.94986 | — | OOF-overfit, not submitted |
| **hillclimb xgb+tabm+ftt (⅓ each)** | **0.94994** | — | **LB 0.94981 — best submission**, genuine cross-family diversity |
| hillclimb xgb+tabm+cat (⅓ each) | 0.94993 | — | ties best OOF; **LB 0.94964** — worse, FTT pick matters |

## Key conclusions
1. **The signal is strong and saturated within any single model family.** A single feature
   (`stress_level`) already gives 0.754; GBDTs plateau at **~0.9497**. EDA found **0 label conflicts**, so
   the target is nearly a deterministic function of a few features — single-family blending/FE/tree-count
   moves under ~0.0005 OOF are noise, not signal (confirmed by the t800 blend regressing on LB despite
   the highest OOF of any GBDT-only combo).
2. **Decision tuning is the real lever, not model choice.** Balanced accuracy is decision-sensitive;
   tuning per-class weights on OOF adds +0.0035 over raw argmax. Every model/blend is scored on the
   tuned decision.
3. **Feature engineering (ratio, missflag) does not help** — raw features already capture the signal.
4. **Cross-family diversity (GBDT + independently-trained NN architectures) is the one lever that
   transferred to LB.** TabM/FTT (pytabkit) are near-GBDT-strength and genuinely decorrelated (~4-5% of
   xgb's errors each), unlike weak plain RealMLP or same-family GBDT variants. But the *specific* pick
   matters more than OOF can resolve at this saturation level: xgb+tabm+ftt (LB 0.94981) beats the
   OOF-tied xgb+tabm+cat (LB 0.94964) by 0.00017 despite <0.00001 OOF difference.
5. **Current best submission:** hillclimb xgb+tabm+ftt, **public LB 0.94981**. Next lever to try: add
   more/different diverse members (fix TabR's "0 features" bug, additional NN seeds/architectures, or
   deeper stacking) rather than re-tuning single models or re-picking among the same 5 base learners.

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
