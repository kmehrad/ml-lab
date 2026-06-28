# Results — Stellar Class (S6E6)

**Metric: Balanced Accuracy.** All numbers below are **5-fold StratifiedKFold OOF** (seed 42), so they
are honest cross-validated estimates, not leaderboard scores. "corrected" = after the per-class
probability-multiplier search that optimizes balanced accuracy on OOF (see `src/metrics.py`).

> ⚠️ **No Kaggle submission has been made yet** — awaiting review. The submission file is built locally
> at `outputs/blend_submission.csv` but has not been uploaded.

## Model leaderboard (OOF balanced accuracy)

| model | OOF raw | OOF corrected | notes |
|-------|--------:|--------------:|-------|
| **lgbm + xgb (blend)** | 0.96393 | **0.96578** | **best** — equal-weight prob avg, mult=(GALAXY 1.0, QSO 2.0, STAR 2.52) |
| lgbm | 0.96364 | 0.96550 | LightGBM, balanced class weights |
| lgbm + xgb + cat | 0.96446 | 0.96529 | CatBoost drags corrected score down |
| xgb | 0.96329 | 0.96510 | XGBoost (hist, native categoricals); slowest (~10 min/fold) |
| lgbm + cat | 0.96438 | 0.96494 | |
| cat | 0.96240 | 0.96246 | CatBoost; little benefit from correction |
| lgbm+xgb+cat+nn | 0.96423 | 0.96459 | adding NN dilutes the strong GBDTs |
| nn (MLP, GPU) | 0.95518 | 0.95518 | already balanced via class-weighted loss (mult=1) |

Full subset search is reproducible — see "Reproduce" below.

### Takeaways
- **redshift + spectral_type carry most of the signal** (see `reports/EDA_FINDINGS.md`); even a single
  GBDT reaches ~0.965 balanced accuracy.
- The **balanced-accuracy decision correction matters more than model choice**: it adds ~+0.0018–0.0020
  to every GBDT by up-weighting the minority classes (QSO, STAR) before argmax.
- **Best ensemble is just LightGBM + XGBoost.** CatBoost and the NN are individually weaker and
  correlated with the GBDTs, so including them lowers the corrected OOF score. They were kept as
  trained artifacts for transparency but excluded from the final blend.
- The GPU MLP self-balances (class-weighted cross-entropy), so its optimal multipliers are all 1.0.

## Best model — per-class breakdown (lgbm+xgb blend, corrected)

Balanced accuracy = mean of per-class recall = **0.9658**.

| class | precision | recall | f1 | support |
|-------|----------:|-------:|---:|--------:|
| GALAXY | 0.9898 | 0.9508 | 0.9699 | 377,480 |
| QSO | 0.9495 | 0.9752 | 0.9622 | 117,143 |
| STAR | 0.8510 | 0.9714 | 0.9072 | 82,724 |

Confusion matrix (rows = true, cols = pred; order GALAXY, QSO, STAR):

```
[[358904   5697  12879]
 [  1715 114236   1192]
 [  1991    378  80355]]
```

The correction trades GALAXY precision (more GALAXY→STAR confusions) for higher STAR/QSO recall —
the right call when each class is weighted equally.

## Predicted test-set class distribution (best blend)
GALAXY 62.9% / QSO 20.8% / STAR 16.3% (vs train prior 65.4 / 20.3 / 14.3 — STAR up-weighted by design).

## Compute
- GBDTs trained on the dev Mac (CPU): LightGBM ~15 min, XGBoost ~50 min, CatBoost ~31 min (5 folds).
- MLP trained on remote RTX 3090 Ti (`kamron@192.168.0.92`); ~1–2 min/fold.

## Reproduce
```bash
uv run python -m src.train --model lgbm        # also xgb, cat
# NN on GPU box: uv run python -m src.dl.train_nn  -> scp nn_*.npy back to experiments/artifacts/
uv run python -m src.blend --models lgbm xgb   # best blend -> blend_*.npy + blend_metrics.json
uv run python -m src.submit --model blend      # writes outputs/blend_submission.csv (no upload)
```

## LB
| date | submission | LB balanced accuracy | notes |
|------|------------|----------------------|-------|
| — | — | — | not submitted yet (pending review) |
