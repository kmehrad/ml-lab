# Results — BirdSpecies (NIPS4B 2013 Bird Challenge)

Metric: **pooled AUC** (one ROC-AUC over all `file × class` pairs). OOF = 5-fold
multilabel-stratified (seed 42). Features = 86-dim aggregated MFCC
(`src/features.py`). Full run log: `experiments/README.md`.

## Summary

Feature levels: **base** = MFCC moments (86) · **ext** = + Δ-MFCC & percentiles
(154) · **full** = + librosa spectral/log-mel from WAV (182).

| Model | Features | OOF pooled AUC | OOF macro AUC | Public LB | Notes |
|-------|----------|---------------:|--------------:|----------:|-------|
| **RandomForest** | **full(182)** | **0.93732** | 0.9190 | _(pending)_ | Best. +audio (spectral+log-mel). |
| RandomForest | base(86) | 0.92981 | 0.9130 | **0.89424** | Baseline. Private 0.88889 (sub 54251123). |
| RandomForest | ext(154) | 0.92983 | 0.9111 | | Δ/percentiles add nothing over base. |
| LogReg | full(182) | 0.89243 | 0.8814 | | Doesn't benefit from richer features. |
| LogReg | base(86) | 0.89515 | 0.8778 | | StandardScaler + balanced, C=1. |
| LightGBM | base(86) | 0.88545 | 0.8781 | | 300 trees; overfits few positives. |

## Notes

- Submissions are gated: report OOF pooled AUC → get approval → upload.
- Adopt a blend only if it beats the best single member by more than fold noise.

## Key findings

1. **RandomForest is the clear baseline leader (0.930).** On 687 clips with
   7–20 positives/class, RF's bagging generalizes better than boosting
   (LightGBM 0.885) or a linear model (LogReg 0.895) — consistent with the
   historical challenge winner's use of per-class random forests.
2. **Rank-normalizing per-class scores *hurts* pooled AUC** (RF 0.930 → 0.913;
   same for all models). The raw one-vs-rest probabilities already pool well
   across classes, so we keep raw scores and drop cross-class calibration.
3. **Pooled > macro AUC** for every model (RF 0.930 vs 0.913): the 14.6% empty
   clips give clean cross-class negatives that the pooled metric rewards.
4. **CV is optimistic vs LB, as predicted.** RF OOF 0.930 → **public LB 0.894
   / private 0.889** (~0.035–0.04 gap). This confirms the EDA covariate-shift
   finding: trust the *relative* OOF ordering, not absolute values, and expect
   real gains to be smaller on the LB than on CV.

5. **Raw-audio features help; MFCC-derived ones don't.** Adding librosa spectral
   shape + 16 log-mel band means (`full`) lifts RF **0.92981 → 0.93732** OOF
   (+0.0075, above noise). MFCC Δ/percentiles (`ext`) are flat — the provided
   MFCC moments already encode them. So the spectral/mel envelope carries the
   extra signal, not higher-order MFCC statistics.

## Leaderboard submissions

| Sub | Model | OOF | Public LB | Private LB |
|-----|-------|----:|----------:|-----------:|
| 54251123 | RF agg-MFCC(86) | 0.92981 | 0.89424 | 0.88889 |
