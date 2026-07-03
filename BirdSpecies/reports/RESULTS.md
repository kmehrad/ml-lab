# Results — BirdSpecies (NIPS4B 2013 Bird Challenge)

Metric: **pooled AUC** (one ROC-AUC over all `file × class` pairs). OOF = 5-fold
multilabel-stratified (seed 42). Features = 86-dim aggregated MFCC
(`src/features.py`). Full run log: `experiments/README.md`.

## Summary

Feature levels: **base** = MFCC moments (86) · **ext** = + Δ-MFCC & percentiles
(154) · **full** = + librosa spectral/log-mel from WAV (182).

| Model | Features | OOF pooled AUC | OOF macro AUC | Public LB | Notes |
|-------|----------|---------------:|--------------:|----------:|-------|
| **Blend (rf_full+rf+cnn)** | mixed | **0.94099** | — | **0.91598** | Final. Global rank-average. Private 0.90821 (sub 54284029). |
| RandomForest | full(182) | 0.93732 | 0.9190 | 0.90799 | Best single. +audio. Private 0.89786 (sub 54254895). |
| RandomForest | base(86) | 0.92981 | 0.9130 | **0.89424** | Baseline. Private 0.88889 (sub 54251123). |
| RandomForest | ext(154) | 0.92983 | 0.9111 | | Δ/percentiles add nothing over base. |
| LogReg | full(182) | 0.89243 | 0.8814 | | Doesn't benefit from richer features. |
| LogReg | base(86) | 0.89515 | 0.8778 | | StandardScaler + balanced, C=1. |
| LightGBM | base(86) | 0.88545 | 0.8781 | | 300 trees; overfits few positives. |
| CNN (effnet_b0) | log-mel | 0.90990 | 0.8953 | | timm transfer, 18ep/mps, SpecAugment. Diversity member. |

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

6. **The CNN underperforms the classical RF but is a diversity source.** A
   timm EfficientNet-B0 on log-mel spectrograms reaches OOF 0.90990 — below
   RF-full (0.937), as expected with only 7–20 positives/class. Its value is
   ensemble diversity (different representation + model family) for the blend.

## Leaderboard submissions

| Sub | Model | OOF | Public LB | Private LB |
|-----|-------|----:|----------:|-----------:|
| 54284029 | **Blend rf_full+rf+cnn** | 0.94099 | **0.91598** | **0.90821** |
| 54254895 | RF full(182) | 0.93732 | 0.90799 | 0.89786 |
| 54251123 | RF base(86) | 0.92981 | 0.89424 | 0.88889 |

Steady climb at every stage — base → +audio features → +CNN blend:
**public 0.89424 → 0.90799 → 0.91598** (private 0.88889 → 0.89786 → 0.90821).
The +0.0075 OOF gain (base→full) became +0.0138 public LB; the blend's +0.0037
OOF became +0.0080 public. OOF gains hold up directionally on the LB even though
absolute CV is optimistic (covariate shift).

## Key findings (7)

7. **The blend wins by diversity, not by strong members.** rf_full+rf alone
   (0.937) doesn't beat rf_full; adding the weaker CNN (0.910) lifts the blend to
   **0.94099 OOF → 0.91598 public LB** — its distinct log-mel representation
   decorrelates the errors. **Global** rank-average (not per-class) is essential:
   raw-mean averaging scores only 0.926 (RF/CNN scales differ).
