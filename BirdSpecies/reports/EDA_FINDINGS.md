# EDA findings — BirdSpecies (NIPS4B 2013 Bird Challenge)

Source: `notebooks/01_eda.ipynb` (regenerate with `notebooks/build_eda.py`).
Metric is **AUC pooled over all `(file × class)` pairs**, so per-class
prevalence *and* cross-class score comparability both matter. Figures embedded
in the notebook and saved to `reports/figures/`; key numbers in
`reports/eda_stats.json`.

## Headline numbers

| Aspect | Value |
|---|---|
| Classes / train files / test files | **87 / 687 / 1000** |
| Feature vector (aggregated MFCC) | 86-dim (5 stats × 17 coeffs + log n_frames) |
| Clip duration | 1.02 – 5.00 s (median **5.00**) |
| MFCC frames/file (median) | 1288 (train = test) |
| Classes per clip (cardinality) | mean **1.71**, max 6 |
| Empty clips (no class) | **100 (14.6%)** |
| Class prevalence | min **7** (`Colpal_song`), median 13, max **20** (`Erirub_song`) |
| Classes never present in train | 0 |
| Taxon mix | 77 bird, 9 insect, 1 amphibian |
| Vocalization mix | 45 song, 41 call, 1 drum |

## 1. Data distribution

Clips are short (≤5 s; most are exactly 5.00 s, a tail down to ~1 s). MFCC frame
counts track duration and are **distributed near-identically train vs test**
(median 1288 each) — so raw clip length is not a train/test discriminator.
Audio is 44.1 kHz mono 16-bit (`tps_canaux_sr_nbits_TRAIN.csv`).

## 2. Target distribution

- **Very few positives per class (7–20).** With 87 one-vs-rest models this is the
  central difficulty: each classifier sees ≤20 positives among 687 rows. Only a
  ~3× imbalance between rarest/commonest class, but the *absolute* positive count
  is what bites. (`class_prevalence.png`)
- **Sparse multi-label:** mean 1.71 classes/clip, max 6; **14.6% of clips are
  empty** (all-zero targets) — legitimate negatives for every class.
  (`label_cardinality.png`)
- Song vs call are separate classes for the same species (e.g. `Erirub_call` vs
  `Erirub_song`), which inflates the class count and creates natural label pairs.

## 3. Missing values

Aggregated features are fully finite (0 NaN/inf in train and test). One raw
train cepstra file (`trainfile060`) contains `-inf`/`nan` on silent frames;
`features.mfcc_stats` masks these with nan-aware reductions, so it needs no
special-casing downstream. No class is absent from the training set.

## 4. Train vs test distribution shift ⚠️

Test labels are hidden, so we compare feature distributions (per-feature KS test).
**36 / 86 features shift at p<0.01 (55 / 86 at p<0.05)**, concentrated in the
MFCC **variance** features (`mfcc7_std`, `mfcc6_std`, `mfcc9_std` most shifted).
Frame-count/mean features barely move. (`train_test_shift.png`)

→ **Covariate shift is real.** OOF CV AUC will likely be *optimistic* relative to
the leaderboard. Mitigations to try: standardize/robust-scale features, prefer
tree models (less sensitive to feature scale) or quantile-normalize inputs, and
trust the *gap-aware* read of CV rather than absolute CV numbers.

## 5. Feature correlation & redundancy

- MFCC coefficients are moderately correlated (adjacent coeffs especially);
  `mfcc_coeff_correlation.png`.
- **No near-zero-variance features** (0 / 86).
- PCA on standardized features: **20 components → 90%** variance, 31 → 95%
  (of 86). Meaningful redundancy → linear models will want regularization /
  optional PCA; trees are indifferent. (`pca_scree.png`)

## 6. Label space & co-occurrence

- **High-cardinality categoricals: N/A** — all inputs are numeric MFCC stats,
  no categorical features.
- Class co-occurrence is **weak**: strongest pairs co-occur in only ~9–10 of 687
  clips (e.g. `Erirub_call`+`Prumod_song`, `Petpet_call`+`Sylcan_song`).
  (`class_cooccurrence.png`) → label correlations are mild; **classifier
  chains / joint models are unlikely to beat independent per-class models** —
  start independent.
- Multilabel-stratified 5-fold keeps per-fold prevalence reasonably consistent
  (rare classes vary most, as expected).

## Implications for modeling

1. **Per-class one-vs-rest on the 86-dim aggregated MFCC** is the natural Phase-3
   baseline (LogReg/RF/LightGBM). Heavy regularization / class weighting given
   ≤20 positives per class.
2. **Calibrate across classes before pooling** — the metric is pooled AUC, so
   compare raw vs per-class rank-normalized OOF and keep what wins.
3. **Treat CV as optimistic** because of the measured train/test shift; favor
   scale-robust models and don't over-tune to CV deltas smaller than fold noise.
4. **Independent classes** (weak co-occurrence) — skip chains initially.
5. **Phase 4** adds temporal/spectral descriptors (MFCC deltas, percentiles,
   mel-spectrogram summaries) that aggregation discards; **Phase 5** mel-spec CNN
   may capture temporal structure but is risky with so few positives per class →
   treat as ensemble diversity with strong augmentation.
