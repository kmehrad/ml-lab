# EDA Findings — Credit-Card Fraud (ULB)

Source: `data/raw/creditcard.csv` (Kaggle `mlg-ulb/creditcardfraud`). 284,807
transactions over ~2 days; columns `Time`, `V1`–`V28` (PCA components), `Amount`,
`Class`. See `notebooks/01_eda.ipynb` for the figures behind these notes.

## 1. Target balance — extreme imbalance
- **492 frauds / 284,807 rows = 0.173% positive.** A trivial "all genuine"
  classifier scores 99.83% accuracy, so **accuracy is meaningless** here.
- The random-baseline **average precision is 0.0017** (the positive rate). This is
  why the project's primary metric is **AUPRC / average precision**, not ROC-AUC
  (ROC-AUC stays high even for weak models because true negatives dominate).

## 2. Missing values & duplicates
- **No missing values** in any column → no imputation needed.
- **1,081 exact duplicate rows** exist. Left in place (they are plausible repeated
  transactions and removing them risks dropping real signal); flagged here so the
  ~0.17% positive rate is understood to include them.

## 3. `Amount`
- Heavy-tailed: genuine median 22.0 (mean 88.3, max 25,691); fraud median **9.25**
  (mean 122.2, max 2,125). Fraud amounts are typically **smaller** but with a fatter
  upper-mid tail → `log1p(Amount)` is used for the linear model.
- 1,798 genuine and 27 fraud transactions have `Amount == 0` → kept as an
  `amount_zero` flag.

## 4. `Time` — mild diurnal signal
- `Time` is seconds since the first transaction (0–172,792 ≈ 48h). Converting to
  hour-of-day, the **fraud rate is ~3–5× higher in the 00:00–08:00 window**
  (4.8/1000 and 4.0/1000) than during the day (~1.0–1.6/1000). A weak but real
  `hour` feature; absolute `Time` is not used as a model feature beyond this.

## 5. Feature–target separation (single-feature ROC-AUC / AP)
- The PCA components carry almost all the signal. Strongest single separators:
  **V14** (AUC 0.051, i.e. strongly *negative*; single-feature AP 0.62), **V4**
  (AUC 0.938), **V12**, **V11** (AUC 0.918), **V10**, **V17**, **V3**. Several
  single components already reach AP > 0.5.
- `Amount`/`Time` are weak on their own (|AUC−0.5| < 0.05) — they matter only at
  the margin.

## 6. Modelling implications
- Tree GBMs on the raw `V*` + `Amount`/`Time` are the natural primary models; the
  Vs are already decorrelated PCA outputs, so **little feature engineering helps**.
- Imbalance handling (class weighting / resampling) is worth testing but AUPRC, not
  accuracy, must judge it.
- Evaluate on a **fixed stratified hold-out** (no leaderboard exists); report AUPRC
  + an operating threshold (precision/recall/F1), since the business question is
  "how many frauds caught at what false-positive cost", not a single score.
