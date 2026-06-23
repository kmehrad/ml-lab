# EDA Findings — Predicting Irrigation Need

Source notebook: [`notebooks/01_eda.ipynb`](../notebooks/01_eda.ipynb).
Figures: [`reports/figures/`](figures/) (git-ignored; regenerate by running the notebook).
Dataset: 630,000 train rows, 270,000 test rows, 19 features (11 numeric, 8 categorical),
target `Irrigation_Need`. Metric: **balanced accuracy**.

> The original question list referenced `SalePrice` (a House-Price template artifact).
> That column does not exist here; Q6 is answered against the actual target,
> `Irrigation_Need`.

## TL;DR

This is a clean, synthetically generated Playground dataset: **no missing values, no
skew, no outliers, and train/test are statistically indistinguishable.** There is no
data-cleaning work to do. The entire difficulty is the **17.6× class imbalance** scored
under **balanced accuracy** — predicting the rare `High` class (3.3%) well is what wins.
Signal concentrates in a handful of agronomic features; most categoricals carry almost
no marginal information.

## Answers to the eight questions

### 1. What is the target distribution?
Three ordered classes, heavily imbalanced (ratio ≈ **17.6×**):

| Class | Count | Share |
|---|---:|---:|
| Low | 369,917 | 58.72% |
| Medium | 239,074 | 37.95% |
| High | 21,009 | 3.33% |

Figure: `q1_target_distribution.png`.

### 2. Which features have missing values?
**None.** Zero `NaN` values in any feature, in both train and test. No imputation
required.

### 3. Which missing values mean "None" rather than missing data?
No values are literally missing, but two categorical levels encode a *real absence of a
practice* rather than unknown data — they should be treated as meaningful categories,
not imputed:

- `Irrigation_Type = "Rainfed"` — no active irrigation applied (**24.7%** of rows).
- `Mulching_Used = "No"` — mulching practice absent (**50.2%** of rows).

`Previous_Irrigation_mm` is numeric and almost never zero (only 0.54% below 1 mm), so it
does not behave like a "none" flag.

### 4. Which numerical features are skewed?
**None are materially skewed.** All 11 numeric features are essentially symmetric — the
largest absolute skew is `Rainfall_mm` at just **−0.12** (everything else is smaller).
Distributions look close to uniform/synthetic. **No log or power transforms are
warranted.** Figure: `q4_numeric_distributions.png`.

### 5. Which categorical features have high cardinality?
**None.** All 8 categoricals are low cardinality (2–6 levels): `Crop_Type` (6),
`Region` (5), `Soil_Type`/`Crop_Growth_Stage`/`Irrigation_Type`/`Water_Source` (4),
`Season` (3), `Mulching_Used` (2). One-hot encoding or native tree categorical handling
both work; no hashing or target encoding needed, and every test category appears in
train.

### 6. Which features relate most strongly to the target?
By mutual information with `Irrigation_Need` (80k sample), signal is concentrated in a
few features; most categoricals are near-zero:

| Feature | Mutual info |
|---|---:|
| Soil_Moisture | 0.204 |
| Rainfall_mm | 0.189 |
| Crop_Growth_Stage | 0.166 |
| Temperature_C | 0.074 |
| Wind_Speed_kmh | 0.064 |
| Previous_Irrigation_mm | 0.049 |
| Humidity | 0.049 |
| Mulching_Used | 0.047 |
| _Field_Area, Organic_Carbon, EC, Soil_pH, Sunlight_Hours_ | ≤ 0.013 |
| _Water_Source, Irrigation_Type, Crop_Type, Season, Soil_Type, Region_ | ≤ 0.001 |

The relationships are intuitive and monotonic across classes (per-class means):

| | Low | Medium | High |
|---|---:|---:|---:|
| Soil_Moisture | 43.3 | 29.7 | **17.7** |
| Rainfall_mm | 1500 | 1444 | **989** |
| Temperature_C | 25.4 | 28.9 | **34.6** |
| Wind_Speed_kmh | 9.2 | 11.8 | **14.6** |

Higher irrigation need tracks **drier soil, less rainfall, hotter temperatures, and
windier conditions** — physically sensible. `Crop_Growth_Stage` is the one categorical
with strong signal. Figure: `q6_mutual_information.png`.

### 7. Are there obvious outliers?
**No.** Every numeric feature is bounded with realistic agronomic limits (e.g. `Soil_pH`
4.8–8.2, `Humidity` 25–95, `Temperature_C` 12–42), and the IQR rule flags **0 outliers**
in all 11 columns. No clipping or winsorizing needed. Figure: `q7_outlier_boxplots.png`.

### 8. Are train and test distributions similar?
**Essentially identical.**
- Numeric: two-sample KS statistics are all ≤ **0.0027** (largest, `Rainfall_mm`),
  with non-significant p-values — no measurable shift.
- Categorical: max absolute proportion gap across all levels is ≤ **0.0021**.
- Adversarial validation (HistGBM, train-vs-test): AUC = **0.499** — a model cannot
  distinguish the two sets.

Cross-validation scores should therefore transfer well to the leaderboard *in
distribution*; any CV↔LB gap will come from the metric/imbalance, not covariate shift.

## Modeling implications

1. **Optimize balanced accuracy directly.** Use `class_weight="balanced"` /
   `sample_weight`, and tune per-class decision thresholds on out-of-fold predictions —
   the rare `High` class drives the score.
2. **Skip cleaning/transform work.** No imputation, no skew transforms, no outlier
   handling. Spend effort on the metric, validation, and interactions instead.
3. **Lead with the high-signal features** (`Soil_Moisture`, `Rainfall_mm`,
   `Crop_Growth_Stage`, `Temperature_C`, `Wind_Speed_kmh`); low-MI categoricals
   (`Region`, `Season`, `Soil_Type`, `Crop_Type`) may add little alone but could matter
   in interactions.
4. **Validation:** stratified K-fold with a fixed seed; track **per-class recall**, not
   just aggregate accuracy. Train/test parity means local CV is a trustworthy guide.
5. **Baseline:** a gradient-boosted-tree classifier (LightGBM/CatBoost/XGBoost) with
   class weighting; treat the ordinal class structure as an experiment to test, not an
   assumption.
