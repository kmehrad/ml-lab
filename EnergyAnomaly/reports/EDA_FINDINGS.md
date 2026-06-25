# EDA Findings — LEAD Energy Anomaly Detection

Source notebook: [`notebooks/01_eda.ipynb`](../notebooks/01_eda.ipynb) (executed, figures in `reports/figures/`).
Competition: [`energy-anomaly-detection`](https://www.kaggle.com/competitions/energy-anomaly-detection).
Data: full-year **2016** hourly electricity meter readings.

## Dataset overview
- **Train:** 1,749,494 rows, **200 buildings**. **Test:** 1,800,567 rows, **206 buildings**.
- **Train/test buildings are disjoint** (0 overlap) → generalization is *across buildings*, not across time.
- Two file flavours per split:
  - `train.csv` / `test.csv` — minimal: `building_id, timestamp, meter_reading (+anomaly)`.
  - `train_features.csv` / `test_features.csv` — same rows **+ 50+ engineered columns**: building
    metadata, weather, calendar (raw + cyclical), weather lag aggregates, and **`gte_*` group
    target-encoding** features.
- Submission: `row_id, anomaly` — **probabilities** (metric = AUC-ROC), 1,800,567 rows.

## Target distribution
- Anomaly rate **2.13%** (37,296 / 1,749,494) — strongly imbalanced but rank metric (AUC) is robust to it.
- **Every building has anomalies** (0 zero-anomaly buildings); per-building rate ranges 0→~8.8%
  → real per-building heterogeneity (building context is informative).
- Anomaly rate by `primary_use`: highest **Other 3.70%**, Lodging/Office 2.27%; lowest Services 1.18%,
  Manufacturing 1.71%. Modest but real signal.

## Missing values
- `train.csv`: only `meter_reading` is missing — **6.15%** of rows (107,659).
  **Missing readings are never anomalies** (0 anomalies among them) — but they are still scored,
  so predict a low probability there.
- `train_features.csv`: metadata (`year_built`, `floor_count`) and weather show **0% NaN** — but
  this is because **missing values were sentinel-imputed**, not genuinely complete. E.g.
  `cloud_coverage == 255` for ~46% of rows (a sentinel). Treat weather "completeness" with caution;
  consider re-deriving NaN flags.

## Duplicates
- **No duplicate rows** and **no duplicate `(building_id, timestamp)`** pairs — clean hourly grid.

## Meter reading & artifacts
- Right-skewed; analyzed on `log1p`. 25 rows read exactly 0 (0% anomaly).
- **Strong rule:** `meter_reading == 1.0` → anomaly in **99.97%** of cases (17,053 / 17,059 rows,
  spanning 150 buildings). This is essentially a deterministic flag and a high-value feature /
  postprocessing override (the 1st-place solution exploited the `==1.0` and series-endpoint artifacts).

## Temporal / seasonal patterns
- Anomaly rate varies by **hour of day, weekday, and month** (see `figures/anomaly_by_time.png`),
  motivating calendar features — already provided as raw + cyclical (`hour_x/y`, etc.).

## Weather relationships
- Weather↔anomaly **linear** correlations are weak (|r| ≲ 0.017; `wind_direction`, `cloud_coverage`
  strongest). Useful only via non-linear interactions in tree models.

## Correlation analysis
- **All features have tiny linear correlation with `anomaly`** (top |r| ≈ 0.02:
  `gte_meter_month`, `meter_reading`, `gte_month`). Confirms the target is **not linearly
  separable** → gradient-boosted trees capturing interactions + per-building context are the right tool;
  a plain linear baseline will underperform.
- Many `gte_*` columns are mutually correlated (redundant encodings of overlapping groupings).

## Train vs test drift
- Same `site_id` set and broadly similar `primary_use` mix across splits, but **disjoint buildings**.
- `square_feet` / weather summaries are comparable across splits (no gross covariate shift).
- **Implication:** CV must be **`GroupKFold` by `building_id`** so validation buildings are unseen —
  this is the single most important methodological choice; a random KFold would leak per-building
  signal and massively overstate CV.

## Implications for modeling (feeds the plan)
1. **CV = GroupKFold(building_id)**, 5 folds — non-negotiable.
2. **Use the provided feature files** as a strong starting point, but **audit `gte_*` for leakage**
   (were they computed on full train? if so, recompute inside folds or drop for honest OOF).
3. Add an explicit **`meter_reading == 1.0`** feature/override and **missing-reading → low-prob** handling.
4. Re-derive **NaN/sentinel flags** for weather (e.g. `cloud_coverage == 255`).
5. **GBDT ensemble** (LightGBM/XGBoost/CatBoost/HistGB) over a linear model; AUC → submit probabilities,
   no threshold tuning.
6. Imbalance: trees handle 2% fine; optionally `scale_pos_weight`/downsampling — validate, don't assume.
