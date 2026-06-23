# EDA Findings — Diabetes (S5E12)

Run `notebooks/01_eda.ipynb`, then fill in measured values. Figures are saved
under `reports/figures/`. Some headline facts are already known from the loaded
data:

- **Rows:** 700,000 train · 300,000 test · **no missing values**.
- **Target `diagnosed_diabetes`:** ~**62.3% positive** (majority class is
  positive). AUC is rank-based, so no resampling/class-weighting is used.

## Q1 — Target balance
Positive rate 0.623. _(figure: target_balance.png)_

## Q2 — Missing values & dtypes
No missing values. 15 numeric (int/float), 3 binary flags, 3 ordinal string
scales, 3 nominal string categoricals. _(see describe() output)_

## Q3 — Numeric distributions & skew
Inspect `bmi`, `triglycerides`, `waist_to_hip_ratio` for skew. Trees handle it
directly; the linear model gets `StandardScaler`. _(measured skew: TBD)_

## Q4 — Cardinality
Binary = 2; ordinal `education_level` (4), `income_level` (5), `smoking_status`
(3); nominal `gender` (3), `ethnicity` (5), `employment_status` (4);
`alcohol_consumption_per_week` is an integer 1–9. _(confirm: TBD)_

## Q5 — Univariate ROC-AUC vs target
Computed on the fully-encoded feature frame. Expected leaders: lipid ratios
(`tg_hdl_ratio`, `chol_hdl_ratio`), `bmi`, `age`, blood-pressure terms,
`hypertension_history`, `family_history_diabetes`. _(measured ranking: TBD)_

## Q6 — Correlations
Expect clusters among lipids (`cholesterol_total`, `ldl_cholesterol`,
`triglycerides`, `non_hdl_cholesterol`) and blood pressure
(`systolic_bp`, `diastolic_bp`, `mean_arterial_pressure`). _(measured: TBD)_

## Q7 — Train vs test similarity
Expect near-identical feature means (same generator). _(measured max |Δmean|: TBD)_

## Q8 — Engineered feature sanity
`tg_hdl_ratio`, `metabolic_risk`, `chol_hdl_ratio`, `bmi_age` should each show
solid univariate AUC, confirming the composites add signal. _(measured: TBD)_

## Modeling implications
- Gradient-boosted trees (LightGBM/XGBoost/CatBoost/HistGB) on raw + engineered
  features; LogisticRegression as a scaled linear baseline.
- Train without class re-weighting (AUC is rank-based).
- Ordinal categoricals → ordered int codes; nominal categoricals → one-hot.
