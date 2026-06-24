# Experiment log — EnergyAnomaly (LEAD)

Metric: **AUC-ROC**. CV: **GroupKFold(building_id)**, 5 folds (test buildings are disjoint).
Submit only after CV review + approval.

| ID | Date | Model | OOF AUC | Fold mean ± std | Notes |
|----|------|-------|--------:|-----------------|-------|
| exp-001 | 2026-06-24 | LightGBM | 0.96091 | 0.96267 ± 0.01151 | Baseline. Built on `*_features.csv` (gte_* dropped — leakage). Engineered: flatline run-length, is_reading_one/zero/missing, per-building lag/diff (1/24/168h), roll24 z-score, sin/cos wind, cloud_coverage 255→NaN. 2000 trees lr 0.03, early stop. |

## Notes
- High per-fold variance (±0.012) reflects building heterogeneity — expected with grouped CV.
- `gte_*` target-encoding columns are dropped pending an in-fold recomputation (leakage audit).
- Next: add XGBoost/CatBoost/HistGB; weather ablation; in-fold target encoding; blend.
