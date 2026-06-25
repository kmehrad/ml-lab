# Experiment log — EnergyAnomaly (LEAD)

Metric: **AUC-ROC**. CV: **GroupKFold(building_id)**, 5 folds (test buildings are disjoint).
Submit only after CV review + approval.

| ID | Date | Model | OOF AUC | Public LB | Private LB | Fold mean ± std | Notes |
|----|------|-------|--------:|----------:|-----------:|-----------------|-------|
| exp-001 | 2026-06-24 | LightGBM | 0.96091 | 0.95870 | 0.97031 | 0.96267 ± 0.01151 | Baseline. Built on `*_features.csv` (gte_* dropped — leakage). Engineered: flatline run-length, is_reading_one/zero/missing, per-building lag/diff (1/24/168h), roll24 z-score, sin/cos wind, cloud_coverage 255→NaN. 2000 trees lr 0.03, early stop. |
| exp-002 | 2026-06-24 | LightGBM + in-fold TE | 0.95521 | — | — | 0.95825 ± 0.00867 | **Rejected (worse −0.0057).** In-fold smoothed target encoding on cross-building groupings (primary_use/site_id/hour/weekday/month + interactions; building_id excluded as non-generalizable). TE adds fold-variance noise over signal LGBM already gets natively. Disabled by default (`--target-encode` flag). Not submitted. |
| exp-003 | 2026-06-24 | XGBoost | 0.96202 | — | — | 0.96154 ± 0.01107 | Same features as exp-001. hist tree_method, native categoricals (train/test category levels aligned), depth 8, early stop. Best single model. |
| exp-004 | 2026-06-24 | HistGradientBoosting | 0.95331 | — | — | 0.95263 ± 0.01417 | sklearn HGB, `from_dtype` categoricals. Weakest; excluded from final blend (drags it down). |
| exp-005 | 2026-06-24 | CatBoost | 0.95809 | — | — | 0.95820 ± 0.00825 | depth 8, AUC eval, early stop. Adds diversity to the blend. |
| exp-006 | 2026-06-24 | Blend lgbm+xgb+cat | **0.96342** | 0.95896 | 0.96934 | — | Best OOF (+0.0025 vs exp-001). Equal-weight rank-average; HistGB excluded (lgbm+xgb+cat+histgb=0.96278). **But LB ~flat: public +0.0003, private −0.0010 vs baseline.** OOF gain did not transfer. |

## Notes
- High per-fold variance (±0.012) reflects building heterogeneity — expected with grouped CV.
- **CV tracks LB well**: OOF 0.96091 vs public 0.95870 (Δ0.0022) → grouped CV is trustworthy for iteration.
- Baseline ≈ mid-pack on the (finished 2022) leaderboard; top public ≈ 0.987. Headroom to climb.
- `gte_*` target-encoding columns are dropped; in-fold recomputation tested (exp-002) and rejected.
- Single-model ranking: xgb > lgbm > cat > histgb. Diversity from xgb+lgbm+cat blends best on OOF.
- **CV↔LB caution:** the +0.0025 OOF blend gain did NOT transfer (public +0.0003, private −0.0010).
  The simple LightGBM baseline (exp-001) remains the best private-LB result (0.97031). Treat small
  OOF gains skeptically; building heterogeneity makes ~0.002 differences leaderboard-noise.
- Next: weather ablation; stronger/diverse features (series-endpoint flags, longer lags); per-model
  tuning; optional GPU DL track for genuinely uncorrelated diversity.
