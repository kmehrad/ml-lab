# Results

## Executed results

| Candidate | Rows / folds | ROC AUC | Fold SD | Runtime |
|---|---:|---:|---:|---:|
| Dummy prior | 50k / 2 | 0.50000 | 0.00000 | 0.7 s |
| LightGBM behavior smoke | 50k / 2 | 0.94764 | 0.00057 | 17 s |
| LightGBM behavior, seed 42 | full / 5 | 0.96237 | 0.00059 | 141 s |
| LightGBM behavior, seed 123 | full / 5 | 0.96243 | 0.00040 | 114 s |
| LightGBM behavior, seed 2026 | full / 5 | 0.96240 | 0.00062 | 79 s |
| Three-seed probability average | full / 15 fits | **0.96266** | — | 334 s total |

The three-seed average was the initial selected candidate. Its 100-resample OOF
bootstrap 95% interval is approximately **[0.96232, 0.96300]**. Seed-level AUC range
is only 0.00006, and averaging improves AUC over every constituent. Predictions are
finite, span 0.00050–1.00000, and the submission has the exact 296,302 IDs and columns
from the sample file.

This was a defensible locally validated candidate, not a claim that the entire planned
model search was complete. The subsequent GPU round below evaluates XGBoost, feature
groups, and source-data ablations; CatBoost, HistGBDT, Optuna, and TabM remain pending.

The final selection gate remains: repeated OOF ROC AUC first, changes smaller than
`max(0.0002, one standard error of the paired delta)` treated as ties, then prefer the
simpler single model or equal-weight probability blend. The generated file is
`outputs/lgbm_behavior_3seed_submission.csv`; it was uploaded only after explicit user
approval.

## Kaggle submission

- Submission reference: `55698328`
- Submitted: 2026-08-22 19:53 UTC
- File: `lgbm_behavior_3seed_submission.csv`
- Public leaderboard ROC AUC: **0.96373**
- Status: complete
- Private score: not available while the competition is active

## GPU improvement round

Full five-fold XGBoost ran on the RTX 3090 Ti. The strongest ablation added raw-column
missing flags and total missing count; despite missing-rate drift, the improvement was
stable enough to retain after three-seed evaluation.

| Candidate | OOF ROC AUC | Decision |
|---|---:|---|
| XGBoost behavior | 0.963349 | Keep as baseline |
| XGBoost + source rule | 0.963363 | Tie; no useful gain |
| XGBoost compact | 0.962858 | Reject |
| XGBoost + missing features | 0.963469 | Promote |
| XGBoost + original data, weight 1 | 0.963258 | Reject |
| XGBoost + original data, weight 5 | 0.963079 | Reject |
| XGBoost missing, three-seed average | **0.963718** | **New selection** |

The new selection improves over the submitted three-seed LightGBM OOF score by
**0.001057**. A 100-resample paired-bootstrap interval for the AUC delta is
**[+0.000998, +0.001113]**, well above the 0.0002 materiality threshold. XGBoost and
LightGBM predictions correlate at 0.9957; blending LightGBM into the repeated XGBoost
predictions reduced OOF AUC, so the pure XGBoost seed average is preferred.

The validated replacement file is `outputs/xgb_missing_3seed_submission.csv`. It has
been submitted to Kaggle after explicit approval.

### Improved Kaggle submission

- Submission reference: `55714637`
- Submitted: 2026-08-23 12:29 UTC
- File: `xgb_missing_3seed_submission.csv`
- Public leaderboard ROC AUC: **0.96479**
- Improvement over the prior public score: **+0.00106**
- Status: complete
- Private score: not available while the competition is active
