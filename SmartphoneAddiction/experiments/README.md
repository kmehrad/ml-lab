# Experiment registry

Every complete run writes aligned `{tag}_oof.npy`, `{tag}_test.npy`, and
`{tag}_metrics.json`. Shared `y.npy`, `folds.npy`, and `test_id.npy` establish row
identity. Metrics JSON records parameters, fold scores, elapsed time, feature group,
augmentation, seed, Python version, and git revision.

Planned sequence: 50k/two-fold smoke; six five-fold baselines; feature ablations;
fold-local original-data augmentation at weights 1 and 5; 30-trial tuning of the two
strongest GBDT families; three-seed finalist evaluation; GPU TabM diversity candidate;
then mean, rank, and cross-fitted nonnegative logistic blends. Runs below the tie
threshold `max(0.0002, one standard error of paired OOF deltas)` do not displace the
simpler candidate. Public leaderboard results are never selection inputs.

| Tag | Status | Notes |
|---|---|---|
| `dummy_smoke` | complete | 50k rows, 2 folds; ROC AUC 0.50000 |
| `lgbm_behavior_smoke` | complete | 50k rows, 2 folds; ROC AUC 0.94764 |
| `lgbm_behavior_s42` | complete | 691,369 rows, 5 folds; ROC AUC 0.96237; 141 s |
| `lgbm_behavior_s123` | complete | 691,369 rows, 5 folds; ROC AUC 0.96243; 114 s |
| `lgbm_behavior_s2026` | complete | 691,369 rows, 5 folds; ROC AUC 0.96240; 79 s |
| `lgbm_behavior_3seed` | complete | mean probabilities; OOF ROC AUC 0.96266 |
| `xgb_behavior_gpu_s42` | complete | full 5-fold GPU; ROC AUC 0.96335 |
| `xgb_source_rule_gpu_s42` | complete | full 5-fold GPU; ROC AUC 0.96336; tied |
| `xgb_compact_gpu_s42` | complete | full 5-fold GPU; ROC AUC 0.96286; rejected |
| `xgb_missing_gpu_s42` | complete | full 5-fold GPU; ROC AUC 0.96347 |
| `xgb_behavior_aug1_gpu_s42` | complete | original weight 1; ROC AUC 0.96326; rejected |
| `xgb_behavior_aug5_gpu_s42` | complete | original weight 5; ROC AUC 0.96308; rejected |
| `xgb_missing_gpu_s123` | complete | full 5-fold GPU; ROC AUC 0.96344 |
| `xgb_missing_gpu_s2026` | complete | full 5-fold GPU; ROC AUC 0.96331 |
| `xgb_missing_3seed` | complete | selected mean probabilities; OOF ROC AUC 0.96372 |

LightGBM and XGBoost finalists, feature ablations, and source-data weights were executed.
CatBoost, HistGBDT, tuning, and TabM remain explicit pending experiments. Do not record
partial-fold artifacts as complete.

## Baseline and Submission 1 — LightGBM behavior, three-seed bag (2026-08-22)

The CPU smoke gate established the expected separation: dummy prior scored 0.50000 and
LightGBM with behavior features scored 0.94764 on 50k rows / 2 folds. The promoted
LightGBM configuration then ran on all 691,369 rows with five stratified folds under
three seeds:

| seed | OOF AUC | fold mean +/- std | wall time |
|---:|---:|---:|---:|
| 42 | 0.962371 | 0.962372 +/- 0.000590 | 141s |
| 123 | 0.962430 | 0.962433 +/- 0.000405 | 114s |
| 2026 | 0.962397 | 0.962398 +/- 0.000615 | 79s |
| **probability average** | **0.962661** | — | 334s total |

The seed average exceeded every constituent and had a 100-resample bootstrap 95% AUC
interval of approximately [0.96232, 0.96300]. After explicit approval it was submitted
as **submission 55698328: public LB 0.96373**. Private LB is unavailable while the
competition is active.

## GPU improvement round — XGBoost and ablations (2026-08-23)

Provisioned an isolated Python 3.11 Conda environment on `pyml` (RTX 3090 Ti). The 50k /
2-fold CUDA smoke completed in 2.9s at 0.94790 OOF AUC, slightly above the comparable
LightGBM smoke, so XGBoost advanced to full CV. All ablations below use identical
five-fold seed-42 validation rows; original source rows were added only to training
partitions.

| run | features / data | OOF AUC | delta vs XGB behavior | verdict |
|---|---|---:|---:|---|
| `xgb_behavior_gpu_s42` | behavior | 0.963349 | — | GPU baseline |
| `xgb_source_rule_gpu_s42` | behavior + source rule | 0.963363 | +0.000014 | Tie; reject complexity |
| `xgb_compact_gpu_s42` | compact seven | 0.962858 | -0.000491 | Reject |
| `xgb_missing_gpu_s42` | behavior + missing flags/count | **0.963469** | **+0.000121** | Promote to repeated CV |
| `xgb_behavior_aug1_gpu_s42` | behavior + 7,500 source rows, weight 1 | 0.963258 | -0.000091 | Reject |
| `xgb_behavior_aug5_gpu_s42` | behavior + 7,500 source rows, weight 5 | 0.963079 | -0.000270 | Reject |

The source dataset was not helpful despite being the likely generative source. The
`compact` result confirms the demographic/context columns retain residual signal. The
missingness result is modest for one seed, but justified a repeated-CV check despite
the documented train/test missing-rate drift.

## XGBoost missingness seed averaging and blend gate (2026-08-23)

| seed | OOF AUC | fold mean +/- std | wall time |
|---:|---:|---:|---:|
| 42 | 0.963469 | 0.963473 +/- 0.000565 | 30s |
| 123 | 0.963442 | 0.963443 +/- 0.000414 | 30s |
| 2026 | 0.963310 | 0.963311 +/- 0.000551 | 30s |
| **probability average** | **0.963718** | — | 90s total |

The three-seed XGBoost average improves on the LightGBM three-seed OOF score by
**+0.001057**. The 100-resample paired-bootstrap delta interval is
**[+0.000998, +0.001113]**, entirely positive and well beyond the 0.0002 materiality
gate. XGBoost/LightGBM predictions correlate at 0.9957. The best coarse blend (80% XGB
missing seed-42 / 20% LightGBM) reached only 0.963515, and every tested LightGBM weight
reduced the repeated-XGBoost score, so the pure XGBoost seed average was selected.

## Submission 2 — XGBoost missingness, three-seed bag (2026-08-23)

After exact sample-column, ID-order, finite-probability, range, and row-count validation,
the user approved `outputs/xgb_missing_3seed_submission.csv`.

**Submission 55714637: OOF 0.963718 -> public LB 0.96479.** This improves the first
submission by +0.00106 public AUC, matching the +0.001057 OOF gain almost exactly.
Private LB is unavailable while the competition is active. This is the current best
model and submission.
