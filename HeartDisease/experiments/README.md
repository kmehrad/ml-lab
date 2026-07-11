# Experiment log — HeartDisease (PS S6E2)

Run log, most recent first. Metric is OOF **ROC AUC** (5-fold `StratifiedKFold(42)`) unless
noted. Log null results too.

## Round 2 — Step 1: interactions feature group (2026-07-10) — NULL RESULT

Evaluated the previously-untested `interactions` group (`Thallium x Chest pain type`,
`ST depression x Slope of ST`) end-to-end on the two cheap models:

| model | features | OOF AUC | vs base |
|---|---|---|---|
| LightGBM | base | 0.95524 | — |
| LightGBM | base+interactions | 0.95517 | -0.00007 |
| XGBoost | base | 0.95529 | — |
| XGBoost | base+interactions | 0.95526 | -0.00003 |

**Verdict: no lift, within noise floor (both actually slightly negative).** Skipped running on
CatBoost per the gating rule (neither cheap model cleared +0.0005). `interactions` is dropped —
all subsequent Round 2 steps use `base` features only. Consistent with the EDA/CLAUDE.md read
that axis-aligned GBDT splits already capture most of what's in these interaction terms
indirectly (Thallium and Chest pain type are both already top-ranked solo features).

## Baseline GBDTs (raw 13 features, full 630k train)

| model | OOF AUC | fold mean +/- std | best_iter (typical) | wall time |
|---|---|---|---|---|
| CatBoost | **0.95547** (best single) | 0.95547 +/- 0.00046 | ~2000-2900 (hit 3000 cap on some folds) | 990s |
| XGBoost | 0.95529 | 0.95530 +/- 0.00043 | ~750-870 | 80s |
| LightGBM | 0.95524 | 0.95524 +/- 0.00043 | ~530-640 | 83s |

Blends (equal-weight probability average):

| blend | OOF AUC | delta vs best base |
|---|---|---|
| lgbm + xgb | 0.95530 | +0.00001 vs xgb |
| xgb + cat | 0.95544 | -0.00003 vs cat |
| lgbm + cat | 0.95542 | -0.00005 vs cat |
| lgbm + xgb + cat | 0.95541 | -0.00006 vs cat |

**Verdict: no blend beats solo CatBoost beyond the fold-noise floor (~0.0004-0.0005 std).**
All three models and every blend combination land within ~0.0002 of each other — i.e.
essentially a 3-way tie, consistent with the EDA read that GBDTs are already close to the
public-LB ceiling (~0.9553-0.9554) on this high-signal dataset. CatBoost edges out on OOF
but is ~12x slower than LGBM/XGB for a gain at the noise floor.

## Submission 1 — solo CatBoost (2026-07-10)

User-approved over the 3-way blend (tie on OOF, cat has the edge and is a simpler single
model). **OOF 0.95547 -> public LB 0.95358 / private LB 0.95513.** Public LB is ~0.0019
below OOF (larger gap than private's ~0.0003) — plausible public/private split variance on
a late submission; private tracks OOF closely, which is the more meaningful number here.
Public LB 0.95358 sits below the live top cluster (~0.9553-0.9554), consistent with the EDA
read that public-leaderboard notebooks use extra levers (tuning, ensembling, calibration)
beyond a raw single-model baseline.

**Next options (Step 5, gate on OOF AUC, only submit if genuinely above noise):** light
per-model tuning (xgb/lgbm didn't hit their tree cap but cat did on 2 folds — more trees
might help cat specifically); original-data augmentation
(`ritwikb3/heart-disease-statlog`); EDA-flagged interaction terms
(`Thallium x Chest pain type`, `ST depression x Slope of ST`).
