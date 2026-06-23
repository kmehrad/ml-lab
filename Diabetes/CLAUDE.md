# CLAUDE.md — Diabetes (Kaggle Playground Series S5E12)

## What this is
Binary classification predicting the probability of a diabetes diagnosis
(`diagnosed_diabetes`) from clinical, lifestyle, and demographic features (700k
train rows, ~62% positive). **Metric: ROC-AUC.** Submissions are positive-class
probabilities, not hard labels. Schema: 15 numeric, 3 binary, 3 ordinal
(`education_level`, `income_level`, `smoking_status`), 3 nominal (`gender`,
`ethnicity`, `employment_status`).

## Conventions
- Run everything from the project root: `python -m src.<module>`.
- CV: 5-fold `StratifiedKFold`, `seed=42`. Score on out-of-fold **probabilities**
  with `roc_auc_score`.
- AUC is rank-based, so models are trained **without** class re-weighting (don't
  add `class_weight="balanced"` / `scale_pos_weight` — it hurts calibration
  without helping AUC here).
- Preprocessing is fit **inside each fold on training rows only** (no leakage).
- Optional model deps (lightgbm/xgboost/catboost) are imported lazily inside
  `build_estimator`.
- Schema constants live in `src/data.py`; `load_raw` raises on schema mismatch.

## Files
- `src/data.py` — schema + loading (`load_raw`, `split_features_target`, `load_original`).
- `src/features.py` — `DiabetesFeatureEngineer` (row-wise, stateless engineered features).
- `src/preprocessing.py` — `build_preprocessor("tree"|"linear")`.
- `src/train.py` — `run_cv`, `build_estimator`, `fit_predict_proba`; writes OOF to `experiments/artifacts/`.
- `src/blend.py` — rank-average OOF blend, writes `blend_weights.json`.
- `src/submit.py` — refit + predict test, validate vs `sample_submission.csv`, optional upload.

## Data & outputs (git-ignored)
`data/raw/{train,test,sample_submission}.csv`; optional `data/raw/original.csv`.
OOF arrays + results in `experiments/artifacts/`; submissions in `outputs/`.

## Workflow
1. `python -m pytest`
2. `python -m src.train --model lgbm --sample 50000` (smoke)
3. `python -m src.train --model all --folds 5` (base pool)
4. `python -m src.train --model all --folds 5 --use-original` (augmented pool)
5. `python -m src.blend` (equal-weight rank-average diversity blend)
6. `python -m src.submit --model blend [--submit -m "..."]`

`train` saves `{key}_oof.npy` + `{key}_test.npy` (bagged test preds) per model;
`_aug` suffix for `--use-original` runs. `submit` reads the saved bagged preds
(no refit). Original data is added to **training folds only** (leakage-safe OOF).

## Discipline
**Local CV does not predict this leaderboard** (concept shift; ~0.03 gap). Do not
chase OOF — the equal-weight diverse blend (incl. original-augmented models)
generalised best. Get explicit approval before each Kaggle submission and log
every run in `experiments/README.md`. The original-data concat was the main lever.
