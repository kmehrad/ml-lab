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
3. `python -m src.train --model all --folds 5`
4. `python -m src.blend`
5. `python -m src.submit --model blend [--submit -m "..."]`

## Discipline
Only commit/submit a change if OOF AUC improves above noise (compare
`auc_fold_std` between runs). Log every meaningful run in `experiments/README.md`.
