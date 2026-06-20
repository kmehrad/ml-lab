# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

ML pipeline for the Kaggle *House Prices - Advanced Regression Techniques* competition (tabular regression, 79 features → `SalePrice`). Scored on **RMSLE**.

## Commands

The interpreter is `python3` (no `python` on PATH).

```bash
# Cross-validate a model (prints CV RMSLE, no file written)
python3 -m src.train --model gradient_boosting --cv 5

# Fit on all data and write data/submissions/submission_<model>.csv
python3 -m src.train --model lightgbm --submit

# List models buildable in the current env
python3 -c "import src.models as m; print(m.available_models())"
```

`--model` choices: `ridge`, `lasso`, `elasticnet`, `random_forest`, `gradient_boosting`, plus `xgboost`/`lightgbm` when installed. Run everything from the project root so the `src` package and `data/` paths resolve. There is no test suite or linter configured.

## Architecture

The core invariant tying the modules together: **the target is trained in log space.** `train.py` fits on `log1p(SalePrice)`, which makes plain RMSE equal the competition RMSLE, then inverts with `expm1` before writing a submission. Don't break this — `evaluate.cross_validate_rmse` expects an already-logged `y`, and metrics computed on raw-scale predictions won't match the leaderboard.

Data flow (all wired in `src/train.py:run`):
1. `data.py` loads CSVs and splits off `Id`/`SalePrice`. Paths resolve relative to the project root via `Path(__file__).parents[1]`, so cwd doesn't matter.
2. `features.engineer_features` applies **domain** transforms (identical call on train and test): fills "NA means none" categoricals with `"None"`, NaN→0 numerics, forces numeric-coded categoricals (`MSSubClass`, etc.) to strings, and derives `TotalSF`, `TotalBathrooms`, `HouseAge`, `GarageAge`, `Has*` flags, … `preprocess()` is the all-in-one entry point (combines train+test, imputes, log1p's skewed numerics + target, one-hot encodes, drops the two `GrLivArea` outliers) and returns `(X_train, X_test, y, test_ids)`.
3. `features.build_preprocessor` builds a generic `ColumnTransformer` (median/mode impute + one-hot) over whatever columns remain.
4. Both wrap into one sklearn `Pipeline` (`preprocess → model`) so CV folds and the test set get identical preprocessing with no leakage.

Key cross-module contracts:
- **Scaling is model-driven.** `models.NEEDS_SCALING` lists the linear models; `train.build_pipeline` reads it to toggle `StandardScaler` in the preprocessor. When adding a model that needs scaled inputs, add its name there.
- **Optional deps are lazy.** `models.py` imports `xgboost`/`lightgbm` in `try/except` and only registers them in `MODEL_REGISTRY` when present. New models go through the `MODEL_REGISTRY` / `get_model` registry, not direct instantiation.

## Conventions

- Add engineered features in `features.engineer_features`, guarding column access with `if ... in df.columns` (test set lacks `SalePrice`; some columns may be absent).
- `RANDOM_STATE = 42` in `models.py` and `seed=42` defaults in `evaluate.py` — keep CV reproducible.
- `data/`, `reports/figures/` contents are gitignored except `.gitkeep`; submissions are named `submission_<model>.csv`.
