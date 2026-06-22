# Repository Guidelines

This is the `IrrigationNeed` project: a tabular multiclass-classification model for the
Kaggle [Predicting Irrigation Need](https://www.kaggle.com/competitions/playground-series-s6e4)
competition. Work inside this directory.

## Project Structure & Module Organization

- `src/`: reusable data loading, feature engineering, modeling, evaluation, and
  training modules.
- `notebooks/`: exploratory analysis and prototypes.
- `data/raw/`: downloaded Kaggle files; never commit these.
- `data/processed/`: generated datasets and cached features.
- `models/`, `outputs/`: trained artifacts and predictions.
- `experiments/`: experiment log (`README.md`) and generated `artifacts/`.
- `reports/`: experiment results, parameters, and figures.

## Build, Test, and Development Commands

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run modules from this project root so package imports resolve correctly:

```bash
python -m src.train
```

Check a modified Python file before committing with `python -m compileall src`.

## Coding Style & Naming Conventions

Use Python with four-space indentation and PEP 8 conventions. Prefer type hints, small
functions, deterministic random seeds, and `pathlib.Path` over hard-coded path strings.
Use `snake_case` for files, functions, variables, and CLI options; `PascalCase` for
classes and `UPPER_CASE` for constants. Keep notebooks for investigation and move
reusable logic into `src/`.

Name experiment artifacts consistently, for example `exp-001_metrics.json`,
`exp-001_model.joblib`, and `exp-001_submission.csv`.

## Testing Guidelines

For new nontrivial logic, add `pytest` tests under `tests/`, using names such as
`tests/test_features.py` and `test_preprocess_handles_missing_values`. Run tests with
`python -m pytest`.

For modeling changes, also run the relevant cross-validation command and record the
metric, folds, seed, and feature set. Validate generated submissions against Kaggle's
sample submission columns and row count.

## Commit & Pull Request Guidelines

Use short, imperative commit subjects. Keep commits focused on one experiment. Pull
requests should summarize code and feature changes, list validation commands and
metrics, and mention data or dependency requirements. Never attach raw competition data,
credentials, or large model files.
