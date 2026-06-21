# Repository Guidelines

## Project Structure & Module Organization

This directory contains the Home Credit Default Risk Kaggle project. Keep
competition-specific work inside this directory.

- `src/`: reusable data loading, feature engineering, modeling, evaluation, and
  training code.
- `notebooks/`: exploratory analysis and prototypes; move reusable logic into
  `src/`.
- `data/raw/`: original Kaggle files; never commit them.
- `data/processed/`: generated datasets and cached features.
- `tests/`: pytest tests for nontrivial reusable logic (create as needed).
- `models/`: serialized estimators.
- `outputs/`: predictions, metrics, plots, and submissions.
- `experiments/`: experiment log and run-specific artifacts.
- `reports/figures/`: report-ready visualizations.

## Build, Test, and Development Commands

Run commands from this project directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download competition data after configuring Kaggle credentials and accepting
the competition rules:

```bash
kaggle competitions download -c home-credit-default-risk -p data/raw
unzip data/raw/home-credit-default-risk.zip -d data/raw
```

Use `python -m compileall src` for a quick syntax check and
`python -m pytest` to run tests. Training entry points should be runnable as
modules, for example `python -m src.train`.

## Coding Style & Naming Conventions

Use Python with four-space indentation and PEP 8 conventions. Prefer type
hints, small focused functions, deterministic random seeds, and `pathlib.Path`
instead of hard-coded path strings. Use `snake_case` for files, functions,
variables, and CLI options; use `PascalCase` for classes and `UPPER_CASE` for
constants.

Name experiment artifacts consistently, such as `exp-001_metrics.json`,
`exp-001_model.joblib`, and `exp-001_submission.csv`.

## Testing Guidelines

Add pytest tests under `tests/`, using names such as `test_features.py` and
`test_preprocess_handles_missing_values`. There is no current coverage
threshold, but new feature engineering and preprocessing logic should cover
missing values, column consistency, and deterministic behavior.

For model changes, record ROC AUC, folds, seed, and feature set in
`experiments/README.md`. Validate submissions against the sample submission's
columns and row count.

## Commit & Pull Request Guidelines

Git history uses short, imperative subjects such as `Implement train.py
pipeline` and `Document analysis and results`. Keep each commit focused.

Pull requests should summarize code and feature changes, list validation
commands and metrics, note dependency or data requirements, and link relevant
issues. Include plots only when they clarify model or EDA comparisons. Never
commit raw data, credentials, generated submissions, or large model files.
