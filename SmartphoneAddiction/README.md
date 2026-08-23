# Predicting Smartphone Addiction

Reproducible binary-probability pipeline for Kaggle Playground Series S6E8. The
selection metric is ROC AUC, so all models emit continuous positive-class probabilities;
no classification threshold is tuned.

## Setup and data

Run commands from this directory (the lockfile lives at the repository root):

```bash
uv sync --package smartphone-addiction
uv sync --package smartphone-addiction --extra gpu  # CUDA host only
kaggle competitions download -c playground-series-s6e8 -p data/raw
unzip data/raw/playground-series-s6e8.zip -d data/raw
kaggle datasets download -d jayjoshi37/smartphone-usage-and-addiction-prediction -p data/original
unzip data/original/smartphone-usage-and-addiction-prediction.zip -d data/original
```

The source dataset is optional. Only the 12 shared predictors and binary target are
mapped. `transaction_id`, `user_id`, and target-derived `addiction_level` are excluded.
Source rows are appended only to a fold's training partition.

## EDA and verification

```bash
uv run python notebooks/make_notebook.py
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb --ExecutePreprocessor.timeout=600
uv run pytest
```

## Experiments

```bash
# CPU smoke (50k rows, shared two folds)
uv run python -m src.train --model dummy --folds 2 --sample 50000 --tag dummy_smoke
uv run python -m src.train --model lgbm --features behavior --folds 2 --sample 50000 --tag lgbm_behavior_smoke

# Full baselines/ablations (repeat model and feature choices as documented)
uv run python -m src.train --model cat --features behavior --folds 5 --seed 42 --tag cat_behavior_s42
uv run python -m src.train --model cat --features behavior --folds 5 --seed 42 --original data/original/Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv --original-weight 5 --tag cat_behavior_aug5_s42
uv run python -m src.tune --model lgbm --trials 30 --sample 250000

# Blend aligned full runs, then make a validated file (does not upload)
uv run python -m src.blend cat_behavior_s42 lgbm_behavior_s42 --method mean --tag finalist_mean
uv run python -m src.submit --tag finalist_mean
```

Full finalists must be rerun with seeds 42, 123, and 2026. Average each candidate's
repeated OOF/test probabilities before model comparison. Use `scripts/remote_run.sh`
for the GPU smoke and TabM candidate. Artifacts are documented in
`experiments/README.md`; selection rules and executed comparisons belong in
`reports/RESULTS.md`.

Submission upload is deliberately absent from the pipeline. After reviewing the final
validated CSV, explicitly authorize a separate `kaggle competitions submit` command.
