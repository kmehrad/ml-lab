# ElectricVehiclePurchases — Kaggle Playground S6E9

[Predicting Electric Vehicle Purchases](https://www.kaggle.com/competitions/playground-series-s6e9):
binary classification, target `Will_Buy_EV`, metric ROC-AUC. Submit probabilities, not labels.

## Access prerequisite

Join the competition and accept its rules in Kaggle before downloading. Access was verified on
September 15, 2026; the CSVs are downloaded locally and ignored by Git. The token is read from
`~/.kaggle/access_token` and is never copied into this project.

## Setup

```bash
uv sync
export KAGGLE_API_TOKEN="$(< ~/.kaggle/access_token)"
uv run kaggle competitions download -c playground-series-s6e9 -p data/raw
unzip data/raw/playground-series-s6e9.zip -d data/raw
uv run pytest
```

## Workflow

```bash
uv run python -m src.eda
uv run python -m src.train --model logreg
uv run python -m src.train --model lgbm
uv run python -m src.train --model xgb
uv run python -m src.train --model cat
uv run python -m src.blend --models lgbm xgb
uv run python -m src.submit --model blend
uv run python -m src.submit --model blend --submit -m "S6E9 OOF-selected blend"
```

Full runs save OOF and fold-averaged test probabilities in `experiments/artifacts/`; `--sample 10000`
is a CV smoke run without test predictions. Compare overall OOF AUC and fold scores before selecting
a submission. Use the public leaderboard only as a secondary check.

## Layout

`data/` ignored CSVs · `src/` modular pipeline · `notebooks/` EDA · `reports/` findings and results ·
`experiments/` experiment log and ignored artifacts · `models/`, `outputs/` ignored generated files ·
`tests/` metric, CV and submission checks · `scripts/` optional GPU host workflow.

## Current result

The LightGBM/XGBoost equal blend leads the evaluated candidates at OOF ROC-AUC **0.942028**;
CatBoost and the three-model blend scored lower. The validated blend was submitted on September 15,
2026 (Kaggle reference `56252477`, public ROC-AUC **0.94182**); see `reports/RESULTS.md`.
