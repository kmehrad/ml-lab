# CreditCardFraudS6E11 — Predicting Credit Card Fraud

Solution workspace for Kaggle [Playground Series S6E11](https://www.kaggle.com/competitions/playground-series-s6e11).

- **Task:** binary probability prediction for `is_fraud`
- **Competition metric:** ROC-AUC
- **Submission:** `id,is_fraud`, with a finite probability in `[0, 1]` for every test row
- **Deadline:** November 30, 2026 at 11:59 PM UTC

## Access and setup

Competition access was verified on November 4, 2026. The three CSVs are downloaded locally and
ignored by Git. To reproduce the download:

```bash
uv sync
export KAGGLE_API_TOKEN="$(< ~/.kaggle/access_token)"
uv run kaggle competitions download -c playground-series-s6e11 -p data/raw
unzip -o data/raw/playground-series-s6e11.zip -d data/raw
```

Credentials, raw data, artifacts, models, and submissions are ignored by Git.

## Intended commands

Planned pipeline interfaces:

```bash
uv run pytest
uv run python -m src.eda
uv run python -m src.train --model logreg
uv run python -m src.train --model lgbm
uv run python -m src.train --model xgb
uv run python -m src.train --model cat
uv run python -m src.blend --models lgbm xgb cat
uv run python -m src.submit --model blend
```

See [PLAN.md](PLAN.md) for the decision-complete workflow.

## Confirmed data

- Train: 699,960 rows; test: 299,984 rows; 24 predictors plus `id`.
- Target: 11,689 frauds, or 1.66995%; stored as binary `0.0`/`1.0`.
- Predictors: 13 numeric, 5 low-cardinality categorical, and 6 boolean risk flags.
- No missing values, duplicate feature rows, or train/test categorical-level mismatch.
- No customer/card/merchant entity key or full transaction timestamp is provided, so the primary
  validation is five-fold shuffled stratified CV.
