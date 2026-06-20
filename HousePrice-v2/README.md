# House Prices - Advanced Regression Techniques

A clean, reproducible ML pipeline for the Kaggle
[House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
competition.

## Goal

Predict the final sale price (`SalePrice`) of residential homes in Ames, Iowa
from 79 explanatory features describing (almost) every aspect of each home.

## Dataset

- **Source:** Kaggle competition *House Prices - Advanced Regression Techniques*.
- **Files** (place in `data/raw/`):
  - `train.csv` — 1460 rows, 79 features + `SalePrice`.
  - `test.csv` — 1459 rows, features only.
  - `sample_submission.csv` — required output format.
  - `data_description.txt` — full field documentation.

Download with the Kaggle CLI:

```bash
kaggle competitions download -c house-prices-advanced-regression-techniques -p data/raw
unzip data/raw/house-prices-advanced-regression-techniques.zip -d data/raw
```

## Metric

Submissions are scored on **RMSLE** — the Root Mean Squared Error between the
log of the predicted price and the log of the observed price:

```
RMSLE = sqrt( mean( (log(1 + y_pred) - log(1 + y_true))^2 ) )
```

We train on `log1p(SalePrice)` so that plain RMSE in log space equals the
competition RMSLE, then invert with `expm1` before submitting.

## Project structure

```
.
├── data/
│   ├── raw/          # original Kaggle CSVs (not committed)
│   ├── processed/    # cached/engineered datasets
│   └── submissions/  # generated submission.csv files
├── notebooks/        # exploratory analysis
├── reports/figures/  # saved plots
├── src/
│   ├── data.py       # load train/test, write submissions
│   ├── features.py   # feature engineering + sklearn preprocessor
│   ├── models.py     # model factories (xgboost/lightgbm optional)
│   ├── evaluate.py   # RMSLE metric + cross-validation
│   └── train.py      # end-to-end training pipeline (CLI)
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`xgboost` and `lightgbm` are optional — the code imports them lazily and only
exposes them as model choices when installed.

## Workflow

1. **Get data** → place the Kaggle CSVs in `data/raw/`.
2. **Explore** → use `notebooks/` for EDA; save plots to `reports/figures/`.
3. **Engineer features** → edit `src/features.py`.
4. **Train + validate** → cross-validate a model in log space:

   ```bash
   python -m src.train --model gradient_boosting --cv 5
   ```

5. **Generate a submission** → fit on all data and write a CSV:

   ```bash
   python -m src.train --model lightgbm --submit
   ```

   Output lands in `data/submissions/submission_<model>.csv`.

Available models: `ridge`, `lasso`, `elasticnet`, `random_forest`,
`gradient_boosting`, plus `xgboost` / `lightgbm` when installed.
