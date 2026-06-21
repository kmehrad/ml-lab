# Home Credit Default Risk

A tabular binary-classification project for the Kaggle
[Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk)
competition.

## Goal

Predict whether a loan applicant will have payment difficulties. The target is `TARGET`
in `application_train.csv`, and Kaggle evaluates submissions using ROC AUC.

## Project structure

```text
HomeCreditDefaultRisk/
├── data/
│   ├── raw/                  # original Kaggle input files
│   └── processed/            # cleaned and feature-engineered datasets
├── experiments/
│   ├── artifacts/            # experiment-specific generated files
│   └── README.md             # experiment log and conventions
├── models/                   # serialized trained models
├── notebooks/                # EDA and prototyping notebooks
├── outputs/                  # predictions, submissions, metrics, and plots
├── reports/
│   └── figures/              # report-ready visualizations
├── src/                      # reusable training and feature code
├── requirements.txt
└── README.md
```

## Download the data

Install and configure the Kaggle CLI, accept the competition rules, then run from this
directory:

```bash
kaggle competitions download -c home-credit-default-risk -p data/raw
unzip data/raw/home-credit-default-risk.zip -d data/raw
```

The raw competition data is intentionally ignored by Git.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Initial workflow

1. Explore the application and relational tables in `notebooks/`.
2. Build reusable data loading and feature engineering in `src/`.
3. Record each run in `experiments/README.md`.
4. Save fitted estimators in `models/`.
5. Save metrics, validation predictions, and Kaggle submissions in `outputs/`.

## Feature Pipeline

Application preprocessing in `src/preprocessing.py` includes:

- semantic handling of `DAYS_EMPLOYED = 365243` and non-applicable car age;
- median numeric imputation with missing-value indicators;
- explicit missing and unknown categorical levels;
- credit, income, annuity, household, age, and employment ratios;
- learned `log1p` copies of non-negative features with absolute skew above 1;
- one-hot encoding for low-cardinality categories;
- cross-fitted target encoding for `ORGANIZATION_TYPE` and `OCCUPATION_TYPE`.

Historical aggregation in `src/features.py` creates one row per applicant from
bureau, bureau balance, previous applications, POS cash, credit-card, and
installment-payment data. It summarizes credit counts, recency, balances,
approval status, delinquency, payment shortfall, and utilization behavior.

The cached aggregate matrix is written to
`data/processed/historical_features.pkl`, which is excluded from Git.

## Training

Run all baseline experiments from the project root:

```bash
python -m src.train --folds 3
```

Run selected models with:

```bash
python -m src.train --models logistic lightgbm --folds 3
```

The command writes metrics and out-of-fold predictions to
`experiments/artifacts/`. All models use the same shuffled stratified folds
with seed 42. ROC AUC is the primary selection metric; log loss and Brier score
are included to monitor probability quality.

Generate the validated fold-averaged blend submission with:

```bash
python -m src.submit
```

This writes `outputs/exp-004_submission.csv` using 35% LightGBM and 65%
CatBoost probabilities.

## Experiment Results

Results below are from three-fold stratified cross-validation on all 307,511
training applications.

| ID | Model | Features | Mean ROC AUC | Fold SD | OOF ROC AUC | Log loss | Brier | Runtime |
|---|---|---|---:|---:|---:|---:|---:|---:|
| exp-001 | Logistic regression | Application preprocessing | 0.751283 | 0.002448 | 0.751271 | 0.248322 | 0.068318 | 28 s |
| exp-002 | LightGBM | Application + historical aggregates | 0.788309 | 0.001485 | 0.788298 | 0.236826 | 0.065871 | 98 s |
| exp-003 | CatBoost | Application + historical aggregates | 0.790896 | 0.000819 | 0.790853 | 0.235855 | 0.065646 | 723 s |
| exp-004 | 35% LightGBM / 65% CatBoost | OOF probability blend | **0.792277** | 0.000965 | **0.792249** | **0.235363** | **0.065526** | negligible |

These are local validation results, not Kaggle leaderboard scores.

The `exp-004` submission was uploaded on June 21, 2026:

| Submission | Public ROC AUC | Private ROC AUC | Kaggle status |
|---|---:|---:|---|
| `exp-004_submission.csv` | 0.79584 | 0.79191 | Complete |

## Experimental Observations

- Logistic regression is a useful preprocessing check, but its AUC is about
  0.037 below LightGBM, confirming important nonlinearities and interactions.
- Historical repayment and credit behavior materially strengthen the feature
  set. The combined tree matrix has 453 columns, including 262 historical
  aggregates.
- CatBoost is approximately 0.0026 AUC stronger than LightGBM and has lower
  fold variance, but takes roughly seven times longer to train.
- The blend gains about 0.0014 AUC over CatBoost, indicating useful model
  diversity.
- The blend weight was selected on the same OOF predictions used to report its
  score. Repeated cross-validation or a held-out validation set should confirm
  the weight before final model selection.
- Neither `class_weight` nor threshold tuning is required for ROC AUC. Final
  submissions must contain continuous probabilities.
- The next highest-value work is feature ablation, recent-history windows,
  tuned boosting parameters, and generation of fold-averaged test predictions.
