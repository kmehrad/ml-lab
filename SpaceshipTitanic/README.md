# Spaceship Titanic

A tabular **binary classification** project for the Kaggle *Spaceship Titanic* competition.

## Project goal
Predict whether a passenger was **transported to an alternate dimension** after the
Spaceship Titanic collided with a spacetime anomaly. Each passenger is classified as
transported (`True`) or not (`False`) based on their personal records recovered from the
ship's damaged computer system.

## Kaggle competition / dataset source
- Competition: https://www.kaggle.com/competitions/spaceship-titanic
- Raw data files (download separately into `data/raw/`):
  - `train.csv` — labeled training set (features + target)
  - `test.csv` — unlabeled test set (features only)
  - `sample_submission.csv` — submission format

Download example (requires the Kaggle CLI and accepted competition rules):

```bash
kaggle competitions download -c spaceship-titanic -p data/raw
unzip data/raw/spaceship-titanic.zip -d data/raw
```

## Target variable
`Transported` — boolean (`True` / `False`). The ID column is `PassengerId`.

## Evaluation metric
**Classification accuracy** — the fraction of test passengers whose `Transported`
label is predicted correctly.

## Expected workflow
1. **Get data** — download the Kaggle files into `data/raw/`.
2. **Explore** — EDA in `notebooks/` to understand distributions, missingness, and the
   structure of `PassengerId`, `Cabin`, and the spending columns.
3. **Preprocess & engineer features** — implement in `src/features.py`
   (e.g. split `Cabin` into deck/num/side, parse passenger groups, handle spend columns
   and missing values).
4. **Define models** — classifier factories in `src/models.py`.
5. **Evaluate** — cross-validation and accuracy utilities in `src/evaluate.py`.
6. **Train & predict** — run the pipeline via `src/train.py`, writing a submission to
   `data/submissions/`.
7. **Submit** — upload the generated submission CSV to Kaggle.

## Project structure
```
SpaceshipTitanic/
├── data/
│   ├── raw/            # original Kaggle files (gitignored)
│   ├── processed/      # cleaned / feature-engineered data (gitignored)
│   └── submissions/    # generated submission CSVs (gitignored)
├── notebooks/          # exploratory analysis
├── src/
│   ├── __init__.py
│   ├── data.py         # load train/test data
│   ├── features.py     # preprocessing & feature engineering
│   ├── models.py       # model definitions
│   ├── evaluate.py     # metrics & cross-validation utilities
│   └── train.py        # training pipeline entry point
├── reports/
│   └── figures/        # saved plots (gitignored)
├── requirements.txt
└── README.md
```

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
