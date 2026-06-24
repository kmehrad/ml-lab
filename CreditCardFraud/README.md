# Credit-Card Fraud (Anomaly) Detection — Kaggle ULB dataset

Detect fraudulent credit-card transactions in the **ULB** dataset
(`mlg-ulb/creditcardfraud`): 284,807 transactions over two days, of which **492
(0.173%) are fraud**. Features are anonymised PCA components `V1`–`V28` plus
`Time` and `Amount`.

- **Task:** binary classification under extreme imbalance (probability output)
- **Primary metric:** **AUPRC / Average Precision** (ROC-AUC reported as secondary)
- **Dataset:** <https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud>

There is **no Kaggle leaderboard or test file**, so evaluation is fully local: a
fixed, stratified **20% hold-out test set** plus 5-fold `StratifiedKFold` CV on the
remainder for model selection.

## Data

```bash
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw
unzip -o data/raw/creditcardfraud.zip -d data/raw
```

The schema (`Time`, `V1`–`V28`, `Amount`, `Class`) is pinned in `src/data.py` and
verified on load. The hold-out row indices are cached once to
`experiments/artifacts/holdout_idx.npz` and reused everywhere.

## Pipeline

| Module | Responsibility |
|---|---|
| `src/data.py` | Schema constants, CSV load + schema check, `split_features_target`, fixed stratified `make_holdout_split` |
| `src/features.py` | `FraudFeatureEngineer` — stateless `log1p(Amount)`, `amount_zero`, `hour`; `V1`–`V28` pass through |
| `src/preprocessing.py` | `build_preprocessor("tree"\|"linear")` — `linear` adds `RobustScaler` on raw-scale columns |
| `src/train.py` | 5-fold CV on the remainder; AUPRC primary; `--imbalance {none,classweight,undersample,smote}` (training-fold only); saves OOF + bagged hold-out preds |
| `src/anomaly.py` | Unsupervised baselines (Isolation Forest, LOF) scored on the same hold-out AUPRC |
| `src/ensemble.py` | Rank-average blend over saved OOF/test arrays |
| `src/evaluate.py` | Final hold-out report: AUPRC/ROC-AUC + operating threshold (max-F1 or precision-floor) + PR curve |

## Usage

```bash
python -m pytest                                   # unit tests
python -m src.train --model lgbm --sample 30000    # quick smoke run
python -m src.train --model all                    # base pool: OOF + bagged hold-out preds
python -m src.train --model all --imbalance classweight   # cost-reweighted variant
python -m src.anomaly --model all                  # Isolation Forest + LOF baselines
python -m src.ensemble --members lgbm xgb catboost histgb # rank-average blend
python -m src.evaluate --model catboost --policy max-f1    # final hold-out report + threshold
python -m src.evaluate --model catboost --policy precision-floor --min-precision 0.9
```

All commands run from this project root. CV uses `seed=42`.

## Results

Best model: **CatBoost** — hold-out **AUPRC 0.877 / ROC-AUC 0.982** (OOF AUPRC
0.851). At the max-F1 operating point it catches **77/98 frauds with 3 false
positives** (P 0.96 / R 0.79); a precision-floor of 0.9 trades to **83/98 caught**
(P 0.88 / R 0.85). The GBMs (LightGBM, XGBoost, HistGB) cluster within noise; an
equal-weight blend does **not** beat CatBoost alone (models rank-correlate highly).
Unsupervised baselines are far weaker — Isolation Forest AUPRC 0.16, LOF ≈ random
— confirming the labels carry signal the supervised models exploit. See
`experiments/README.md` for the full run history.

## EDA

`notebooks/01_eda.ipynb` answers a fixed framework (class balance, missing values,
`Amount`/`Time` distributions, single-feature separation, temporal fraud rhythm).
Findings live in `reports/EDA_FINDINGS.md`.
