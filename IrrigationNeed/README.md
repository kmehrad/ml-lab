# Predicting Irrigation Need

A tabular multiclass-classification project for the Kaggle
[Predicting Irrigation Need](https://www.kaggle.com/competitions/playground-series-s6e4)
competition (Playground Series S6E4).

## Goal

Predict the `Irrigation_Need` of an agricultural field from soil, weather, crop, and
management features. The target has three ordered classes — `Low`, `Medium`, and
`High` — and submissions are hard class labels.

**Evaluation metric: balanced accuracy** (the unweighted mean of per-class recall).
This is the single most important fact for modeling: each class contributes equally to
the score regardless of size, so the 3.3% `High` minority class matters as much as the
58.7% `Low` majority. Optimizing plain accuracy will *not* optimize the leaderboard —
class weighting, resampling, and decision-threshold tuning are central, not optional.

## Data overview

| File | Rows | Columns |
|---|---:|---|
| `train.csv` | 630,000 | 20 features + `id` + `Irrigation_Need` |
| `test.csv` | 270,000 | 20 features + `id` |
| `sample_submission.csv` | 270,000 | `id`, `Irrigation_Need` |

Target distribution in `train.csv` (imbalanced):

| Class | Count | Share |
|---|---:|---:|
| Low | 369,917 | 58.7% |
| Medium | 239,074 | 38.0% |
| High | 21,009 | 3.3% |

Feature columns: `Soil_Type`, `Soil_pH`, `Soil_Moisture`, `Organic_Carbon`,
`Electrical_Conductivity`, `Temperature_C`, `Humidity`, `Rainfall_mm`,
`Sunlight_Hours`, `Wind_Speed_kmh`, `Crop_Type`, `Crop_Growth_Stage`, `Season`,
`Irrigation_Type`, `Water_Source`, `Field_Area_hectare`, `Mulching_Used`,
`Previous_Irrigation_mm`, `Region`. A mix of numeric and categorical fields.

## Preliminary analysis

### Task type
A **supervised multiclass classification** problem (3 classes). The classes are loosely
ordered (`Low` < `Medium` < `High`), so ordinal approaches are worth a look, but the
balanced-accuracy metric and the dominant minority-class challenge make standard
multiclass classifiers the natural starting point. Gradient-boosted trees (LightGBM,
CatBoost, XGBoost) fit the tabular, mixed-type data well; a linear/logistic baseline is
useful as a sanity check.

### Features (19 total)
- **11 numeric:** `Soil_pH`, `Soil_Moisture`, `Organic_Carbon`,
  `Electrical_Conductivity`, `Temperature_C`, `Humidity`, `Rainfall_mm`,
  `Sunlight_Hours`, `Wind_Speed_kmh`, `Field_Area_hectare`, `Previous_Irrigation_mm`.
  All are continuous and bounded with plausible agronomic ranges (e.g. `Soil_pH`
  4.8–8.2, `Humidity` 25–95%), no obvious outliers or sentinel values.
- **8 categorical (all low cardinality, 2–6 levels):** `Soil_Type` (4), `Crop_Type`
  (6), `Crop_Growth_Stage` (4), `Season` (3), `Irrigation_Type` (4), `Water_Source`
  (4), `Mulching_Used` (2), `Region` (5). Every category present in `test.csv` also
  appears in `train.csv`, so no unseen-level handling is required. Low cardinality means
  one-hot or native categorical handling (CatBoost/LightGBM) both work.

### Data quality
- **No missing values** in either train or test.
- **No duplicate rows** (excluding `id`).
- Train/test categorical levels are consistent. This is a synthetically generated
  Playground dataset, so the data is clean — feature *engineering* and *interactions*,
  not cleaning, are where gains will come from.

### Target & class imbalance
| Class | Count | Share |
|---|---:|---:|
| Low | 369,917 | 58.7% |
| Medium | 239,074 | 38.0% |
| High | 21,009 | 3.3% |

Imbalance ratio ≈ **17.6×** (Low : High). Combined with the balanced-accuracy metric,
correctly recalling the rare `High` class is disproportionately valuable. Expect a
tricky CV↔leaderboard relationship; use **stratified** cross-validation and track
per-class recall, not just the aggregate score.

### Modeling implications
1. Optimize for **balanced accuracy**, not accuracy — use `class_weight="balanced"` /
   `sample_weight`, and consider tuning per-class decision thresholds on out-of-fold
   predictions.
2. Use **stratified K-fold** CV and a fixed seed for a reliable validation signal.
3. Start with a gradient-boosted-tree baseline; treat the ordinal class structure as an
   experiment, not an assumption.

## Project structure

```text
IrrigationNeed/
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
kaggle competitions download -c playground-series-s6e4 -p data/raw
unzip data/raw/playground-series-s6e4.zip -d data/raw
```

The raw competition data is intentionally ignored by Git.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Feature pipeline

Implemented in `src/` (see `reports/EDA_FINDINGS.md` for the rationale):

- `src/data.py` — fixed column schema, CSV loading with stable categorical levels,
  and ordinal target encoding (`Low`/`Medium`/`High` → `0`/`1`/`2`).
- `src/features.py` — `IrrigationFeatureEngineer`, which appends agronomic features:
  an evapotranspiration (water-demand) proxy `et_demand`, demand/supply ratios
  (`temp_to_moisture`, `moisture_rain_ratio`, `wind_dryness`), a standardized
  `aridity_index` composite of the four strongest EDA signals (fit on train only to
  avoid leakage), and the `is_rainfed` / `no_mulch` "no practice applied" flags.
- `src/preprocessing.py` — `build_preprocessor(kind)` returns a fit/transform pipeline:
  `"tree"` keeps native `category` dtypes for LightGBM/CatBoost/XGBoost; `"linear"`
  one-hot encodes categoricals and standardizes numerics. The clean data needs no
  imputation, skew transforms, or outlier handling.

```python
from src import load_raw, split_features_target, encode_target, build_preprocessor

train = load_raw("train")
X, y = split_features_target(train)
y = encode_target(y)
pipe = build_preprocessor("tree").fit(X, y)   # fit on train fold only
X_model = pipe.transform(X)
```

Run the tests with `python -m pytest`.

## Initial workflow

1. Explore the features and target in `notebooks/`.
2. Build reusable data loading and feature engineering in `src/`.
3. Record each run in `experiments/README.md`.
4. Save fitted estimators in `models/`.
5. Save metrics, validation predictions, and Kaggle submissions in `outputs/`.
