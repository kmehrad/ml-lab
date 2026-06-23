# Diabetes Prediction Challenge — Kaggle Playground Series S5E12

Predict the **probability that a patient is diagnosed with diabetes**
(`diagnosed_diabetes`) from clinical, lifestyle, and demographic features. The
data is synthetic (700k train / 300k test rows, no missing values).

- **Task:** binary classification (probability output)
- **Metric:** ROC-AUC
- **Competition:** <https://www.kaggle.com/competitions/playground-series-s5e12>

## Data

Place the Kaggle files in `data/raw/` (`train.csv`, `test.csv`,
`sample_submission.csv`). The schema is pinned in `src/data.py` and verified on
load:

| Group | Columns |
|---|---|
| Numeric (15) | `age, alcohol_consumption_per_week, physical_activity_minutes_per_week, diet_score, sleep_hours_per_day, screen_time_hours_per_day, bmi, waist_to_hip_ratio, systolic_bp, diastolic_bp, heart_rate, cholesterol_total, hdl_cholesterol, ldl_cholesterol, triglycerides` |
| Binary (3) | `family_history_diabetes, hypertension_history, cardiovascular_history` |
| Ordinal (3) | `education_level, income_level, smoking_status` (mapped to ordered int codes) |
| Nominal (3) | `gender, ethnicity, employment_status` (one-hot encoded) |
| Target | `diagnosed_diabetes` (0/1, ~62% positive) |

```bash
kaggle competitions download -c playground-series-s5e12 -p data/raw
unzip -o data/raw/playground-series-s5e12.zip -d data/raw
```

Optional augmentation: place a schema-compatible external dataset at
`data/raw/original.csv` and pass `--use-original` to training/submission.

## Pipeline

| Module | Responsibility |
|---|---|
| `src/data.py` | Schema constants, CSV loading + schema check, train/test split, optional original-data loader |
| `src/features.py` | `DiabetesFeatureEngineer` — ordinal encoding + clinical composites (lipid ratios, blood-pressure summaries, metabolic risk, activity balance). Row-wise, no leakage |
| `src/preprocessing.py` | `build_preprocessor("tree"\|"linear")` — one-hot nominal categoricals; `linear` also `StandardScaler`s the rest |
| `src/train.py` | 5-fold `StratifiedKFold` CV; regularised configs; predicts **probabilities** (ROC-AUC); saves OOF + **bagged test** predictions. `--use-original` adds the source dataset to each *training* fold only |
| `src/blend.py` | Equal-weight **rank-average diversity blend** over base + augmented GBMs (the winning recipe; see `experiments/README.md`) |
| `src/submit.py` | Build a validated probability submission from saved bagged predictions (single model or blend), optionally upload |

## Usage

```bash
python -m pytest                                  # unit tests
python -m src.train --model lgbm --sample 50000   # quick smoke run
# winning recipe: base + original-augmented pools, then the diversity blend
python -m src.train --model all --folds 5         # base pool
python -m src.train --model all --folds 5 --use-original   # augmented pool
python -m src.blend                               # equal-weight rank-average blend
python -m src.submit --model blend                # write outputs/blend_submission.csv
python -m src.submit --model blend --submit -m "…"  # upload via Kaggle CLI
```

All commands run from the project root. CV uses `seed=42`. The target is fairly
balanced, but AUC is rank-based so models are trained **without** class
re-weighting.

## Results

Best leaderboard: **public 0.69814 / private 0.69477** (diversity blend of base +
original-augmented regularised GBMs); top of the leaderboard is 0.70504. The
**original-data concat** was the decisive lever. Key learning: local CV does
**not** predict this leaderboard (a ~0.03 gap from concept shift / intrinsic test
difficulty), so model selection here is driven by leaderboard feedback, not OOF.
See `experiments/README.md` for the full run history and findings.

## EDA

`notebooks/01_eda.ipynb` answers an 8-question framework (target balance, missing
values, distributions, feature–target association via single-feature AUC,
correlations, train/test similarity). Findings go to `reports/EDA_FINDINGS.md`.
