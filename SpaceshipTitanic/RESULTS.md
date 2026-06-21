# Spaceship Titanic — Analysis & Results

End-to-end record of the exploratory analysis, modeling decisions, and
experiments for the Kaggle [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic)
competition. Target: `Transported` (boolean). Metric: **accuracy**.

## 1. Data

- `train.csv`: 8,693 rows × 14 cols (incl. target) · `test.csv`: 4,277 rows × 13 cols.
- Target is **balanced** (~50.4% / 49.6%), so accuracy is appropriate and no
  resampling is needed.
- ~2% values missing, scattered across almost every column (`PassengerId` and
  `Transported` are complete).

## 2. EDA findings (`notebooks/01_eda.ipynb`)

| # | Question | Finding |
|---|----------|---------|
| 1 | Target distribution | ~50/50 balanced |
| 2 | Missing values | ~2% scattered; `PassengerId`/`Transported` complete |
| 3 | Missing = "None" | **Spend columns** → 0 (100% of CryoSleep passengers have zero spend); categoricals/`Age` are true missing |
| 4 | Skewed numerics | All 5 spend columns heavily right-skewed → `log1p`; `Age` ~symmetric |
| 5 | High cardinality | `PassengerId`, `Name`, `Cabin` are identifier-like → engineer, don't encode |
| 6 | Association with target | `CryoSleep` strongest; spend cols negative (`RoomService` r=−0.25, `Spa` −0.22, `VRDeck` −0.21); `HomePlanet`/`Destination` moderate |
| 7 | Outliers | Heavy-tailed spend maxima (genuine) → transform, not delete; `Age` fine |
| 8 | Train vs test | Same distribution (KS p≥0.26, matching proportions) — no covariate shift |

## 3. Feature engineering (`src/features.py`)

- **Cabin** → `Deck` (target-encoded, transport rate 0.20–0.73 across decks),
  `CabinNum` (numeric), `Side` (one-hot, S=0.56 vs P=0.45). Raw string dropped.
- **PassengerId** → `GroupSize`, `IsAlone` (the *count*, not the identifier).
- **Spend**: missing → 0; add `TotalSpend`; `log1p` transform.
- **CryoSleep/VIP**: inferred/defaulted from spend; genuinely-missing categoricals
  get an explicit `Unknown` category.
- **Identifiers dropped**: `PassengerId`, `Name`, `Cabin`, `Group`, `Surname`.
- **Encoding**: one-hot (low-cardinality nominal), target-encoding (`Deck`),
  optional ordinal. `TargetEncoder` cross-fitting is **seeded** for reproducible CV.

## 4. Model results (deterministic 5-fold stratified CV, accuracy)

| model | CV accuracy |
|-------|-------------|
| logreg | 0.777 |
| random forest | 0.802 |
| lgbm (default) | 0.808 |
| xgb (default) | 0.809 |
| **hgb** | **0.8134 ± 0.0055** |

An additional CatBoost experiment produced complementary errors:

| model | 5-fold accuracy | 10-fold accuracy |
|-------|----------------:|-----------------:|
| hgb | 0.8134 | 0.8148 |
| catboost | 0.8123 | 0.8156 |
| **hgb + catboost soft vote** | **0.8147** | **0.8169** |

Across three independent 5-fold seeds, the blend averaged **0.8161**, versus
0.8111 for hgb alone. The soft-voting blend is therefore the new default when
CatBoost is installed; the pipeline falls back to hgb otherwise.

## 5. Experiments that did NOT improve results

All evaluated on the same deterministic CV; the bar to beat is hgb **0.8134**.

- **Optuna hyperparameter tuning** (lgbm 60 trials, hgb 50 trials):
  best tuned lgbm 0.8136, tuned hgb 0.8127 — within ~0.007 fold std of baseline.
  *(A first round reported a misleading 0.8177 because `TargetEncoder` was
  unseeded; fixing that determinism collapsed the apparent gain.)*
- **Group-spend + family features** (`GroupTotalSpend`, `GroupMeanSpend`,
  `FamilySize`): a small **net negative** (hgb 0.8127 with vs 0.8134 without).
  Kept available via `EXPERIMENTAL_*` lists but excluded from defaults.
- **Earlier ensembles** (soft-voting / stacking over hgb/lgbm/xgb): within noise —
  voting looked best at 5-fold but the ranking flipped at 10-fold. CatBoost was
  subsequently tested and provided enough complementary signal for a more stable
  two-model blend.

**Conclusion:** the project remains near the well-known ~0.81 plateau, but the
HGB/CatBoost blend gives a small, repeatable improvement over the original
single-model baseline.

## 6. Submission

- **Previous submitted model:** default HistGradientBoosting pipeline.
- **Kaggle public leaderboard: 0.80266** (tracks the 0.8134 CV estimate; the
  small gap confirms no overfitting/leakage).
- **New candidate:** HGB/CatBoost soft-voting blend. Submitted to Kaggle on
  June 20, 2026; leaderboard score pending.
- Reproduce: `python -m src.train`.

## 7. Reproducing the analysis

```bash
pip install -r requirements.txt
kaggle competitions download -c spaceship-titanic -p data/raw && \
  unzip data/raw/spaceship-titanic.zip -d data/raw
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb  # EDA
python -m src.train --model hgb        # train + submission
python -m src.tune  --model lgbm --n-trials 60   # (optional) tuning
```
