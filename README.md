# ml-lab

A lab repository for machine-learning experiments. The first project is two
iterations of a solution to the Kaggle
[House Prices – Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
competition (tabular regression, metric = **RMSLE**).

```
ml-lab/
├── HousePrice/      # v1 — first pass (ENet/LGBM/XGB blend)
└── HousePrice-v2/   # v2 — modular pipeline, feature engineering, stacking
```

The competition data is **not** committed (Kaggle terms / size). Download it
into each project's `data/raw/` with the Kaggle CLI:

```bash
kaggle competitions download -c house-prices-advanced-regression-techniques -p HousePrice-v2/data/raw
unzip HousePrice-v2/data/raw/*.zip -d HousePrice-v2/data/raw
```

---

## HousePrice-v2 — pipeline overview

A clean, modular pipeline (see `HousePrice-v2/CLAUDE.md` for architecture):

- **`src/data.py`** — load train/test, write submissions.
- **`src/features.py`** — `preprocess()` combines train+test (no target leak),
  handles structural missingness, log-transforms skewed features and the
  target, engineers features, and one-hot encodes.
- **`src/models.py`** — model registry (Ridge/Lasso/ElasticNet/RF/GBM +
  optional XGBoost/LightGBM/CatBoost).
- **`src/evaluate.py`** — RMSLE, k-fold CV, out-of-fold predictions.
- **`src/benchmark.py`** — three-tier model benchmark.
- **`src/ensemble.py`** — OOF stacking/blending (NNLS + Ridge meta-learner)
  with fold-aware target encoding.

All scoring is **5-fold CV (shuffle, seed 42), RMSE on `log1p(SalePrice)`** =
the competition RMSLE.

---

## Results

### Single-model benchmark (5-fold CV RMSLE)

| Tier | Model | CV RMSLE |
|---|---|---|
| 1 | **ElasticNet** | **0.11065** |
| 1 | Lasso | 0.11279 |
| 1 | Ridge | 0.11336 |
| 1 | GradientBoosting | 0.12282 |
| 1 | RandomForest | 0.13652 |
| 2 | **CatBoost** | **0.11252** |
| 2 | XGBoost | 0.11607 |
| 2 | LightGBM | 0.12581 |

### Kaggle submissions (public leaderboard)

| Model | CV RMSLE | Public LB |
|---|---|---|
| ENet + LGBM + XGB NNLS blend (v1) | 0.11101 | **0.12210** ← best LB |
| 7-model NNLS stack, clean one-hot (v2) | **0.11062** ← best CV | 0.12540 |
| 7-model NNLS stack, ordinal + target-encoded (v2) | 0.11108 | 0.12437 |
| ElasticNet (v2) | 0.11065 | 0.12656 |
| CatBoost (v2) | 0.11252 | 0.12868 |

---

## Key findings

1. **Log-transform the target.** `SalePrice` is right-skewed (skew ≈ 1.88);
   `log1p` makes it near-normal (≈ 0.12) and aligns plain RMSE with RMSLE.
2. **Most missingness is structural** ("no basement/garage/pool"), so it's
   filled with `"None"`/`0` rather than imputed. Only `LotFrontage`/`Electrical`
   are genuinely missing.
3. **Regularized linear models are the surprise leaders.** After one-hot +
   log-skew, the dataset is very linear-friendly — ElasticNet (0.11065) beats
   every tree model and the averaging ensemble.
4. **Ordinal encoding of quality grades and target-encoding `Neighborhood`
   *hurt* the linear models** (ElasticNet 0.110 → 0.123): collapsing the
   high-dimensional one-hot signal removed exactly what the linear learners
   relied on. Tree models were indifferent. Net: not worth it here.
5. **Stacking matched but didn't beat a good blend.** A 7-model NNLS stack
   reached the best CV (0.11062) but couldn't out-run signal lost upstream.
6. **CV and the public LB are decorrelated.** The best-CV model scored among
   the *worst* on the public LB, and a middling-CV blend leads the board. The
   public LB is only ~730 rows (half the test set) — small enough that ±0.003
   swings are noise. **The 5-fold CV is the more trustworthy estimate;
   over-tuning to the public LB is itself overfitting.**

### Next steps (not yet done)
- Fix a scaling/regularization mismatch: linear models in the stacking pipeline
  are `StandardScaler`-ed but use `alpha`s tuned for unscaled data — retune with
  `RidgeCV`/`LassoCV`/`ElasticNetCV` to recover ~0.110 and strengthen the blend.
- Select final Kaggle entries by CV, not public LB.

---

## Setup

```bash
cd HousePrice-v2
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.benchmark          # benchmark models
python -m src.ensemble --submit  # build + write the stacking submission
```
