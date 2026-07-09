# EDA findings — StudentTestScores (Kaggle PS S6E1)

_Generated from `notebooks/01_eda.ipynb` (built by `notebooks/eda_build.py`). Figures in
`reports/figures/`._

## Data quality
- **630,000 train / 270,000 test rows. Zero missing cells. Zero duplicate feature rows.**
- **No train/test drift:** numeric means match to 3 decimals; every categorical level share differs
  by ≤0.2% between splits. → plain `KFold` CV should track the leaderboard closely.

## Target (`exam_score`)
- Near-symmetric, mild spread: mean **62.51**, std **18.92**, range **19.60–100.0**, skew −0.05,
  kurtosis −0.62. No log/transform needed.
- **Naive floor = predict the mean → RMSE 18.92** (= target std). Public LB tops ~8.57, so a good
  model roughly halves the naive error.

## Feature → target signal (this is a *high-signal* dataset, unlike the sibling synthetic tasks)
Ranked by mutual information / single-feature "predict the binned mean" RMSE:

| Feature | Pearson r | MI | Single-feature RMSE | Read |
|---|---:|---:|---:|---|
| **study_hours** | **0.762** | **0.912** | **12.30** | Dominant. Alone cuts RMSE 18.92→12.30. |
| class_attendance | 0.361 | 0.302 | 17.64 | Clear second signal. |
| study_method | — | 0.042 | 18.44 | Categorical, ~11.6-pt mean spread across levels. |
| sleep_quality | — | 0.037 | 18.38 | Categorical, ~10.9-pt spread. |
| facility_rating | — | 0.031 | 18.58 | Categorical, ~8.8-pt spread. |
| sleep_hours | 0.167 | 0.073 | 18.65 | Mild. |
| age / course / gender / exam_difficulty / internet_access | ~0 | ≤0.018 | ≈18.9 | **~No signal.** |

- **`study_hours` is by far the strongest predictor** (r 0.76, MI 0.91). `class_attendance` is the
  clear second; `study_method`, `sleep_quality`, `facility_rating` add moderate categorical signal.
- **`age`, `course`, `gender`, `exam_difficulty`, `internet_access` carry essentially no marginal
  signal** — GBDTs will down-weight them; they mainly add (small) overfit surface. Notably
  `exam_difficulty` looks inert (0.46-pt spread), which is counter-intuitive but consistent across MI
  and group-means.

## Modelling implications
- Real, mostly-monotone signal → GBDTs should do well; expect OOF RMSE in the mid-8s (public LB ~8.5–9).
- Interactions/ratios among the informative numerics (`study_hours × class_attendance`,
  `study_hours / sleep_hours`) are the most plausible FE lever — but gate on OOF RMSE; axis-aligned
  trees already capture much of it. The near-zero-signal features are candidates to consider dropping
  only if it measurably helps OOF (usually it won't with GBDTs).
- No drift + no missing + no duplicates → the CV↔LB gap should be small; iterate on 5-fold OOF RMSE.

## Figures (`reports/figures/`)
`target_distribution.png` · `numeric_distributions.png` · `numeric_correlation.png` ·
`per_category_means.png` · `mutual_information.png` · `train_test_drift.png`
