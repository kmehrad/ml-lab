# Experiment log — StudentTestScores (Kaggle PS S6E1)

Metric: **RMSE** (lower is better). CV: **KFold(n_splits=5, shuffle=True, random_state=42)**.
Submit only after CV review + approval. Naive floor (predict the train mean ≈ 62.5) = train std ≈
**18.92** RMSE. Public LB tops out ~8.57.

| ID | Date | Model | OOF RMSE | Public LB | Fold mean ± std | Notes |
|----|------|-------|---------:|----------:|-----------------|-------|
| naive | 2026-07-07 | Predict-mean | 18.92 | — | — | Floor = train-target std. |
