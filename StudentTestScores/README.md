# StudentTestScores — Predicting Student Test Scores (Kaggle PS S6E1)

Solution to the Kaggle Playground Series competition
[`playground-series-s6e1`](https://www.kaggle.com/competitions/playground-series-s6e1):
predict each student's continuous **`exam_score`** from study/lifestyle features.

- **Task:** regression (`exam_score`, range ~19.6–100, mean 62.5, std 18.9).
- **Metric:** **RMSE** (lower is better). Submission is `id,exam_score`. Public LB tops out ~8.57.
- **Size:** train 630,000 rows · test 270,000 rows. 11 features, no missing values:
  4 numeric (`age, study_hours, class_attendance, sleep_hours`) + 7 low-cardinality categoricals
  (`gender, course, internet_access, sleep_quality, study_method, facility_rating, exam_difficulty`).
- **Character:** clean synthetic dataset; native GBDT categorical handling is fine. Gate every
  FE/tuning change on OOF RMSE and keep only above-noise gains.

## Status

Baseline GBDTs (LightGBM / XGBoost / CatBoost) + equal-weight blend. See
[`reports/RESULTS.md`](reports/RESULTS.md) and the experiment log
[`experiments/README.md`](experiments/README.md).

## Setup (uv)

```bash
uv sync                                   # create env from pyproject.toml / uv.lock
export KAGGLE_API_TOKEN="$(cat ~/.kaggle/access_token)"
uv run kaggle competitions download -c playground-series-s6e1 -p data/raw
unzip -o data/raw/playground-series-s6e1.zip -d data/raw
```

## Pipeline

```bash
uv run pytest                             # RMSE metric + schema/feature tests
uv run python -m src.train --model lgbm   # 5-fold CV -> OOF + bagged test .npy in experiments/artifacts/
uv run python -m src.blend                # equal-weight average of base learners
uv run python -m src.submit --model blend # build + validate submission vs sample_submission.csv
```

Base learners write `{model}_oof.npy` + `{model}_test.npy` + `{model}_metrics.json` to
`experiments/artifacts/`, decoupling training from blending/submission.

## Layout

`src/` pipeline · `notebooks/` EDA (generated) · `reports/` findings & figures ·
`experiments/` log & artifacts · `data/` (git-ignored) · `models/` `outputs/` (git-ignored).

## Docs

- EDA findings: [`reports/EDA_FINDINGS.md`](reports/EDA_FINDINGS.md)
- Results: [`reports/RESULTS.md`](reports/RESULTS.md)
- Experiment log: [`experiments/README.md`](experiments/README.md)
