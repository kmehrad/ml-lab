# OptimalFertilizer — Predicting Optimal Fertilizers (Kaggle PS S5E6)

Solution to the Kaggle Playground Series competition
[`playground-series-s5e6`](https://www.kaggle.com/competitions/playground-series-s5e6):
recommend the best **`Fertilizer Name`** (7 classes) for each row of soil/crop/weather conditions.

- **Task:** multi-class classification, 7 classes
  (`Urea, DAP, 14-35-14, 28-28, 17-17-17, 20-20, 10-26-26`).
- **Metric:** **MAP@3** — submit up to 3 space-separated class names per row, best-first
  (`id,Fertilizer Name`). Per-row score is `1 / rank` if the truth is in the top 3, else `0`.
  → we need well-*ranked* class probabilities, then take the top-3.
- **Size:** train 750,000 rows · test 250,000 rows. 8 features
  (`Temperature, Humidity, Moisture`, `Soil Type`, `Crop Type`, `Nitrogen, Potassium, Phosphorous`).
- **Character:** low-signal synthetic dataset; public LB clusters tightly. Ranking quality and top-3
  tie-handling dominate — small honest CV gains only.

## Status

Steps 0–5 complete. Best submission: **proba-average(lgbm, xgb) blend** — public LB **0.34402** /
private **0.34550** (edges the xgb baseline 0.34415 / 0.34486 on private). All gains past the ~0.342
baseline are within the ±0.0005 fold-noise floor. Full summary in
[`reports/RESULTS.md`](reports/RESULTS.md).

## Setup (uv)

```bash
uv sync                                   # create env from pyproject.toml / uv.lock
export KAGGLE_API_TOKEN="$(cat ~/.kaggle/access_token)"
uv run kaggle competitions download -c playground-series-s5e6 -p data/raw
unzip -o data/raw/playground-series-s5e6.zip -d data/raw
```

## Pipeline

```bash
uv run pytest                             # MAP@3 metric + submission-format tests
uv run python -m src.train --model lgbm   # 5-fold CV -> OOF + bagged test .npy in experiments/artifacts/
uv run python -m src.blend                # equal-weight probability-average of base learners
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
