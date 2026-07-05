# StudentHealthRisk — Predicting Student Health Risk (Kaggle PS S6E7)

Solution to the Kaggle Playground Series competition
[`playground-series-s6e7`](https://www.kaggle.com/competitions/playground-series-s6e7):
predict a student's **health risk** from behavioural/lifestyle features (tabular classification).

- **Task:** tabular classification. Target, class cardinality (binary vs multiclass) and the exact
  metric are confirmed from the real CSVs + the competition Evaluation tab during setup (see
  `reports/EDA_FINDINGS.md` / `CLAUDE.md`).
- **Approach:** GBDT ensemble (LightGBM + XGBoost + CatBoost) plus a tabular NN (MLP) for blend
  diversity; equal-weight blend / greedy hill-climb over saved OOF/test prediction arrays.
- **Compute:** CV/blend/submit run locally; the NN and GPU GBDT/target-encoding work run on the remote
  RTX 3090 Ti box via `scripts/remote_run.sh` (the Mac has no CUDA).

## Setup (uv)

```bash
uv sync                                   # create env from pyproject.toml / uv.lock
export KAGGLE_API_TOKEN="$(cat ~/.kaggle/access_token)"
uv run kaggle competitions download -c playground-series-s6e7 -p data/raw
unzip -o data/raw/playground-series-s6e7.zip -d data/raw
```

## Pipeline

```bash
uv run pytest                             # metric + submission-format tests
uv run python -m src.train --model lgbm   # 5-fold CV -> OOF + bagged test .npy in experiments/artifacts/
uv run python -m src.blend                # equal-weight blend of base learners
uv run python -m src.submit --model blend # build + validate submission vs sample_submission.csv
```

Base learners write `{model}_oof.npy` + `{model}_test.npy` + `{model}_metrics.json` to
`experiments/artifacts/`, decoupling training from blending/submission.

## Remote GPU (RTX 3090 Ti box)

```bash
scripts/remote_run.sh push
scripts/remote_run.sh run  python -m src.models_nn --seeds 10 --tag nn_s10
scripts/remote_run.sh pull
```

## Layout

`src/` pipeline · `notebooks/` EDA (generated) · `reports/` findings & figures ·
`experiments/` log & artifacts · `data/` (git-ignored) · `models/` `outputs/` (git-ignored).

## Docs

- EDA findings: [`reports/EDA_FINDINGS.md`](reports/EDA_FINDINGS.md)
- Results: [`reports/RESULTS.md`](reports/RESULTS.md)
- Experiment log: [`experiments/README.md`](experiments/README.md)
