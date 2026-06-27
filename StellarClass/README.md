# StellarClass — Kaggle "Predicting Stellar Class" (Playground Series S6E6)

3-class classification of celestial objects (**GALAXY / STAR / QSO**) from SDSS-style photometric +
spectroscopic features. Competition: [`playground-series-s6e6`](https://www.kaggle.com/competitions/playground-series-s6e6).

| | |
|---|---|
| **Task** | Multiclass classification (3 classes) |
| **Metric** | Balanced Accuracy (hard labels) |
| **Submission** | `id,class` |

## Setup

```bash
uv sync
```

## Data

```bash
export KAGGLE_API_TOKEN="$(cat ~/.kaggle/access_token)"
uv run kaggle competitions download -c playground-series-s6e6 -p data/raw
unzip -o data/raw/playground-series-s6e6.zip -d data/raw
```

Yields `train.csv`, `test.csv`, `sample_submission.csv` under `data/raw/` (git-ignored).

## Pipeline

The pipeline communicates through **saved prediction arrays** (`{model}_oof.npy` + fold-averaged
`{model}_test.npy` in `experiments/artifacts/`) so training and submission are decoupled and any
subset of models can be blended after the fact.

| Module | Responsibility |
|---|---|
| `src/data.py` | Schema constants + loaders (raise on schema mismatch); label encoder |
| `src/features.py` | Leakage-safe feature engineering (color indices, redshift transforms, sentinel handling) |
| `src/cv.py` | Shared `StratifiedKFold(5)` on `class` so OOF arrays are blend-compatible |
| `src/train.py` | GBDT CV loop → OOF + bagged test probabilities; reports balanced accuracy |
| `src/dl/train_nn.py` | GPU MLP ensemble member (remote RTX 3090 Ti), same folds, same `.npy` format |
| `src/blend.py` | Probability blend + balanced-accuracy decision (per-class prior correction) |
| `src/submit.py` | Build + validate submission CSV; optional Kaggle upload |

## Usage

```bash
uv run pytest                                  # schema + feature tests
uv run python -m src.train --model lgbm        # also: xgb, cat
uv run python -m src.blend                     # ensemble + tune decision for balanced accuracy
uv run python -m src.submit --model blend      # build outputs/blend_submission.csv
# uv run python -m src.submit --model blend --submit -m "msg"   # upload (after approval)
```

## Results

See `reports/RESULTS.md` (leaderboard + best model) and `reports/EDA_FINDINGS.md` (EDA insights).
