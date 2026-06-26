# DisasterTweets — Kaggle "NLP with Disaster Tweets"

Solution to the Kaggle competition
[`nlp-getting-started`](https://www.kaggle.com/competitions/nlp-getting-started):
classify each tweet as describing a **real disaster** (`target=1`) or **not** (`0`).

- **Task:** binary text classification.
- **Metric:** **F1 score** (hard labels → the decision threshold is tuned on OOF predictions).
- **Data:** small. `train.csv` (~7,613 rows), `test.csv` (~3,263 rows), `sample_submission.csv`.
- **CV:** `StratifiedKFold(5, shuffle=True, random_state=42)` on `target` (~43% positive); the
  same folds are shared across all models so OOF/test predictions are blend-compatible.

## Data

| Column | Notes |
|---|---|
| `id` | row id |
| `keyword` | ~221 disaster keywords, some missing |
| `location` | free-text user location, ~1/3 missing, noisy |
| `text` | the tweet (primary signal) |
| `target` | 1 = real disaster, 0 = not (train only) |

Place the Kaggle files in `data/raw/` (git-ignored):

```bash
uv sync
export KAGGLE_API_TOKEN="$(cat ~/.kaggle/access_token)"
uv run kaggle competitions download -c nlp-getting-started -p data/raw
unzip -o data/raw/nlp-getting-started.zip -d data/raw
```

## Pipeline

| Module | Responsibility |
|---|---|
| `src/data.py` | schema constants, CSV loaders + schema check |
| `src/features.py` | `clean_text`, combined keyword+text field, optional dedup/label-fix |
| `src/train.py` | classic CV (`TfidfVectorizer` → linear models); OOF **F1** + threshold tuning; saves `.npy`/`.json` |
| `src/transformer.py` | fine-tune RoBERTa / DistilBERT over the same folds; device auto-detect `cuda→mps→cpu` |
| `src/blend.py` | rank-average ensemble of saved OOF/test predictions + threshold tuning |
| `src/submit.py` | build a validated `id,target` submission CSV; optional Kaggle upload |

All models save predictions (not pickled models) to `experiments/artifacts/`:
`{model}_oof.npy`, `{model}_test.npy`, `{model}_metrics.json`.

## Usage

```bash
uv run python -m pytest
uv run python -m src.train --model tfidf --sample 1500   # quick smoke run
uv run python -m src.train --model tfidf                 # classic baseline
uv run python -m src.transformer --model roberta         # transformer (or --model distilbert)
uv run python -m src.blend
uv run python -m src.submit --model blend                # writes outputs/blend_submission.csv
```

## Results

_Pending — see `experiments/README.md` for the run history and OOF F1 table._

## EDA

`notebooks/01_eda.ipynb` (executed; figures in `reports/figures/`). Findings in
`reports/EDA_FINDINGS.md`.
