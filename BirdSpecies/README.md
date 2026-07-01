# BirdSpecies — MLSP 2013 Bird Classification 🐦

Solution for Kaggle [**MLSP 2013 Bird Classification Challenge**](https://www.kaggle.com/competitions/mlsp-2013-birds)
(`mlsp-2013-birds`; also listed as "Multi-label Bird Species Classification -
NIPS 2013").

- **Task:** multi-label — for each 10 s recording, predict which of **19 bird
  species** are present.
- **Metric:** **flattened-pair AUC** (one ROC-AUC over all pooled
  `recording × species` pairs).
- **Data:** 645 mono 16 kHz WAVs + organizer features (100-dim
  histogram-of-segments, 38-dim segment acoustic features). Fixed train/test
  split (~322 / ~323) via `CVfolds_2.txt`; test labels hidden.
- **Approach:** classical per-species models on provided/engineered features
  (headline), plus a mel-spectrogram transfer CNN (ensemble diversity), blended
  by rank-average.

## Data

Not committed (Kaggle terms). Download into `data/raw/`:

```bash
export KAGGLE_API_TOKEN="$(cat ~/.kaggle/access_token)"
uv run kaggle competitions download -c mlsp-2013-birds -p data/raw
unzip -o data/raw/mlsp-2013-birds.zip -d data/raw
```

## Pipeline

| module | responsibility |
|--------|----------------|
| `src/data.py`      | labels, folds, feature loaders, WAV paths, submission Ids |
| `src/cv.py`        | shared 5-fold multilabel-stratified folds (seed 42) |
| `src/features.py`  | engineered segment-feature aggregations |
| `src/audio.py` / `src/dataset.py` | WAV → mel-spec cache + torch dataset (CNN) |
| `src/train_clf.py` | per-species classical models → OOF/test probs |
| `src/train_cnn.py` | timm multi-label CNN fine-tune → OOF/test probs |
| `src/blend.py`     | rank-average ensemble |
| `src/submit.py`    | build/validate `Id,Probability` submission |

## Usage

```bash
uv sync                 # classical + torch deps
uv sync --extra audio   # + librosa/soundfile for the CNN track
uv run python -m pytest
uv run python -m src.train_clf --model lgbm
uv run python -m src.submit --name <run>   # --submit to upload (after CV approval)
```

## Results

See `experiments/README.md` (run log) and `reports/RESULTS.md` (summary). EDA in
`notebooks/01_eda.ipynb` → `reports/EDA_FINDINGS.md`.

_(populated as phases complete)_
