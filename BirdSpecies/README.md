# BirdSpecies — NIPS4B 2013 Bird Challenge 🐦

Solution for Kaggle [**Multi-label Bird Species Classification – NIPS 2013**](https://www.kaggle.com/competitions/multilabel-bird-species-classification-nips2013)
(the **NIPS4B 2013** bird-song challenge).

- **Task:** multi-label — for each recording, predict which of **87 classes**
  (species × call/song) are present.
- **Metric:** **AUC** (one ROC-AUC over all pooled `test file × class` pairs).
- **Data:** **687 train + 1000 test** WAVs (44.1 kHz mono, ≤5 s) + precomputed
  MFCC cepstra (17 coeffs × N frames). Test labels hidden.
- **Approach:** classical per-class models on aggregated MFCC/spectrogram
  features (headline), plus a mel-spectrogram transfer CNN (ensemble diversity),
  blended by rank-average.

## Data

Not committed (Kaggle terms). Older comp → download needs rules acceptance on the
website, then place the zip in `data/raw/` and extract:

```bash
# after accepting rules at the competition page:
export KAGGLE_API_TOKEN="$(cat ~/.kaggle/access_token)"
uv run kaggle competitions download -c multilabel-bird-species-classification-nips2013 -p data/raw
cd data/raw && unzip -o *.zip
for t in NIPS4B_BIRD_CHALLENGE_*.tar*; do tar -xf "$t"; done
```

Layout after extraction: `NIPS4B_BIRD_CHALLENGE_TRAIN_LABELS/`,
`…_TRAIN_TEST_WAV/{train,test}/`, `…_TRAIN_TEST_MFCC/{train,test}/`.

## Pipeline

| module | responsibility |
|--------|----------------|
| `src/data.py`      | classes, labels, MFCC/WAV loaders, submission ids |
| `src/cv.py`        | shared 5-fold multilabel-stratified folds (seed 42) |
| `src/features.py`  | fixed-length per-file features (MFCC aggregations) |
| `src/audio.py` / `src/dataset.py` | WAV → mel-spec cache + torch dataset (CNN) |
| `src/train_clf.py` | per-class classical models → OOF/test probs |
| `src/train_cnn.py` | timm multi-label CNN fine-tune → OOF/test probs |
| `src/blend.py`     | rank-average ensemble |
| `src/submit.py`    | build/validate the 87k-row `ID,Probability` submission |

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
