# BirdSpecies — MLSP 2013 Bird Classification (Kaggle `mlsp-2013-birds`)

Multi-label **audio** classification: for each 10-second recording, predict which
of **19 bird species** are present. Metric: **flattened-pair AUC** (a single
ROC-AUC over all pooled `(recording × species)` prediction pairs). Submission:
`Id,Probability` where `Id = rec_id*100 + species_idx`.

> The "Multi-label Bird Species Classification - NIPS 2013" Kaggle page is a
> mirror; the data-serving slug is **`mlsp-2013-birds`**.

## Data

Provided (both raw audio **and** organizer-precomputed features):

- `essential_data/` — `src_wavs/` (645 mono 16 kHz 10 s WAVs),
  `rec_id2filename.txt`, `species_list.txt` (19 species, idx 0–18),
  `rec_labels_test_hidden.txt` (645×19 multi-label; test rows hidden with `?`),
  `CVfolds_2.txt` (fixed train/test split, ~322 / ~323 recordings).
- `supplemental_data/` — `histogram_of_segments.txt` (100-dim bag-of-segments
  per recording, `hos_0..hos_99`), segment-level 38-dim acoustic features,
  segment rectangles / spectrogram material.

**Small-data problem** (~322 labeled recordings). The winning recipe
(gaborfodor ~0.954) leaned on hand-engineered spectrogram features + the provided
histogram → per-species RandomForest. Classical ML carries the headline score;
the CNN is ensemble diversity.

Download (Mac is fine — data is small):
```bash
export KAGGLE_API_TOKEN="$(cat ~/.kaggle/access_token)"
uv run kaggle competitions download -c mlsp-2013-birds -p data/raw
unzip -o data/raw/mlsp-2013-birds.zip -d data/raw
```

## Compute

- Classical models + EDA run on this **Mac (CPU)** — data is tiny.
- The mel-spectrogram **CNN** (Phase 5) trains on the remote **3090 Ti box**
  (`kamron@192.168.0.92`); Mac does CPU `--sample` smoke only. Code is
  device-portable (`cuda→mps→cpu` auto-detect).

## Pipeline

| module | responsibility |
|--------|----------------|
| `src/data.py`      | species list, labels, folds, histogram + segment feature loaders, WAV paths, submission-Id helpers |
| `src/cv.py`        | shared 5-fold `MultilabelStratifiedKFold` (SEED=42) over train recordings |
| `src/features.py`  | engineered per-recording features (segment-feature aggregations, optional template matching) |
| `src/audio.py`     | WAV → log-mel spectrogram cache (CNN track) |
| `src/dataset.py`   | torch multi-label Dataset over mel-specs + augmentation |
| `src/train_clf.py` | per-species classical models → OOF/test probs + flattened-pair AUC |
| `src/train_cnn.py` | timm multi-label CNN fine-tune → OOF/test probs |
| `src/blend.py`     | equal-weight rank-average of model probability arrays |
| `src/submit.py`    | build/validate `Id,Probability` submission, optional upload |

Prediction-array contract (decoupled train/submit): each model writes
`{model}_oof.npy` / `{model}_test.npy` / `{model}_metrics.json` to
`experiments/artifacts/`, aligned row-for-row via the shared folds; blend any
subset afterward. Models are **not** pickled.

## Metric detail (important)

AUC is over **pooled** `(recording, species)` pairs, so scores must be comparable
**across species**, not just within. Per-species one-vs-rest models need
cross-species calibration/rank-normalization before pooling — validate raw vs
rank-normalized on OOF and keep what wins.

## Usage

```bash
uv sync                                        # classical + torch deps
uv sync --extra audio                          # + librosa/soundfile for the CNN
uv run python -m pytest
uv run python -m src.train_clf --model lgbm    # classical baseline (add --sample for smoke)
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
uv run python -m src.blend  --models lgbm,rf,cnn
uv run python -m src.submit --name <run>       # --submit to upload (after CV approval)
```

## Workflow rules

- **Show OOF flattened-pair AUC and get explicit approval before any Kaggle
  submission.** Submit only on above-noise improvement.
- Prefer a diverse equal-weight (rank-average) blend over OOF-optimized stacking.
- Commit incrementally after each meaningful step; stage **only** `BirdSpecies/`
  files (monorepo).
- Results live in `experiments/README.md` (run log), `reports/RESULTS.md`
  (summary), `reports/EDA_FINDINGS.md` (EDA), and the project `README.md`.
