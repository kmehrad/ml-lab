# BirdSpecies — NIPS4B 2013 Bird Challenge (Kaggle `multilabel-bird-species-classification-nips2013`)

Multi-label **audio** classification: for each recording, predict which of **87
classes** (species × vocalization type — e.g. `Aegcau_call` vs `Aegcau_song`
count separately) are present. Metric: **AUC** — a single ROC-AUC over all pooled
`(test file × class)` prediction pairs. Submission: `ID,Probability` with
`ID = nips4b_birds_testfile{NNNN}.wav_classnumber_{C}` (C = 1..87 → 87,000 rows).

> This is the **NIPS4B 2013** challenge (the competition the Kaggle "…NIPS 2013"
> page actually serves). It is **not** the MLSP-2013 `mlsp-2013-birds` comp.

## Data (`data/raw/`, extracted from the competition zip)

- `NIPS4B_BIRD_CHALLENGE_TRAIN_LABELS/` — `numero_file_train.csv` (687×89:
  file# + 87 class 0/1 columns + duration), `nips4b_birdchallenge_espece_list.csv`
  (87 classes + names/taxon; plus an `Empty` sentinel = no animal),
  `example_NIPS4B13_submission_format_test_prediction.csv` (87,000-row template).
- `NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_WAV/{train,test}/` — **687 train + 1000
  test** WAVs, 44.1 kHz mono 16-bit, ≤5 s (median 5.0 s, min ~1.0 s).
- `NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_MFCC/{train,test}/` — precomputed cepstra,
  one `.txt` per file, **17 coefficients × N frames** (variable length).

**Key facts (from EDA sanity):** 87 classes, mean **1.71** classes/clip (max 6);
**14.6% of clips are empty** (all-zero); class prevalence 7–20 per class (~3×
imbalance). Test labels are hidden → we make our own CV on the 687 train files.

Download is manual (older comp needs rules acceptance). The zip
`multilabel-bird-species-classification-nips2013.zip` was placed in `data/raw/`
and extracted; it contains three archives (LABELS `.tar`, WAV/MFCC `.tar.gz`).

## Compute

- Classical models + EDA run on this **Mac (CPU)** — data is small.
- The mel-spectrogram **CNN** (Phase 5) trains on the remote **3090 Ti box**
  (`kamron@192.168.0.92`); Mac does CPU `--sample` smoke only. Code is
  device-portable (`cuda→mps→cpu`).

## Pipeline

| module | responsibility |
|--------|----------------|
| `src/data.py`      | class list, train labels/durations, MFCC + WAV path/loaders, submission-id helpers |
| `src/cv.py`        | shared 5-fold `MultilabelStratifiedKFold` (SEED=42) over the 687 train files |
| `src/features.py`  | fixed-length features per file (MFCC time-aggregations; mel-spec stats) |
| `src/audio.py`     | WAV → log-mel spectrogram cache (CNN track) |
| `src/dataset.py`   | torch multi-label Dataset over mel-specs + augmentation |
| `src/train_clf.py` | per-class classical models on aggregated features → OOF/test probs + AUC |
| `src/train_cnn.py` | timm multi-label CNN fine-tune → OOF/test probs |
| `src/blend.py`     | equal-weight rank-average of model probability arrays |
| `src/submit.py`    | build/validate the 87k-row `ID,Probability` submission, optional upload |

Prediction-array contract: each model writes `{model}_oof.npy` (687×87) /
`{model}_test.npy` (1000×87) / `{model}_metrics.json` to `experiments/artifacts/`,
aligned via the shared folds; blend any subset afterward. Models are not pickled.

## Metric detail (important)

AUC is over **pooled** `(file, class)` pairs, so scores must be comparable
**across classes**, not just within. Per-class one-vs-rest models need
cross-class calibration/rank-normalization before pooling — validate raw vs
rank-normalized on OOF and keep what wins. The 14.6% empty clips (all-zero
targets) are legitimate negatives across all 87 classes.

## Usage

```bash
uv sync                                        # classical + torch deps
uv sync --extra audio                          # + librosa/soundfile for the CNN
uv run python -m pytest
uv run python -m src.train_clf --model lgbm    # classical baseline (--sample for smoke)
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
uv run python -m src.blend  --models lgbm,rf,cnn
uv run python -m src.submit --name <run>       # --submit to upload (after CV approval)
```

## Workflow rules

- **Show OOF AUC and get explicit approval before any Kaggle submission.** Submit
  only on above-noise improvement.
- Prefer a diverse equal-weight (rank-average) blend over OOF-optimized stacking.
- Commit incrementally after each meaningful step; stage **only** `BirdSpecies/`
  files (monorepo).
- Results: `experiments/README.md` (run log), `reports/RESULTS.md` (summary),
  `reports/EDA_FINDINGS.md` (EDA), project `README.md`.

## Gotchas

- `kagglesdk==0.1.32` breaks `import kaggle`; pinned to `0.1.30` in pyproject.
- MFCC files are stored **coeffs × frames**; `data.load_mfcc` transposes to
  frames × 17.
