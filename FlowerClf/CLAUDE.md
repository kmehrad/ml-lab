# FlowerClf — Petals to the Metal (Kaggle `tpu-getting-started`)

104-class flower **image** classification. Metric: **macro F1**. Submission: `id,label`.

Official data is **TFRecords** (designed for TPU + Keras) at 4 resolutions
(192/224/331/512). We do **PyTorch + timm transfer learning on a GPU** instead.

## Compute

- The 3090 Ti is on a **remote Linux box (SSH)**. This Mac is **dev/edit + CPU smoke-test only**
  (no CUDA here). Download / convert / train run on the remote box.
- Code is device-portable (`cuda → mps → cpu` auto-detect). Use `--sample` for tiny CPU runs.

## Commands

```bash
uv sync                                   # train/predict deps (TF-free)
uv sync --extra convert                   # adds tensorflow-cpu for the one-time decode
uv run python -m pytest

# Download (remote box)
export KAGGLE_API_TOKEN="$(cat ~/.kaggle/access_token)"
uv run kaggle competitions download -c tpu-getting-started -p data/raw
unzip -o data/raw/tpu-getting-started.zip -d data/raw

# One-time TFRecord -> JPEG + metadata parquet
uv run python -m src.convert --res 224 --splits train,val,test

# Train / predict / submit
uv run python -m src.train   --model tf_efficientnet_b3_ns --res 224
uv run python -m src.predict --run <run_id>
uv run python -m src.submit  --run <run_id>          # add --submit to upload (AFTER approval)
```

## Architecture (`src/`)

- `data.py`   — `CLASSES` (104), counts, paths, `load_metadata(res)`.
- `convert.py`— TFRecord → JPEG (raw bytes) + `metadata_{res}.parquet` (one-time, needs TF).
- `dataset.py`— torch Dataset + train/eval transforms, label maps.
- `train.py`  — timm backbone fine-tune; AMP; val macro-F1; saves `{run}_val_probs.npy`,
  `{run}_test_probs.npy`, `{run}_metrics.json`; appends to `experiments/README.md`.
- `predict.py`— best ckpt → test softmax probs (+ optional hflip TTA).
- `submit.py` — argmax → `outputs/{run}_submission.csv`; validate vs `sample_submission.csv`.
- `blend.py`  — average softmax probs across runs/folds → argmax.

## Workflow rules

- **Commit after every meaningful step** (descriptive messages).
- **Show val macro-F1 (CV/OOF) and wait for approval before any Kaggle submission.**
- **Submit only on above-noise improvement.**
- Validation: provided `val` split as holdout first; merged train+val StratifiedKFold later.
