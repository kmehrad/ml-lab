# FlowerClf — Petals to the Metal 🌸

Solution for Kaggle [**Petals to the Metal - Flower Classification on TPU**](https://www.kaggle.com/competitions/tpu-getting-started)
(`tpu-getting-started`).

- **Task:** classify flower images into **104 classes**.
- **Metric:** **macro F1**.
- **Approach:** PyTorch + `timm` transfer learning (EfficientNet/ConvNeXt/ViT), fine-tuned on a GPU.
- **Data:** official TFRecords (192/224/331/512) → decoded once to JPEG + a metadata parquet.

## Data

| split | labeled | count (approx) | features |
|-------|---------|---------------:|----------|
| train | yes     | 12,753         | `image`, `class`, `id` |
| val   | yes     |  3,712         | `image`, `class`, `id` |
| test  | no      |  7,382         | `image`, `id` |

Download (run on the GPU box):

```bash
export KAGGLE_API_TOKEN="$(cat ~/.kaggle/access_token)"
kaggle competitions download -c tpu-getting-started -p data/raw
unzip -o data/raw/tpu-getting-started.zip -d data/raw
```

## Pipeline

| module | responsibility |
|--------|----------------|
| `src/data.py`    | class list, paths, metadata loader |
| `src/convert.py` | TFRecord → JPEG + metadata parquet (one-time) |
| `src/dataset.py` | torch datasets + transforms |
| `src/train.py`   | timm fine-tune, val macro-F1, save preds/metrics |
| `src/predict.py` | test softmax probs (+ TTA) |
| `src/submit.py`  | build/validate `submission.csv` |
| `src/blend.py`   | ensemble softmax probs |

## Usage

```bash
uv sync                                   # train/predict deps
uv sync --extra convert                   # + tensorflow-cpu for decode
uv run python -m src.convert --res 224 --splits train,val,test
uv run python -m src.train   --model tf_efficientnet_b3_ns --res 224
uv run python -m src.predict --run <run_id>
uv run python -m src.submit  --run <run_id>   # --submit to upload (after approval)
```

## Results

See `experiments/README.md` (run log) and `reports/RESULTS.md` (summary). EDA in
`notebooks/01_eda.ipynb` → `reports/EDA_FINDINGS.md`.
