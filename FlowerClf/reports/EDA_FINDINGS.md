# EDA findings — FlowerClf (Petals to the Metal)

Source: `notebooks/01_eda.ipynb` (run on the GPU box after `src/convert.py`).
Metric is **macro F1**, so per-class (especially rare-class) performance matters more than
overall accuracy.

> **TODO (fill after running the notebook on the converted data):**
> - Classes present: __ / 104
> - Per-class counts — min / median / max: __ / __ / __
> - Imbalance ratio (max/min): __
> - 10 rarest classes: __
> - Val fraction per class — mean / min / max: __ / __ / __  (is val a consistent slice?)
> - Image sizes (should match the chosen resolution, e.g. 224×224): __

## Expected (from competition docs, to confirm)

- ~12,753 train + ~3,712 val labeled images; ~7,382 unlabeled test.
- 104 flower classes; known to be **imbalanced** → likely need one of:
  inverse-frequency `WeightedRandomSampler` (`--balanced-sampler`) or class-balanced loss,
  plus heavy augmentation (`--auto-augment`, mixup) to lift rare-class F1.

## Implications for modeling

- Start: `tf_efficientnet_b3_ns @224`, label smoothing 0.1, RandAugment, cosine schedule.
- If rare-class F1 is low → enable `--balanced-sampler` and/or `--mixup 0.2`.
- Try 512-res / larger backbone for the final push; ensemble via `src/blend.py`.
