# Results — FlowerClf (Petals to the Metal)

Metric: **macro F1**. Validation = provided `val` split until merged-K-fold is added.
Full run log with hyperparameters: `experiments/README.md`.

## Summary

| Model | Res | Val macro-F1 | Public LB | Notes |
|-------|----:|-------------:|----------:|-------|
| _(baseline pending)_ | 224 | | | tf_efficientnet_b3_ns, RandAugment, ls=0.1 |

## Notes

- Submissions are gated: report val macro-F1 → get approval → upload (per workflow rules).
- Adopt a blend only if it beats the best single member by more than fold/run noise.

## Next ideas

- Rare-class handling: `--balanced-sampler`, `--mixup`, class-balanced loss.
- Higher resolution (331/512) and larger / alternative backbones (ConvNeXt, ViT).
- Test-time augmentation (`--tta`) and multi-model `src/blend.py` ensembling.
