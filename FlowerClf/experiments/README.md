# Experiments — FlowerClf

Run log. Each `src/train.py` run appends a row and writes artifacts to
`experiments/artifacts/` (`{run}_val_probs.npy`, `{run}_test_probs.npy`, `{run}_metrics.json`).

Validation = provided `val` split (macro F1) until merged-K-fold is added.

| ID | Date | Model | Res | Val macro-F1 | Public LB | Epochs | Notes |
|----|------|-------|----:|-------------:|----------:|-------:|-------|
| tf_efficientnet_b3_ns_r224 | 2026-06-26 | tf_efficientnet_b3_ns | 224 | 0.93969 |  | 15 | Baseline. 278s on 3090 Ti. per-class mean 0.940; 1 class <0.80 (sweet pea 0.286). |
