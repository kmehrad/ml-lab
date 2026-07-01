# Results — BirdSpecies (MLSP 2013 Birds)

Metric: **flattened-pair AUC** (one ROC-AUC over all pooled `recording × species`
pairs). Full run log with hyperparameters: `experiments/README.md`.

## Summary

| Model | Features | OOF AUC | Public LB | Private LB | Notes |
|-------|----------|--------:|----------:|-----------:|-------|
| _(pending)_ | | | | | |

## Notes

- Submissions are gated: report OOF flattened-pair AUC → get approval → upload.
- Adopt a blend only if it beats the best single member by more than fold noise.
- Cross-species probability calibration matters (pooled-pair metric) — validated
  raw vs rank-normalized on OOF.

## Key findings

_(populated as phases complete)_
