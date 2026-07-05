# Experiment log — StudentHealthRisk (PS S6E7)

Metric: **Balanced Accuracy Score** (mean per-class recall). Always report the **OOF-tuned** score
(decision weights tuned on OOF via `metric.tune_weights`), and the raw-argmax score for reference.
Every run gets a row; rejected experiments are kept with the reason. Gate every change on OOF; keep only
above-fold-noise gains.

| ID | Date | Model / change | Features | OOF tuned | OOF raw | Fold mean±std (raw) | Public LB | Notes |
|----|------|----------------|----------|-----------|---------|---------------------|-----------|-------|
| smoke | 2026-07-05 | lgbm (40k sample, 3-fold) | base | 0.94576 | 0.91535 | 0.91535 ± 0.00410 | — | pipeline smoke test; not an official run |
| 001 | 2026-07-05 | lgbm baseline (5-fold, full) | base | **0.94956** | 0.94602 | 0.94602 ± 0.00147 | — | reference. decision weights [0.412,1.422,1.166]. `best_iter=2000` hit cap → early stopping never fired, more trees may help. Noise floor ≈0.0015. |

## Verdicts / narrative

- **Setup (Steps 0–2).** Scaffolded the `OptimalFertilizer/`-style prediction-file pipeline; confirmed
  schema + metric (balanced accuracy, 3-class 86/8/6 imbalance, label submission). Smoke test shows the
  features are strongly predictive and OOF decision tuning adds a large lift over raw argmax (~+0.03 on
  the sample) — expected, since balanced accuracy is decision-sensitive under heavy imbalance.
- **exp-001 lgbm baseline.** OOF tuned **0.94956** (raw 0.94602), folds tight at ±0.00147. Decision
  tuning adds +0.0035 over raw argmax. `best_iter` pinned at the 2000-tree cap on every fold → the model
  is not yet overfitting; **raising `--trees` (and/or lowering LR) is the first Step-5 lever to test**.
  Fold std ≈0.0015 sets the "above-noise" bar for accepting any change.
