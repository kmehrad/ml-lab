# Experiment log — StudentHealthRisk (PS S6E7)

Metric: **Balanced Accuracy Score** (mean per-class recall). Always report the **OOF-tuned** score
(decision weights tuned on OOF via `metric.tune_weights`), and the raw-argmax score for reference.
Every run gets a row; rejected experiments are kept with the reason. Gate every change on OOF; keep only
above-fold-noise gains.

| ID | Date | Model / change | Features | OOF tuned | OOF raw | Fold mean±std (raw) | Public LB | Notes |
|----|------|----------------|----------|-----------|---------|---------------------|-----------|-------|
| smoke | 2026-07-05 | lgbm (40k sample, 3-fold) | base | 0.94576 | 0.91535 | 0.91535 ± 0.00410 | — | pipeline smoke test; not an official run |
| 001 | 2026-07-05 | lgbm baseline (5-fold, full) | base | **0.94956** | 0.94602 | 0.94602 ± 0.00147 | — | reference. decision weights [0.412,1.422,1.166]. `best_iter=2000` hit cap → early stopping never fired, more trees may help. Noise floor ≈0.0015. |
| 002 | 2026-07-05 | xgb (GPU, balanced sample-wt) | base | 0.94972 | 0.94883 | 0.94883 ± 0.00151 | — | GPU box, 79s. Ties lgbm within noise. |
| 003 | 2026-07-05 | catboost (GPU, auto-balanced) | base | 0.94928 | 0.94924 | 0.94924 ± 0.00119 | — | GPU box, 84s. raw≈tuned (already balanced). |
| 004 | 2026-07-05 | MLP nn seeds=5 (GPU) | base | 0.94805 | 0.94804 | — | — | GPU box, 127s. Lower but diverse; class-weighted CE. |
| 005 | 2026-07-05 | FE test: lgbm +ratio+missflag | base+ratio+missflag | 0.94935 | 0.94433 | — | — | **REJECT** — below base lgbm (0.94956); raw features already saturate the signal. |
| 006 | 2026-07-05 | blend lgbm+xgb+cat+nn (proba avg) | base | 0.94976 | 0.94944 | — | — | +0.00004 vs best base — noise. |
| 007 | 2026-07-05 | blend lgbm+xgb+cat (GBDT only) | base | **0.94979** | 0.94914 | — | — | +0.00007 vs best base — noise. Candidate submission (equal-weight generalizes). |
| 008 | 2026-07-05 | hillclimb (all 4) | base | 0.94986 | — | — | — | picks xgb+cat 50/50; +0.00014 vs best single — still sub-noise. |

## Verdicts / narrative

- **Setup (Steps 0–2).** Scaffolded the `OptimalFertilizer/`-style prediction-file pipeline; confirmed
  schema + metric (balanced accuracy, 3-class 86/8/6 imbalance, label submission). Smoke test shows the
  features are strongly predictive and OOF decision tuning adds a large lift over raw argmax (~+0.03 on
  the sample) — expected, since balanced accuracy is decision-sensitive under heavy imbalance.
- **exp-001 lgbm baseline.** OOF tuned **0.94956** (raw 0.94602), folds tight at ±0.00147. Decision
  tuning adds +0.0035 over raw argmax. `best_iter` pinned at the 2000-tree cap on every fold → the model
  is not yet overfitting; **raising `--trees` (and/or lowering LR) is the first Step-5 lever to test**.
  Fold std ≈0.0015 sets the "above-noise" bar for accepting any change.
- **exp-002..008 roster / FE / ensembling — signal is saturated.** All three GBDTs land within noise
  (xgb 0.94972, lgbm 0.94956, cat 0.94928); the class-weighted MLP is a touch lower (0.94805) but
  diverse. **FE (ratio+missflag) does not help** (0.94935 < base) — the raw features already capture the
  strong `stress_level`/`sleep_duration` signal. **Blending/ensembling adds nothing above noise**: GBDT
  blend 0.94979, hillclimb xgb+cat 0.94986, both within ±0.0002 of single xgb. The models are highly
  correlated because the label is nearly a deterministic function of a few features (0 label conflicts in
  EDA). **Conclusion: the ~0.9497 plateau is the ceiling for this feature set;** further single-model
  tuning (more trees) is the only remaining lever and unlikely to move the plateau. Candidate submission
  = equal-weight 3-GBDT blend (0.94979) — generalizes better than the OOF-overfit hillclimb pick.
