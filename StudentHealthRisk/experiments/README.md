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
| 007 | 2026-07-05 | blend lgbm+xgb+cat (GBDT only) | base | **0.94979** | 0.94914 | — | **0.94953** | SUBMITTED. CV↔LB gap −0.00026 (within noise) — CV tracks LB. |
| 008 | 2026-07-05 | hillclimb (all 4) | base | 0.94986 | — | — | — | picks xgb+cat 50/50; +0.00014 vs best single — still sub-noise. |
| 009 | 2026-07-06 | blend xgb+lgbm @800 trees (metric-aligned) | base | 0.94998 | — | — | **0.94938** | SUBMITTED, REGRESSED below exp-007 (0.94953) despite highest-yet OOF. Confirms OOF↔LB coupling is at the ±0.0002 noise floor for single-GBDT-family moves. |
| 010 | 2026-07-07 | hillclimb xgb+tabm+ftt (⅓ each) — cross-family | base + pytabkit TabM/FTT | **0.94994** | — | — | **0.94981** | SUBMITTED, new best. TabM/FTT are near-GBDT-strength *and* decorrelated (each fixes ~4-5% of xgb errors) — genuine cross-family diversity, not single-family noise. Gap −0.00013, transferred. **Best submission.** |
| 011 | 2026-07-13 | hillclimb xgb+tabm+cat (⅓ each) — swap ftt→cat | base + pytabkit TabM | 0.94993 | — | — | 0.94964 | SUBMITTED. OOF ties exp-010 (Δ 0.00001) but LB is 0.00017 *below* it — swapping FTT for CatBoost loses the diversity that made exp-010 work, even though OOF couldn't tell the two picks apart. Still beats exp-007/009, so cross-family (GBDT+NN) blends as a class outperform single-family blends; the *specific* pick (which NN, which GBDT) matters and isn't resolved by OOF alone at this noise floor. |

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
- **exp-009 tree-count "win" was noise.** Metric-aligned early stopping (~800 trees beats the log-loss
  optimum) lifted OOF to 0.94998 (highest yet) but LB **regressed** to 0.94938 — worse than exp-007.
  Confirms single-GBDT-family OOF deltas under ~0.0005 are not trustworthy signal here.
- **exp-010 breakthrough: cross-family diversity.** Generalizing the NN learner to a pytabkit arch zoo
  found TabM and FTT are near-GBDT-strength *and* genuinely decorrelated (unlike the earlier weak plain
  RealMLP). Hillclimb xgb+tabm+ftt reached OOF 0.94994 → **LB 0.94981**, the first improvement that
  actually transferred (gap only −0.00013). This is the current best submission.
- **exp-011 pick sensitivity.** Swapping CatBoost in for FTT (xgb+tabm+cat) ties exp-010 on OOF (within
  0.00001) but lands at LB 0.94964 — 0.00017 *worse*. So it's not "any 3-model cross-family hillclimb
  ties the best": the specific FTT pick matters, and OOF at this saturation level can't distinguish which
  pick will generalize. Both cross-family blends still beat every single-GBDT-family blend (exp-007/009),
  so **the actionable hint is: cross-family diversity (GBDT + independently-trained NN architectures) is
  the real lever left; the next experiments should add more/different diverse members (e.g. TabR once its
  "0 features" bug is fixed, a second NN seed/arch, deeper stacking) rather than re-tuning single models
  or re-picking among the same 5 base learners.**
