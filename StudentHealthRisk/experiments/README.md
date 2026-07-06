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
| 009 | 2026-07-05 | **Lever 1** augment +50k original: lgbm | base | 0.94918 | 0.94587 | — | — | **REJECT** −0.0004 vs base lgbm. |
| 010 | 2026-07-05 | Lever 1 augment: xgb | base | 0.94922 | 0.94852 | — | — | **REJECT** −0.0005 vs base xgb. |
| 011 | 2026-07-05 | Lever 1 augment: cat | base | 0.94906 | 0.94902 | — | — | **REJECT** −0.0002 vs base cat. Real rows have 0 missing; synthetic test is missing-heavy → off-distribution. |
| 012 | 2026-07-05 | **Lever 2** xgb trees=8000 lr=0.015 | base | 0.94903 | 0.94444 | — | — | **REJECT** −0.0007. best_iter=cap → early stop (log-loss) misaligned with balanced acc; more trees overfit log-loss. |
| 013 | 2026-07-05 | Lever 2 cat trees=8000 lr=0.015 | base | 0.94930 | 0.94922 | — | — | flat (early-stop fired ~3000); 2000-tree default already optimal. |
| 014 | 2026-07-05 | **Lever 3** decision-rule global opt + per-fold | base | 0.94980 | — | — | — | no headroom: coord-ascent already global optimum; honest per-fold 0.94959 ≈ LB 0.94953. |
| 015 | 2026-07-05 | **Lever 2b** xgb tree-count sweep (metric-aligned) | base | **0.94994** @800t | — | — | — | balanced-acc peaks at ~800 trees (t400 .94980, t800 .94994, t1200 .94989, t1600 .94981) vs 2000t .94972. Log-loss ES over-trains. **New best single.** |
| 016 | 2026-07-05 | lgbm 800 trees | base | 0.94976 | 0.94945 | — | — | +0.0002 vs 2000t — tree-count win generalizes to lgbm. |
| 017 | 2026-07-05 | cat 1000 trees | base | 0.94916 | — | — | — | worse than cat auto-ES (0.94928); cat already stops well. Keep cat baseline. |
| 018 | 2026-07-05 | blend xgb_t800+lgbm_t800 (fewer trees) | base | 0.94998 | — | — | **0.94938** | **SUBMITTED — LB REGRESSED** vs the 0.94953 baseline despite higher OOF. The tree-count OOF gain did NOT transfer: it was OOF-overfit noise, not real. |
| 022 | 2026-07-06 | **RealMLP** (pytabkit, n_cv=1, batch 2048) | base+missflag | 0.94785 | 0.86809 | — | — | Public-notebook lever: strong tabular NN, cross-family diversity. Solo weaker than GBDTs but decorrelated (fixes 5.1% of xgb errors). 12 min GPU. |
| 023 | 2026-07-06 | blend/stack xgb+lgbm+cat+realmlp | base | 0.94977 / 0.94973 | — | — | — | equal-weight blend 0.94977, LogReg stack 0.94973 — RealMLP too weak solo to help at equal weight. Need stronger RealMLP (n_cv=4) + optimal weighting. |
| 019 | 2026-07-05 | **Lever 4** TE order-2: lgbm / xgb | base+TE | 0.94938 / 0.94967 | — | — | — | **REJECT** below non-TE counterparts; 3-level cats + binned numerics already captured by trees. |
| 020 | 2026-07-05 | **Lever 5** hillclimb 6-model diverse roster | base(+TE,NN) | 0.94998 | — | — | — | TE/NN learners not selected — no diversity value; picks xgb_t800+lgbm_t800 again. |
| 021 | 2026-07-05 | **Pseudo-labeling** (177k confident test rows) | base | 0.94968 | 0.94954 | — | — | **REJECT** −0.0003; confident self-labels reinforce known patterns, add noise. |

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
- **Improvement phase (exp-009..014) — the standard levers are exhausted.** Chasing the 0.951 LB cluster:
  **(1) original-data augmentation REJECTED** — the real 50k rows are complete while the synthetic
  train/test are missing-heavy, so appending them pushes every GBDT off-distribution (−0.0002..−0.0005).
  **(2) more-trees/lower-LR REJECTED** — early stopping optimizes log-loss, misaligned with balanced
  accuracy; the 2000-tree default is already optimal (xgb overfits log-loss with more trees).
  **(3) decision-rule REJECTED** — coordinate-ascent already hits the global optimum; the honest per-fold
  estimate (0.94959) matches the LB (0.94953), so there is no hidden decision headroom. Net: the ~0.9497
  plateau is robust; reaching the 0.951 cluster (+0.0016) needs a non-obvious lever (candidates:
  combination target encoding, metric-aligned early stopping, pseudo-labeling) — higher effort, lower
  probability. Paused for a steer.
- **Improvement phase conclusion (exp-015..021) — NO real gain; all levers exhausted.** Every new-signal
  lever failed on OOF (augmentation, target encoding, diverse ensembling, pseudo-labeling). The one that
  *looked* like a win — metric-aligned tree count (0.94979 → 0.94998 OOF) — **REGRESSED on the LB (0.94938
  < 0.94953 baseline)**: the OOF gain was overfit noise that did not transfer. This is the definitive
  lesson of the phase: at this plateau the OOF↔LB coupling is at the ±0.0002 noise floor, so *no* OOF
  move below ~0.001 is trustworthy. **Best submission remains the original equal-weight 3-GBDT blend,
  public LB 0.94953.** The label is near-deterministic in a few raw features → ~0.9495 is the practical
  ceiling; the 0.951 cluster (+0.0016) uses a non-obvious trick this investigation did not find. Chasing
  it further means research, not the standard playbook.
