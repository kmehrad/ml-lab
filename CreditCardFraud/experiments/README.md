# Experiment Log

**AUPRC (average precision)** is the primary metric — `oof_ap` is on 5-fold
out-of-fold predictions over the train remainder (`seed=42`); `test_ap` is on the
fixed stratified 20% hold-out. Higher is better. The random-AP floor is **0.0017**
(the 0.173% positive rate). ROC-AUC is recorded but is optimistic here. Only adopt
a change if its OOF AUPRC beats the current best by more than `oof_ap_fold_std`
(~0.045 — folds are noisy because the hold-out test has only ~98 frauds).

## Model comparison (`--imbalance none` unless noted)

| Date | Model | OOF AUPRC | Fold mean ± std | OOF ROC-AUC | Hold-out AUPRC | Notes |
|------|-------|----------:|-----------------|------------:|---------------:|-------|
| 2026-06-23 | **catboost** | **0.85128** | 0.851 ± 0.046 | 0.98328 | **0.87734** | best single |
| 2026-06-23 | xgb | 0.84463 | 0.845 ± 0.046 | 0.98465 | 0.87498 | |
| 2026-06-23 | xgb (classweight) | 0.84735 | 0.849 ± 0.041 | 0.98543 | 0.88264 | ~flat OOF |
| 2026-06-23 | lgbm | 0.84399 | 0.844 ± 0.046 | 0.98543 | 0.87443 | |
| 2026-06-23 | lgbm (classweight) | 0.84380 | 0.845 ± 0.045 | 0.98485 | 0.88143 | ~flat OOF |
| 2026-06-23 | catboost (classweight) | 0.82879 | 0.828 ± 0.042 | 0.98198 | 0.86764 | reweighting **hurt** |
| 2026-06-23 | histgb | 0.82360 | 0.826 ± 0.047 | 0.98131 | 0.86618 | |
| 2026-06-23 | histgb (classweight) | 0.74459 | 0.747 ± 0.041 | 0.96204 | 0.74719 | reweighting **hurt** badly |
| 2026-06-23 | logreg | 0.75971 | 0.760 ± 0.044 | 0.97948 | 0.73803 | scaled features |
| 2026-06-23 | logreg (classweight) | 0.74776 | 0.748 ± 0.044 | 0.97979 | 0.72438 | reweighting hurt |
| 2026-06-23 | blend (5: GBMs+logreg) | 0.84737 | — | — | 0.86819 | no lift; logreg drags it |
| 2026-06-23 | blend (4 GBMs) | 0.84790 | — | — | 0.87553 | still < catboost OOF |

### Unsupervised baselines (no OOF; hold-out only)

| Model | Hold-out AUPRC | Hold-out ROC-AUC | secs | Notes |
|-------|---------------:|-----------------:|-----:|-------|
| GaussianMixture (8, full) | **0.35946** | 0.93933 | 24 | best unsupervised; density of a multimodal normal manifold |
| SGD-One-Class SVM (Nystroem RBF) | 0.22075 | 0.94492 | 1 | scalable OCSVM; signed distance to boundary |
| PCA reconstruction (15 comp) | 0.18445 | 0.95786 | 0 | squared reconstruction error |
| IsolationForest | 0.16258 | 0.95444 | 2 | contamination = train positive rate |
| EllipticEnvelope (robust Gaussian) | 0.06047 | 0.92903 | 8 | single-mode MCD; normal data is *not* one ellipse |
| LocalOutlierFactor | 0.00351 | 0.51398 | 17 | ≈ random; struggles in 30-d PCA space |

**Findings:**
- **CatBoost is the best single model** (OOF AUPRC 0.851). The four GBMs cluster
  within one fold-std and rank-correlate highly, so an equal-weight rank-average
  blend does **not** beat CatBoost on OOF. Logreg is materially weaker and drags
  the 5-member blend below the GBM-only blend.
- **Class weighting does not help AUPRC and often hurts.** It is ~flat for LightGBM
  and XGBoost (within fold-std) but clearly *hurts* CatBoost (0.851 → 0.829) and
  HistGB (0.824 → 0.745), and hurts logreg. AUPRC is rank-based, so reweighting mainly
  moves the decision threshold and distorts the leaf/split statistics rather than
  improving ranking → **handle imbalance at the threshold, not by reweighting.**
- **Unsupervised anomaly detection is far behind** the supervised models. Among the
  six detectors, **GaussianMixture density is best (0.359)** — well above IForest
  (0.163) but still less than half of CatBoost's 0.851. The ordering is informative:
  a single-mode robust Gaussian (EllipticEnvelope, 0.060) does *worse* than an
  8-component mixture, so the genuine-transaction manifold is **multimodal**, not one
  ellipse; LOF is ≈ random in this 30-d space. The PCA features carry label-aligned
  signal that only the supervised GBMs fully exploit, so unsupervised methods are
  kept for contrast, not as contenders.

## Operating points (CatBoost, threshold chosen on OOF)
- `max-f1` → thr 0.535: **P 0.96 / R 0.79 / F1 0.865**, catches 77/98 frauds, 3 FP.
- `precision-floor 0.9` → thr 0.195: **P 0.88 / R 0.85 / F1 0.865**, catches 83/98
  frauds, 11 FP. Pick by the false-positive budget of the review team.

## Artifact naming
- `experiments/artifacts/holdout_idx.npz` — fixed stratified train/test row indices.
- `experiments/artifacts/{model}_oof.npy` — OOF positive-class probabilities.
- `experiments/artifacts/{model}_test.npy` — bagged hold-out probabilities.
- `experiments/artifacts/y_oof.npy` / `y_test.npy` — aligned 0/1 labels.
- `experiments/artifacts/model_results.{json,csv}` — metrics per model.
- `experiments/artifacts/blend_{oof,test}.npy`, `blend_weights.json` — blend members + scores.
- `outputs/{model}_eval.json`, `reports/figures/{model}_pr_curve.png` — final report.

## Run history
- 2026-06-23 — `src.train --model all` (base) + `--imbalance classweight`; `src.ensemble`
  (5-member and GBM-only). Best: CatBoost OOF AUPRC 0.85128 / hold-out 0.87734. Blend
  and class weighting gave no OOF lift.
- 2026-06-24 — Added four detectors to `src.anomaly` (`ocsvm`, `elliptic`, `gmm`,
  `pca`) alongside `iforest`/`lof`. GaussianMixture best unsupervised at hold-out
  AUPRC 0.359; full sweep in the table above.

## Next ideas (untested)
- `undersample` / `smote` sweeps (training-fold only) — expected similar AUPRC story.
- Light hyper-parameter tuning of CatBoost/LightGBM on OOF AUPRC.
- An **autoencoder** reconstruction-error baseline (adds a torch dep) — the next
  anomaly-detector to try; GMM density is the bar to beat (0.359).
- Seed-bagging CatBoost for a small variance reduction.
