# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project — Credit-Card Fraud (Kaggle ULB `mlg-ulb/creditcardfraud`)
Binary classification detecting fraudulent transactions under **extreme imbalance**
— 284,807 rows, **492 frauds (0.173%)**. Features are anonymised PCA components
`V1`–`V28` plus `Time` and `Amount`; target `Class` (1 = fraud). Schema pinned in
`src/data.py`. **Metric: AUPRC / average precision** (ROC-AUC is optimistic under
this imbalance and is only secondary). **There is no Kaggle leaderboard** — a fixed
stratified 20% hold-out + 5-fold CV on the remainder is the whole evaluation.

## Commands
Python 3.11; `pip install -r requirements.txt` (includes `imbalanced-learn`). **All
commands run from this project directory (`CreditCardFraud/`)** — modules import the
top-level `src` package.

```bash
python3 -m pytest                                   # all tests
python3 -m src.train --model lgbm --sample 30000    # smoke run (keeps all positives)
python3 -m src.train --model all                    # base pool: OOF + bagged hold-out preds
python3 -m src.train --model all --imbalance classweight   # {none,classweight,undersample,smote}
python3 -m src.anomaly --model all                  # Isolation Forest + LOF
python3 -m src.ensemble --members lgbm xgb catboost histgb # rank-average blend
python3 -m src.evaluate --model catboost --policy max-f1
python3 -m src.evaluate --model catboost --policy precision-floor --min-precision 0.9
```

Data: `kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw && unzip -o data/raw/*.zip -d data/raw`.

## Architecture (prediction-file pipeline)
Training and evaluation are decoupled through **saved prediction arrays** in
`experiments/artifacts/`, so any subset of models can be blended/re-evaluated
without refitting.

- `src/data.py` — schema constants + `load_raw` (raises on schema mismatch),
  `split_features_target`, `make_holdout_split` (fixed stratified test indices
  cached to `holdout_idx.npz`).
- `src/features.py` — `FraudFeatureEngineer`: stateless `log1p(Amount)`,
  `amount_zero`, `hour`; `V1`–`V28` untouched. Sets a fitted marker so
  `Pipeline.transform` accepts it.
- `src/preprocessing.py` — `build_preprocessor("tree"|"linear")`; `linear` adds
  `RobustScaler` on `Time`/`Amount`/engineered cols (Vs already ~unit-scale). Fit
  **inside each fold**.
- `src/train.py` — `run_cv` does 5-fold `StratifiedKFold` on the train remainder,
  scores OOF with `average_precision_score`, saves `{key}_oof.npy` + bagged
  `{key}_test.npy` (hold-out). `--imbalance` reweights (`classweight`) or resamples
  (`undersample`/`smote`) **training folds only**; the strategy is encoded in the
  key (e.g. `lgbm_cw`). `y_oof.npy`/`y_test.npy` saved from the base `none` run.
- `src/anomaly.py` — Isolation Forest / LOF (`novelty=True`) fit on the remainder,
  scored on the hold-out; saved as `{name}_test.npy`. Reuses `train._record_results`.
- `src/ensemble.py` — equal-weight **rank-average** over members' OOF + test arrays.
- `src/evaluate.py` — final hold-out report: AUPRC/ROC-AUC + operating threshold
  picked on **OOF** (never the test set) via `max-f1` or `precision-floor`; writes
  `outputs/{model}_eval.json` + a PR curve to `reports/figures/`.

Git-ignored: `data/`, `models/`, `outputs/`, `experiments/artifacts/`, `reports/figures/`.

## Conventions & hard-won lessons
- **AUPRC is the metric, not accuracy/ROC-AUC.** Random AP ≈ 0.0017; ROC-AUC stays
  >0.95 even for weak models, so judge changes by AUPRC + OOF fold std.
- Supervised GBMs dominate; **CatBoost is the best single model** and the
  rank-average blend does **not** beat it (GBMs rank-correlate highly). Logreg drags
  blends down — exclude it.
- Unsupervised anomaly detection (IsolationForest/LOF) is a **weak baseline** here
  (AUPRC 0.16 / ≈random) — kept only for contrast, not as a contender.
- Resampling/weighting must touch **training folds only**; report it on AUPRC.
- Threshold is chosen on **OOF**, then applied to the hold-out, to avoid optimistic
  selection. Per user prefs: show CV/OOF metrics and get approval before declaring a
  result final; commit incrementally; adopt a change only on above-noise gains.
