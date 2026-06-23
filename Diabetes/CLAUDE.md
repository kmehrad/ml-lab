# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project — Diabetes (Kaggle Playground Series S5E12)
Binary classification predicting the probability of a diabetes diagnosis
(`diagnosed_diabetes`) from clinical, lifestyle, and demographic features (700k
train rows, ~62% positive). **Metric: ROC-AUC** — submissions are positive-class
probabilities, not hard labels. Schema: 15 numeric, 3 binary, 3 ordinal
(`education_level`, `income_level`, `smoking_status`), 3 nominal (`gender`,
`ethnicity`, `employment_status`); pinned in `src/data.py`.

## Commands
Python 3.11; install deps with `pip install -r requirements.txt`. **All commands
run from this project directory (`Diabetes/`), not the git repo root** — modules
import the top-level `src` package, and pytest fails to collect from the repo
root.

```bash
python3 -m pytest                              # all tests
python3 -m pytest tests/test_features.py -q     # one file
python3 -m pytest tests/test_features.py::test_composites_match_semantics  # one test
python3 -m src.train --model lgbm --sample 50000   # smoke run (subsample)
python3 -m src.train --model all --folds 5         # base pool: OOF + bagged test preds
python3 -m src.train --model all --folds 5 --use-original   # augmented pool (_aug suffix)
python3 -m src.blend                               # equal-weight rank-average blend
python3 -m src.submit --model blend                # write outputs/blend_submission.csv
python3 -m src.submit --model blend --submit -m "…"  # upload via Kaggle CLI
```

Data download (Kaggle CLI must be configured): `kaggle competitions download -c
playground-series-s5e12 -p data/raw && unzip -o data/raw/*.zip -d data/raw`.
Optional `data/raw/original.csv` is the source dataset
(`alamshihab075/health-and-lifestyle-data-for-diabetes-prediction`, `Age`→`age`).

## Architecture (prediction-file pipeline)
The pipeline communicates through **saved prediction arrays**, not in-memory
objects, so training and submission are decoupled and any subset of models can be
blended after the fact.

- `src/data.py` — schema constants + `load_raw` (raises on schema mismatch),
  `split_features_target`, `load_original` (augmentation source).
- `src/features.py` — `DiabetesFeatureEngineer`: row-wise, **stateless** clinical
  composites (lipid ratios, BP summaries, metabolic risk) + ordinal encoding;
  nominals left as strings for the encoder.
- `src/preprocessing.py` — `build_preprocessor("tree"|"linear")`: one-hot nominal
  categoricals (+ `StandardScaler` for `linear`). Fit **inside each fold**.
- `src/train.py` — `run_cv` does 5-fold `StratifiedKFold` (`seed=42`), scores OOF
  **probabilities** with `roc_auc_score`, and saves `{key}_oof.npy` +
  fold-averaged (bagged) `{key}_test.npy` to `experiments/artifacts/`. `--use-original`
  concatenates the source dataset into **training folds only** (key gets `_aug`
  suffix). Estimators built in `build_estimator` (lazy model imports).
- `src/blend.py` — equal-weight **rank-average** over selected members' saved OOF
  + test arrays; writes `blend_weights.json` + `blend_test.npy`.
- `src/submit.py` — builds a validated submission from saved bagged predictions
  (no refit), checks shape/ids/range against `sample_submission.csv`, optional upload.

Git-ignored: `data/`, `models/`, `outputs/`, `experiments/artifacts/`,
`reports/figures/`.

## Conventions & hard-won lessons
- AUC is rank-based → train **without** class re-weighting (`class_weight` /
  `scale_pos_weight` hurt calibration without helping AUC here).
- **Local CV does NOT predict this leaderboard** (~0.03 gap from concept shift).
  Do not chase OOF: OOF-optimized weighting (stacking, hill-climbing) *overfits*
  and loses on the LB; an **equal-weight diverse blend** generalised best. The
  **original-data concat** was the single biggest LB lever.
- Get explicit approval before each Kaggle submission; log every run in
  `experiments/README.md` (full history + findings live there).
