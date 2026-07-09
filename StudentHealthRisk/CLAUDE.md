# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project — StudentHealthRisk (Kaggle PS S6E7)
[`playground-series-s6e7`](https://www.kaggle.com/competitions/playground-series-s6e7):
**multi-class classification** predicting a student's `health_condition` (3 classes) from behavioural /
lifestyle features. Train 690,088 rows · test 295,753 rows.

**Schema (confirmed from raw CSVs):** target `health_condition` ∈ {`at-risk`, `fit`, `unhealthy`},
**heavily imbalanced** (at-risk ~86%, unhealthy ~8%, fit ~6%). 7 numeric behavioural features
(`sleep_duration, heart_rate, bmi, calorie_expenditure, step_count, exercise_duration, water_intake`)
and 6 low-cardinality (3-level) categoricals (`diet_type, stress_level, sleep_quality,
physical_activity_level, smoking_alcohol, gender`). **Missing values in nearly every column** (~449k
cells) — kept as-is; GBDTs split on NaN natively and categoricals are `category` dtype so NaN is its own
level. Submission is `id,health_condition` with a single predicted class **label** per row.

**Metric: Balanced Accuracy Score** (mean of per-class recall) — confirmed via the Kaggle API
`evaluationMetric` field (web pages are JS-rendered and unfetchable). This is **decision-sensitive**:
plain argmax chases the 86% majority and scores poorly (naive "always at-risk" = 0.333). We predict
class probabilities, then tune per-class **decision weights** (`metric.tune_weights`) on OOF to maximise
balanced accuracy and apply the same weights to test. **Always tune/compare on the tuned score, never
raw argmax.** `src/metric.py` is the single source of truth for scoring, weight tuning, and building the
label submission.

## Environment & commands
Uses **uv** (not pip/`requirements.txt`). Deps pinned in `pyproject.toml` / `uv.lock`. **Run everything
from this project dir (`StudentHealthRisk/`)** and prefix Python with `uv run`:

```bash
uv sync                                          # create/refresh the venv
uv run pytest                                    # metric + schema/format tests
uv run python -m src.train --model lgbm          # 5-fold CV -> OOF + bagged test arrays
uv run python -m src.blend --models lgbm xgb cat # equal-weight probability average
uv run python -m src.submit --model blend        # build + validate submission (add --submit to upload)
```

**Kaggle auth uses a `KGAT_` token at `~/.kaggle/access_token` (no `kaggle.json`).** The CLI reads it
from `KAGGLE_API_TOKEN`:

```bash
export KAGGLE_API_TOKEN="$(cat ~/.kaggle/access_token)"
uv run kaggle competitions download -c playground-series-s6e7 -p data/raw
unzip -o data/raw/playground-series-s6e7.zip -d data/raw
```

## Architecture (prediction-file pipeline — mirrors sibling `OptimalFertilizer/`)
`src/` communicates through **saved prediction arrays** so training and submission are decoupled and any
subset of models can be blended after the fact. Each base learner writes `{model}_oof.npy` (OOF
probability matrix `(n_train, 3)`) + fold-bagged `{model}_test.npy` `(n_test, 3)` + `{model}_metrics.json`
to `experiments/artifacts/`, alongside shared `y.npy`, `classes.npy`, `test_id.npy`.
- `src/data.py` — schema constants (`TARGET`, `CATEGORICAL`, `NUMERIC`, `ID`, `CLASSES`) + loaders (raise
  on schema mismatch; cache to `data/processed/`); `encode_target`/`decode_target`; `load_sample_submission`.
- `src/cv.py` — `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on the target.
- `src/metric.py` — `balanced_accuracy`, `tune_weights` (coordinate-ascent decision-weight search on
  OOF), `predict_codes`/`score_proba`, `to_submission_labels`.
- `src/features.py` — leakage-safe FE groups (`base` always on; optional `ratio`, `missflag`).
- `src/train.py` — GBDT CV loop (LightGBM/XGBoost/CatBoost, all with balanced class weights); reports
  raw + OOF-tuned balanced accuracy; saves OOF + bagged test arrays. `--model lgbm|xgb|cat`, `--device
  cpu|cuda`, `--seeds`, `--features`, `--tag`.
- `src/models_nn.py` — self-contained PyTorch MLP (one-hot cats + imputed numerics + missing flags,
  class-weighted CE, seed-averaged); same artifact format. Runs on the GPU box (`torch` = `gpu` extra).
- `src/blend.py` — equal-weight **probability average** of base learners, scored on OOF-tuned balanced
  accuracy.
- `src/hillclimb.py` — greedy forward-selection ensemble over the artifacts (tuned-metric objective).
- `src/submit.py` — tune weights on OOF, apply to test, build + hard-validate the label submission vs
  `sample_submission.csv`; optional Kaggle upload with `--submit`.

Git-ignored: `data/`, `models/`, `outputs/`, `experiments/artifacts/`, `reports/figures/`, `.venv/`.

## Remote GPU (RTX 3090 Ti box)
The Mac has no CUDA; the NN and any GPU GBDT work run on `kamron@192.168.0.92` via
`scripts/remote_run.sh {push|run|pull|all}` (rsync code+data up, run in the box's uv `.venv`, pull
artifacts back). Box dir default `StudentHealthRisk` (override with `SHR_BOX`/`SHR_REMOTE_DIR`).

## EDA
`notebooks/01_eda.ipynb` is **generated by** `notebooks/eda_build.py` (nbformat), then executed with
`uv run jupyter nbconvert --to notebook --execute --inplace`. Revise the generator, re-run — do not
hand-edit the notebook JSON. Figures → `reports/figures/`; conclusions → `reports/EDA_FINDINGS.md`.

## Conventions & hard-won lessons
- **Tune/compare on the real metric.** Balanced accuracy is decision-sensitive; always score OOF with
  tuned decision weights, not raw argmax or training log-loss.
- Prefer an **equal-weight diverse blend** over OOF-optimized weighting; treat sub-fold-std OOF
  differences as ties.
- **Show CV/OOF metrics and get explicit approval before any Kaggle submission.** Log every run in
  `experiments/README.md`. Commit incrementally after each meaningful step; stage only this project's
  files (the monorepo contains unrelated sibling projects).
- The living plan is at `~/.claude/plans/i-want-to-build-cozy-seahorse.md`.
