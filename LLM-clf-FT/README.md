# LLM Classification Finetuning — Chatbot Arena Preference Prediction

Kaggle competition: <https://www.kaggle.com/competitions/llm-classification-finetuning>

## Task
Given a user `prompt` and two competing LLM answers (`response_a`, `response_b`),
predict which answer a human preferred. Three mutually exclusive outcomes:

| target            | meaning              |
|-------------------|----------------------|
| `winner_model_a`  | A was preferred      |
| `winner_model_b`  | B was preferred      |
| `winner_tie`      | tie / both bad       |

This is the LMSYS **Chatbot Arena** human-preference data framed as a
3-class **probability** problem.

## Evaluation
Multi-class **log loss** over the three class probabilities (rows sum to 1).

## Data (`data/raw/`, not committed)
| file                    | rows    | notes |
|-------------------------|---------|-------|
| `train.csv`             | 57,477  | full schema incl. `model_a/model_b` and one-hot targets |
| `test.csv`              | ~3 vis. | `id, prompt, response_a, response_b` only — **hidden test, swapped at scoring** |
| `sample_submission.csv` | —       | `id, winner_model_a, winner_model_b, winner_tie` |

**This is a Code Competition**: you submit an inference notebook that runs
**offline** against the hidden test set. Plan accordingly (no internet at scoring;
models/weights must be attached as Kaggle datasets).

### Key facts from EDA
- Class balance ≈ **34.9 / 34.2 / 30.9** (A / B / tie) — nearly uniform.
- Prior-only log loss = **1.0972** vs uniform 1.0986 → priors carry ~no signal;
  **all signal is in the text.**
- `prompt` / `response_*` are **JSON-encoded lists** (multi-turn). 13% multi-turn,
  up to 36 turns. p95 ≈ 2000 tokens, p99 ≈ 3750.
- 64 distinct models, but **model names do not exist in test** → not usable as features.

## Results

OOF log loss on the shared stratified 5-fold split (seed 42) is the source of
truth. Lower is better; uniform floor = 1.0986.

| phase | model | OOF log loss | vs prior | notes |
|-------|-------|------------:|---------:|-------|
| — | uniform (1/3) | 1.0986 | — | theoretical floor |
| — | class prior | 1.0972 | — | priors carry ~no signal |
| 1 | **LightGBM baseline** | **1.01541** | **−0.082** | 5-fold OOF; folds 1.013–1.018 (stable) |
| 2 | DeBERTa-v3-base (fold 0, 2 ep) | 1.06397 | −0.033 | single fold; learns but < baseline |
| 2 | **DeBERTa-v3-large** (5-fold, 2 ep) | **0.99963** | **−0.098** | folds 0.996–1.003; first to beat baseline |
| 2 | **LightGBM + large blend** | **0.99221** | **−0.105** | 0.35·LGBM + 0.65·large (full OOF) |

### Submissions (public leaderboard)

| submission | public LB | OOF | notes |
|------------|----------:|----:|-------|
| LightGBM baseline (`kaggle/lgbm_baseline/`) | **1.01515** | 1.0161 | CV ≈ LB — OOF is a trustworthy proxy |

The first submission validated the full **code-competition** pipeline end-to-end:
a self-contained offline kernel that finds the mounted data, trains 5-fold
LightGBM, and emits `submission.csv`, submitted via
`kaggle competitions submit -k <kernel> -v <ver> -f submission.csv`. Public LB
(1.01515) matched local OOF (1.0161) almost exactly — no leak, CV is reliable.

**Phase 1 — LightGBM baseline** (`src/train_baseline.py`): engineered
length/format diffs + TF-IDF→SVD text features fed to LightGBM on the shared
folds. OOF **1.01541** — a real but weak floor, as expected when a tabular
approach is applied to a fundamentally NLP sequence-pair task. Establishes the
CV harness, `FeatureBuilder`, and submission format.

**Phase 2 — DeBERTa-v3 fine-tune** (`src/train_deberta.py`, GPU workstation):
sequence-pair input with textual role markers (`"Prompt:" / "Response A:" /
"Response B:"`), per-field head+tail truncation, and **A/B-swap TTA** at eval.
- **base** plateaus at ~1.06 (single fold) — it *learns* (gradients verified by
  overfitting 64 examples to ~0.15) but trails the baseline; diagnosed as a
  capacity limit, not a bug (labels/alignment/eval all check out, calibration
  has no headroom, and it adds ~nothing to a blend).
- **large** (max_len 1024, bs 2×8, lr 1e-5, 2 epochs) reaches **0.99963** OOF
  over the full 5 folds (per-fold 0.99845 / 0.99977 / 0.99581 / 1.00116 /
  1.00297 — tight) — the first model to beat the baseline. Blended with the
  LightGBM OOF (**0.35·LGBM + 0.65·large**) it drops to **0.99221**; the two
  agree only ~68% of the time, so they are genuinely complementary. Both gains
  are well above CV noise.

The blended OOF (0.99221) is not yet on the leaderboard: the CV run computed OOF
but did not checkpoint weights, so the offline inference kernel still needs a
saved large model (retrain-with-save → upload as a Kaggle dataset → GPU kernel
running large + A/B-swap TTA, blended in-kernel with LightGBM).

> **Infra note.** DeBERTa-v3 training requires a pinned stack — see
> `setup_env.sh`. On `transformers 5.10` the rewritten DeBERTa-v2 backward
> produces broken gradients (the model trains to a *random* 1.097 while RoBERTa
> on the same box trains fine); `transformers 4.46.3` + `torch 2.5.1` is the
> known-good combo. `run_fold.sh` launches a detached fold run on the GPU box.

## Layout
```
data/raw/         # downloaded CSVs (gitignored)
data/processed/   # folds, tokenized caches
data/submissions/ # generated submissions
notebooks/        # 01_eda.ipynb, ...
src/              # reusable pipeline code
reports/figures/  # saved plots
PLAN.md           # model-building plan & roadmap
```

## Setup
```bash
pip install -r requirements.txt
kaggle competitions download -c llm-classification-finetuning -p data/raw
unzip -o data/raw/*.zip -d data/raw
```

See **[PLAN.md](PLAN.md)** for the modeling roadmap and model recommendations.
