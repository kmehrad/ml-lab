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
|-------|-------|-------------:|---------:|-------|
| — | uniform (1/3) | 1.0986 | — | theoretical floor |
| — | class prior | 1.0972 | — | priors carry ~no signal |
| 1 | **LightGBM baseline** | **1.01541** | **−0.082** | folds 1.013–1.018 (stable) |

**Phase 1 — LightGBM baseline** (`src/train_baseline.py`): engineered
length/format diffs + TF-IDF→SVD text features fed to LightGBM on the shared
folds. OOF **1.01541** — a real but weak floor, as expected when a tabular
approach is applied to a fundamentally NLP sequence-pair task. Its value is
establishing the CV harness, `FeatureBuilder`, and submission format; the score
will be won by the transformer/LLM models in Phases 2–3 (see [PLAN.md](PLAN.md)).

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
