# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Kaggle solution for **llm-classification-finetuning** — predicting which of two
LLM answers a human preferred (Chatbot Arena / LMSYS data). It is a 3-class
**probability** problem scored by multi-class **log loss**. See `README.md` for the
data dictionary and `PLAN.md` for the modeling roadmap and model choices.

## Commands

```bash
pip install -r requirements.txt

# Data (gitignored; re-download as needed)
kaggle competitions download -c llm-classification-finetuning -p data/raw
unzip -o data/raw/*.zip -d data/raw

# Run a notebook end-to-end (CI-style check that it still executes)
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb \
  --ExecutePreprocessor.timeout=300

# Submit
kaggle competitions submit -c llm-classification-finetuning \
  -f data/submissions/<file>.csv -m "<message>"
```

There is no test suite or linter configured. The EDA notebook is the current source
of truth; validate changes by executing notebooks with `nbconvert`.

## Constraints that shape every decision

- **This is a Code Competition.** `test.csv` is a ~3-row stub; the real test set is
  hidden and swapped in when your **inference notebook runs offline at scoring**. No
  internet at inference — any model weights must be attached as a Kaggle dataset.
  Anything you build toward submission must run offline within Kaggle time/memory limits.
- **Model names are absent from `test.csv`** (`model_a`/`model_b` exist only in train).
  Never use them as model features — they won't exist at inference.
- **Signal is entirely in the text.** Prior-only log loss is 1.0972 vs the 1.0986
  uniform floor, so class priors are worthless. Treat this as an NLP sequence-pair
  task, not a tabular one; scalar features (lengths, etc.) are marginal at best.
- **`prompt` / `response_a` / `response_b` are JSON-encoded lists** (multi-turn,
  up to 36 turns). Always `json.loads` them before use; truncate per-field (head+tail)
  so one long answer can't crowd out the other. ~p95 ≈ 2000 tokens → `max_len` ≈ 2048.

## Conventions

- **Folds are shared.** `data/processed/folds.parquet` (stratified 5-fold, seed 42,
  built in `notebooks/01_eda.ipynb`) is the single fold assignment every model must
  reuse, so out-of-fold log losses are comparable. Don't re-split per model.
- **OOF log loss is the source of truth**, not the leaderboard (hidden, small-batch).
  Watch calibration — log loss punishes overconfidence, and ties (31%) are ambiguous.
- **A/B position bias is real.** Use A/B-swap TTA at inference (predict on (A,B) and
  (B,A), average with the A↔B probabilities swapped).
- Only ship a change when OOF improvement is above CV noise (see user memory
  `submit-only-if-genuinely-improved`).
- `data/` outputs and model artifacts are gitignored via the repo-root `.gitignore`;
  commit code, notebooks, `PLAN.md`/`README.md`, and `reports/figures` only.

## Compute

Local Mac (MPS, no CUDA) handles EDA, LightGBM, and slow DeBERTa-base only. The
contender LLM fine-tunes (Gemma-2 9B / Llama-3 8B QLoRA) need a real GPU — target
**Kaggle Notebooks (free 2×T4)** or Colab/cloud A100. Confirm the GPU environment
before starting Phase 3 in `PLAN.md`.
