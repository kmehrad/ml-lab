# Results — DisasterTweets (Kaggle `nlp-getting-started`)

Binary disaster-tweet classification. Metric: **F1**. Two model families were evaluated:
**trained** models (TF-IDF baseline, fine-tuned RoBERTa) and **zero/few-shot LLMs** via the
Groq API (no training). Last updated 2026-06-25.

> ⚠️ Qwen 3 few-shot is still running; its cell is marked _pending_ and will be filled in.

## 1. Headline

| Model | Type | Val F1 (400-slice) | 5-fold OOF F1 | Public LB |
|---|---|---:|---:|---:|
| **RoBERTa-base (fine-tuned)** | trained | **0.8012** | **0.80445** | **0.83879** |
| TF-IDF + LogisticRegression | trained | 0.7701 | 0.77446 | 0.80324 |
| Qwen 3-32B, few-shot(8) | LLM, no training | _pending_ | — | — |
| Llama 4 Scout, few-shot(8) | LLM, no training | 0.7079 | — | — |
| Qwen 3-32B, zero-shot | LLM, no training | 0.7034 | — | — |
| Llama 4 Scout, zero-shot | LLM, no training | 0.6047 | — | — |

**Fine-tuned RoBERTa is the best model** (public LB 0.83879). The best LLM so far (Qwen 3
few-shot) is competitive with the *classic* baseline but does **not** match RoBERTa.

## 2. Apples-to-apples — same 400-tweet validation slice

All rows below are scored on the identical stratified val slice
(`StratifiedShuffleSplit`, seed 42, n=400, 43% positive), so trained and LLM models are
directly comparable. Trained-model rows apply each model's OOF-tuned threshold.

| Model | F1 | Precision | Recall | Accuracy |
|---|---:|---:|---:|---:|
| RoBERTa-base (fine-tuned) | **0.8012** | — | — | — |
| TF-IDF + LogReg (baseline) | 0.7701 | — | — | — |
| Qwen 3-32B — few-shot(8) | _pending_ | _pending_ | _pending_ | _pending_ |
| Llama 4 Scout — few-shot(8) | 0.7079 | 0.866 | 0.599 | 0.788 |
| Qwen 3-32B — zero-shot | 0.7034 | 0.864 | 0.593 | 0.785 |
| Llama 4 Scout — zero-shot | 0.6047 | 0.907 | 0.453 | 0.745 |

### Key finding: LLMs are recall-bound
Zero-shot LLMs have **high precision but low recall** — they only flag unambiguous disasters
and miss figurative/indirect ones. Adding **8 balanced few-shot examples** raises recall
sharply (Llama 4: 0.453 → 0.599, F1 0.605 → 0.708) with little precision loss. Even so,
few-shot LLMs land near the *classic* baseline, below fine-tuned RoBERTa.

## 3. Trained models — full detail (5-fold StratifiedKFold OOF)

| Model | OOF F1 | Threshold | Fold mean ± std | Public LB | Notes |
|---|---:|---:|---:|---:|---|
| **roberta-base** | **0.80445** | 0.547 | 0.80229 ± 0.00750 | **0.83879** | 3 epochs, bs32, max_len128, RTX 3090 Ti, 212 s |
| tfidf_lr (baseline) | 0.77446 | 0.461 | 0.77074 ± 0.01484 | 0.80324 | word(1-2)+char(3-5) TF-IDF → LogReg |
| tfidf_nb | 0.77181 | 0.466 | 0.77126 ± 0.01143 | — | ComplementNB(α=0.3) |
| tfidf_svc | 0.76025 | 0.477 | 0.75761 ± 0.01082 | — | LinearSVC(C=0.5) |
| blend (lr:roberta 1:4) | 0.80521 | 0.574 | — | — | rank-avg; +0.0008 vs RoBERTa → within noise, rejected |

- **RoBERTa beats the baseline by +0.030 OOF / +0.036 LB** (~4× fold std — a genuine gain).
- The TF-IDF/RoBERTa blend was within fold-std noise of RoBERTa, so the simpler single model
  was submitted.
- OOF↔LB tracked closely (gap ≈ +0.03 for both submitted models), so the CV is trustworthy.

## 4. LLM setup (Groq, no training)

- **Models:** `meta-llama/llama-4-scout-17b-16e-instruct`, `qwen/qwen3-32b`.
- **Prompt:** system message defines "real disaster vs. metaphorical/unrelated"; the model
  replies with a single `1`/`0`. `temperature=0`. Qwen runs with `/no_think`.
- **Few-shot:** 8 balanced examples (4 pos / 4 neg) sampled from train, excluded from the val
  slice to avoid leakage.
- **Harness:** `src/llm_eval.py` — concurrent calls, retry honoring Groq's `retry-after`,
  per-id response cache keyed by `(model, split, shots)`; failures excluded from scoring.
- **Rate limits (free tier):** Qwen 3 ~6k tokens/min (~30 req/min) — few-shot is token-heavy
  and slow; budget tens of minutes for the full 3,263-row test set.

## 5. Reproduce

```bash
# trained models
uv run python -m src.train --model tfidf            # baseline (OOF F1 0.77446)
uv run python -m src.transformer --model roberta    # RoBERTa (OOF F1 0.80445; GPU)
uv run python -m src.blend                           # blend (within noise)

# LLMs (Groq; needs GROQ_API_KEY in .env)
uv run python -m src.llm_eval --model llama4 --n 400            # zero-shot val
uv run python -m src.llm_eval --model qwen3  --n 400 --shots 8  # few-shot val
uv run python -m src.llm_eval --model qwen3  --split test --shots 8   # full test set
```

See `experiments/README.md` for the chronological run log.
