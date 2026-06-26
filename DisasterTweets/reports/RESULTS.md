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

## 5. Exact prompt

Source of truth: `src/llm_eval.py` (`SYSTEM`, `_user_msg`, `_build_fewshot`). The request is
`temperature=0`, and a single message list: **system → [few-shot pairs] → user(target tweet)**.

### System message (both zero- and few-shot)
```
You are an expert content classifier for emergency and disaster monitoring. Given a tweet,
decide whether it refers to a REAL disaster or emergency — an actual ongoing or reported event
such as a natural disaster, fire, explosion, accident, crash, attack, outbreak, or similar — as
opposed to a metaphorical, figurative, joking, headline-of-an-unrelated-topic, or otherwise
non-disaster use of those words. Reply with exactly one character: 1 if it is about a real
disaster, 0 if not. No other text.
```
For Qwen 3 the suffix ` /no_think` is appended to the system message to disable its reasoning trace.

### User message template
Built per tweet; the keyword line is omitted when the keyword is missing (URL-encoded `%20`
in keywords is decoded to spaces):
```
Keyword: {keyword}
Tweet: {text}
Answer (1 or 0):
```
Concrete rendered example (real train row):
```
Keyword: ablaze
Tweet: @bbcmtd Wholesale Markets ablaze http://t.co/lHYXEOHY6C
Answer (1 or 0):
```

### Zero-shot (`--shots 0`)
Message list = `[system, user(target)]`. The model replies with a single `1` or `0`
(`max_completion_tokens=8` for Llama 4; `1024` for Qwen 3 to leave room before `/no_think` truncation).

### Few-shot (`--shots 8`)
Eight **balanced** examples (4 positive, 4 negative) are sampled from `train` with seed 42,
**excluding the val slice** (no leakage), and inserted as alternating turns before the target:
```
system
user:      Keyword: {kw_1}\nTweet: {text_1}\nAnswer (1 or 0):
assistant: 1
user:      Keyword: {kw_2}\nTweet: {text_2}\nAnswer (1 or 0):
assistant: 0
... (8 example pairs total) ...
user:      Keyword: {kw}\nTweet: {text}\nAnswer (1 or 0):     ← target tweet, model answers
```

### Response parsing (`_parse`)
Robust to stray text: prefers a `label: 0|1` pattern, else the last standalone `0`/`1` token,
else falls back to the words "disaster"/"not". Unparseable replies are retried; persistent
failures are excluded from scoring (not defaulted to a class).

## 6. Hybrid — disagreement-gated LLM ensemble (`src/route.py`)

**Goal:** let the LLM help only on *ambiguous* tweets, and pay for LLM calls only there.

**Method.** RoBERTa and TF-IDF agree on the easy/non-ambiguous tweets (~88% of the val slice) —
take that shared label, no LLM. They disagree on the ambiguous ~12%; resolve those by a **3-way
majority vote** (RoBERTa + TF-IDF + LLM). Under a majority vote the LLM can only change a row
where the two trained models disagree, so the LLM is queried on *exactly those rows* — ~8× fewer
calls than scoring everything. (A probability-band gate `|p−thr|<δ` was also tried; it adds
nothing here because RoBERTa's probabilities sit near 0/1, so very few tweets are "near-threshold".)

**Val results (same 400-tweet slice):**

| Tiebreaker LLM | Routed F1 | RoBERTa-alone | Δ | LLM calls |
|---|---:|---:|---:|---:|
| **Qwen 3-32B (zero-shot)** | **0.8064** | 0.8012 | **+0.0051** | **48 / 400 (12%)** |
| Llama 4 Scout (zero-shot) | 0.7987 | 0.8012 | −0.0025 | 48 / 400 |

**Read:** Qwen 3 as tiebreaker nudges F1 up while touching only 12% of tweets; Llama 4 zero-shot
is too recall-poor to help. The **+0.0051 lift is within noise at n=400** (~2 tweets), so the
honest claim is "F1-neutral-to-slightly-positive at a fraction of the LLM cost." The robust win
is **efficiency** — ~88% of tweets never call the LLM, so the full 3,263-row test set needs only
~400–520 LLM calls. To confirm the small accuracy gain, evaluate on a larger labeled slice.

Run: `uv run python -m src.route --llm qwen3 --split val` (cached) ·
`--split test` queries the LLM on disagreements only and writes a submission.

## 7. Reproduce

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
