# Modeling Plan — Chatbot Arena Preference Prediction

Goal: minimize multi-class **log loss** predicting `[winner_model_a, winner_model_b, winner_tie]`.

## 1. What kind of problem is this, really?

The priors give log loss 1.0972 (≈ the 1.0986 floor). **There is essentially no
signal outside the text.** So this is fundamentally an **NLP sequence-pair
classification** task: read the prompt + both answers and judge which is better.
It is *not* a tabular problem — engineered scalar features (length, etc.) only
help at the margins. The winning approaches on this exact dataset (LMSYS) were
**fine-tuned LLMs**, not GBMs.

Hard parts:
- **Ties are 31%** and genuinely ambiguous → calibration matters a lot for log loss.
- **No model identity at test time** → can't memorize "gpt-4 usually wins".
- **Long, multi-turn** inputs (p99 ≈ 3750 tokens) → truncation strategy matters.
- Position bias: humans/judges favor whichever answer is shown first → the A/B
  ordering is itself signal *and* a source of bias to neutralize.

## 2. Which models are most suitable

Ranked by expected score on this task (best first):

| Tier | Model | Why | Compute |
|------|-------|-----|---------|
| **A — top** | **Gemma-2 9B** (QLoRA, seq-classification head) | The strongest single models on this exact dataset. Strong base reasoning transfers to judging answer quality. | 1×A100/2×T4, 4-bit, ~hours |
| **A — top** | **Llama-3 / 3.1 8B** (QLoRA) | Same recipe; slightly behind Gemma-2 9B but very strong; great ensemble partner. | 2×T4 (Kaggle) feasible |
| **B — strong/cheap** | **DeBERTa-v3-large** (full or LoRA fine-tune) | Best non-LLM encoder; far cheaper, no 4-bit tricks. A solid ~0.95–1.00 log-loss baseline. Good first real model. | 1×T4/P100, or slow on Mac MPS |
| **C — baseline only** | TF-IDF / length features + **LightGBM** | Pipeline sanity check & fast CV harness. Will land ~1.03–1.05 — barely beats priors. Not competitive; keep as a floor. | Local CPU/Mac |

**Recommendation:** validate the pipeline with Tier C locally, get a real
encoder baseline with **DeBERTa-v3** (Tier B), then invest in **Gemma-2 9B QLoRA**
(Tier A) as the contender. Final = ensemble of Gemma-2 9B + Llama-3 8B (+ DeBERTa).

### Input formatting (applies to all transformer models)
Concatenate with explicit role markers, e.g.:
```
<PROMPT> {prompt}
<RESPONSE A> {response_a}
<RESPONSE B> {response_b}
```
- Join multi-turn lists with turn separators; truncate **per-field** (keep head+tail
  of each response) so one long answer can't crowd out the other.
- `max_len` 1536–2048 covers ~p95; going to 3072 helps marginally at higher cost.
- Head: 3-way softmax classification head over the pooled/last-token representation.

## 3. Validation
- **Stratified K-Fold (5)** on the 3-class label, fixed `seed`, saved to
  `data/processed/folds.parquet` so every model uses identical folds.
- Track **out-of-fold log loss** as the single source of truth; trust it over LB
  (test is hidden and small-batch scored).
- Watch calibration: temperature-scale / blend toward priors if OOF is overconfident.
- Memory note: only ship a change if OOF improvement is above CV noise
  ([[submit-only-if-genuinely-improved]]).

## 4. Roadmap (phased)

- **Phase 0 — Setup & EDA** *(done / in `notebooks/01_eda.ipynb`)*
  Confirm schema, class balance, lengths, multi-turn, baselines, build folds.
- **Phase 1 — Baselines (local, CPU/MPS)** *(done — `src/train_baseline.py`)*
  (a) prior/uniform log loss (1.0972); (b) engineered (length/format diffs) +
  TF-IDF→SVD features → LightGBM, shared 5-fold. **OOF log loss 1.01541**
  (folds 1.013–1.018), ≈0.082 above priors — a real but weak floor, as expected.
  Establishes the CV harness, `FeatureBuilder`, and submission format.
- **Phase 2 — DeBERTa-v3 fine-tune (Tier B)** *(in progress — `src/train_deberta.py`)*
  First genuine model. Target OOF log loss ≈ 0.95–1.00. Establishes the input
  format, truncation, and inference notebook end-to-end. Runs on the GPU
  workstation under a pinned stack (`setup_env.sh`: torch 2.5.1 + transformers
  4.46.3 — `transformers 5.10` has broken DeBERTa-v2 gradients). First fold-0
  run = **1.06397** (learns but trails Phase 1, undertrained); tuning with role
  markers + 4 epochs before committing to full 5-fold CV.
- **Phase 3 — Gemma-2 9B / Llama-3 8B QLoRA (Tier A)** *(the contender)*
  4-bit QLoRA on Kaggle/Colab GPU. Target sub-0.90 OOF. This is where the score is won.
- **Phase 4 — Robustness & ensemble**
  - **A/B swap TTA**: run each test row twice (A,B) and (B,A), average with A↔B
    probs swapped → removes position bias, reliable boost.
  - Ensemble Gemma + Llama (+ DeBERTa) by averaging probabilities; optional
    light stacking.
  - Optional: pseudo-label extra public LMSYS arena data to expand training.
- **Phase 5 — Submission packaging (Code Competition)**
  Offline inference notebook; attach fine-tuned weights as a Kaggle dataset;
  no-internet, fits time/memory limits. A/B-swap TTA inside it.

## 5. Compute reality — decide before Phase 3
Tier-A LLM fine-tuning is **not feasible on the local Mac** (no CUDA; 4-bit
`bitsandbytes` and 9B training need a real GPU). Realistic options:
- **Kaggle Notebooks** — free 2×T4 (16 GB×2) or P100, ~30 GPU-hrs/week. Native fit
  for a Code Competition. **Default recommendation.**
- **Colab** (Pro/A100) or any cloud A100 — faster, paid.
- **Local Mac (MPS)** — fine for EDA, LightGBM, and *slow* DeBERTa-base only.

Phases 0–2 run locally now; Phases 3–5 target Kaggle GPU. Confirm the GPU
environment before starting Phase 3.

## 6. Open questions
- Which GPU env for Tier A (Kaggle vs Colab vs other)?
- Is bringing in external public LMSYS arena data allowed under the rules
  (pseudo-labeling)? Verify before Phase 4.
