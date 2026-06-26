# Experiment log — DisasterTweets (nlp-getting-started)

Metric: **F1**. CV: **StratifiedKFold(5, shuffle=True, random_state=42)** on `target`.
Threshold tuned on OOF predictions to maximize F1; stored per model.
Only adopt a change if its OOF F1 beats the current best by more than the fold std.
Submit only after CV review + approval.

| ID | Date | Model | OOF F1 | Threshold | Public LB | Fold mean ± std | Notes |
|----|------|-------|-------:|----------:|----------:|-----------------|-------|
| exp-001 | 2026-06-25 | **tfidf_lr** | **0.77446** | 0.461 | **0.80324** | 0.77074 ± 0.01484 | Baseline. TF-IDF word(1-2)+char(3-5) → LogisticRegression on `keyword+text`. Best classic. **Submitted.** |
| exp-002 | 2026-06-25 | tfidf_nb | 0.77181 | 0.466 | — | 0.77126 ± 0.01143 | ComplementNB(α=0.3); ≈ lr, within fold std. |
| exp-003 | 2026-06-25 | tfidf_svc | 0.76025 | 0.477 | — | 0.75761 ± 0.01082 | LinearSVC(C=0.5), sigmoid scores. Weakest. |
| exp-004 | 2026-06-25 | tfidf_lr +fix-dups | 0.77396 | 0.477 | — | 0.76861 ± 0.02230 | Dup label majority-vote: ~flat (within noise) → **rejected**. |

| exp-005 | 2026-06-25 | **roberta-base** | **0.80445** | 0.547 | **0.83879** | 0.80229 ± 0.00750 | 5-fold fine-tune, 3 epochs, bs32, max_len128, RTX 3090 Ti (212s). **+0.030 OOF / +0.036 LB over baseline. Submitted.** |
| exp-006 | 2026-06-25 | blend (lr:rob 1:4) | 0.80521 | 0.574 | — | — | Rank-avg. +0.00076 over roberta → **within noise, rejected**; submit roberta alone. |

**exp-001 adopted as classic baseline** (public LB 0.80324). Char n-grams matter (56% of test
tokens are OOV — see EDA). **exp-005 (RoBERTa) is the best model** — a genuine +0.030 OOF gain
over the baseline (≈4× fold std). The TF-IDF/RoBERTa blend (exp-006) is within noise, so RoBERTa
alone is preferred for submission.

## Groq LLM zero-shot (no training) — estimated on a fixed 400-tweet stratified val slice

Metric is F1 on the labeled val subset (`StratifiedShuffleSplit`, seed 42). These are
**zero-shot** (just a prompt, no fine-tuning), for comparison against the trained models.

| ID | Date | Model (Groq) | Shots | F1 | Precision | Recall | Acc | Notes |
|----|------|--------------|------:|---:|----------:|-------:|----:|-------|
| llm-001 | 2026-06-25 | qwen/qwen3-32b | 0 | 0.703 | 0.864 | 0.593 | 0.785 | Best LLM zero-shot; recall-bound. |
| llm-002 | 2026-06-25 | meta-llama/llama-4-scout-17b | 0 | 0.605 | 0.907 | 0.453 | 0.745 | High precision, low recall (over-conservative). |
| llm-003 | 2026-06-25 | meta-llama/llama-4-scout-17b | 8 | 0.708 | 0.866 | 0.599 | 0.788 | Few-shot lifts recall 0.453→0.599 (+0.10 F1). |
| llm-004 | 2026-06-25 | qwen/qwen3-32b | 8 | 0.728 | 0.856 | 0.633 | 0.796 | Partial (348/400 covered; hit Groq 1000/day cap). Few-shot lifts recall 0.593→0.633. |

### Hybrid: disagreement-gated LLM ensemble (`src/route.py`)

| ID | Date | Method | Val F1 | Public LB | Notes |
|----|------|--------|-------:|----------:|-------|
| route-001 | 2026-06-25 | RoBERTa+TFIDF agree, else qwen3 tiebreaker | 0.8064 (vs 0.8012) | **0.83879** | LLM only on 11% disagreement rows. **Submitted; LB = RoBERTa alone → F1-neutral.** 286/364 test ambig LLM-decided, 78 RoBERTa fallback (daily cap). |

**Conclusion:** the LLM tiebreaker is F1-neutral on the leaderboard (0.83879 = RoBERTa alone).
Its value is cost/efficiency (~9× fewer LLM calls), not accuracy. **RoBERTa remains the best,
simplest production model.**

For a directly comparable view (trained models scored on the same 400-tweet val slice) and the
full write-up, see [`reports/RESULTS.md`](../reports/RESULTS.md): RoBERTa **0.8012**, baseline
**0.7701** on that slice — both above the LLMs.

**Takeaway:** zero-shot LLMs (0.60–0.70) trail fine-tuned RoBERTa (0.804) by a wide margin,
limited by **recall** — they only flag unambiguous disasters. Few-shot examples are the obvious
lever to raise recall before spending calls on the 3,263-row test set. Groq free tier is
rate-limited (qwen3 ~6k TPM → ~30 req/min), so the full test set is slow (~tens of minutes).
