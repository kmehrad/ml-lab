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

**exp-001 adopted as baseline.** Char n-grams matter (56% of test tokens are OOV — see EDA).
Transformer (RoBERTa) expected to clearly beat this; blend planned afterward.
