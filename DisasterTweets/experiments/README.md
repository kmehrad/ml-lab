# Experiment log — DisasterTweets (nlp-getting-started)

Metric: **F1**. CV: **StratifiedKFold(5, shuffle=True, random_state=42)** on `target`.
Threshold tuned on OOF predictions to maximize F1; stored per model.
Only adopt a change if its OOF F1 beats the current best by more than the fold std.
Submit only after CV review + approval.

| ID | Date | Model | OOF F1 | Threshold | Public LB | Fold mean ± std | Notes |
|----|------|-------|-------:|----------:|----------:|-----------------|-------|
| — | — | — | — | — | — | — | _no runs yet_ |
