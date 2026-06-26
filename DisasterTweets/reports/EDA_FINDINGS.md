# EDA findings — DisasterTweets (nlp-getting-started)

Source: `notebooks/01_eda.ipynb` (executed). Figures in `reports/figures/`.

## Headline

- **Binary classification, mildly imbalanced:** 7,613 train rows, **42.97% positive** (disaster).
  Test is 3,263 rows. Imbalance is mild — no resampling needed; metric is **F1**.
- **`text` is the signal.** It is never missing in train or test. `keyword` is missing for
  61 train / 26 test rows; `location` is missing ~33% (2,533 train / 1,105 test) and is free-text
  noise (e.g. "Worldwide", "ð\x9f\x98...") — low value, used only as a weak side feature if at all.

## Class balance & text stats
![class balance](figures/class_balance.png)
![text length by class](figures/text_length_by_class.png)

Disaster tweets are slightly **longer** (108 vs 96 chars; 15.2 vs 14.7 words) and carry **more URLs**
(0.77 vs 0.51) and hashtags (0.50 vs 0.39); non-disaster tweets have more **@mentions** (0.42 vs 0.27)
and exclamation marks. These are weak signals individually but useful for the classic model.

## Keyword is highly predictive
![keyword disaster rate](figures/keyword_disaster_rate.png)

With ≥25 occurrences, `keyword` disaster rate spans the full range:
- **~1.0 disaster:** `derailment`, `wreckage`, `debris`, `outbreak`, `oil%20spill`, `typhoon`,
  `suicide%20bombing`.
- **~0.0 disaster:** `aftershock` (0.00!), `body%20bags`, `ruin`, `blazing`, `electrocute`,
  `screaming`, `panicking` — figurative/metaphorical usage dominates.

Keywords are URL-encoded (`%20` for space). Prepending the (decoded) keyword to the text is a cheap,
strong feature → the classic model uses a combined `keyword + text` field.

## Data-quality issues (important)

- **Mislabeled duplicates:** 69 groups of exact-duplicate texts; **18 of them have conflicting
  labels** (55 rows). This caps achievable F1 and adds label noise. Option (behind a CV-validated
  flag): majority-vote relabel duplicates in training.
- **Train/test text overlap:** **68 tweets appear verbatim in both** train and test — a small,
  legitimate signal (their train labels can inform those test rows).

## Generalization

- Test vocabulary is large and **56% of test tokens are unseen in train** (typical for short, noisy
  tweets). This favors:
  - **char n-grams** in TF-IDF (robust to OOV/typos/hashtags), and
  - a **pretrained transformer** (subword tokenization handles OOV) — expected to clearly beat the
    classic baseline.

## Modeling implications

1. Use `keyword + text` as the input field; light cleaning (strip URLs/mentions, normalize), keep
   hashtag words.
2. Classic baseline: TF-IDF **word (1–2) + char (3–5) n-grams** → linear model; tune threshold for F1.
3. Transformer (RoBERTa-base) on near-raw text — primary model.
4. `StratifiedKFold(5)` on `target`; share folds across models for blending.
5. Consider duplicate label-fixing and the 68 train/test verbatim matches as a post-hoc override.
