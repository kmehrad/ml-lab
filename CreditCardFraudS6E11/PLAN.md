# S6E11 solution plan

## 1. Data contract and EDA

1. Use the downloaded `train.csv` (699,960 × 26), `test.csv` (299,984 × 25), and
   `sample_submission.csv`. The contract is `id`, 24 predictors, and binary float target `is_fraud`.
2. Pin `id` and `is_fraud` in `src/data.py`. Fail fast on missing or extra columns, duplicate IDs,
   non-binary targets, train/test feature mismatch, or a sample submission that does not align with
   test IDs.
3. Build `notebooks/01_eda.ipynb` plus `reports/EDA_FINDINGS.md` covering the confirmed 1.66995%
   fraud prevalence,
   missingness, duplicates, cardinalities, numeric distributions, categorical target rates,
   train/test drift, suspicious identifiers, timestamps, and possible entity/group structure.
4. Because fraud is normally rare, report class prevalence and average precision as diagnostics,
   but optimize and select models by the official ROC-AUC. Never threshold predictions for Kaggle.
5. Use five-fold shuffled stratified CV: the schema contains hour/day fields but no full timestamp,
   transaction sequence, or repeated customer/card/merchant entity key that would support honest
   time/group splitting. Keep folds fixed across all models.

## 2. Baselines and feature preparation

1. Establish a constant/random reference, then a fold-safe logistic regression baseline with
   imputation, scaling for numeric columns, and one-hot encoding for low-cardinality categoricals.
2. Train LightGBM, XGBoost, and CatBoost baselines with native categorical handling where supported,
   early stopping, fixed folds, and seed 42. Save OOF probabilities, fold-averaged test
   probabilities, fold scores, runtime, and configuration in `experiments/artifacts/`.
3. Keep preprocessing inside each fold. Do not globally fit imputers, encoders, frequency/target
   encodings, resampling, or calibration. Do not use test labels or leaderboard feedback as a
   training signal.
4. Add only schema-supported features: `log1p` transforms for amount, balance, distance, and elapsed
   time; amount-to-balance ratio; cyclic hour/day encodings; velocity/count and risk-score
   interactions; sums/counts of boolean risk flags; and selected interactions involving foreign/IP,
   VPN, new-merchant, mismatch, CVV-retry, and AI-scam flags. There is no basis for entity history.
5. Treat SMOTE/undersampling/class weighting as controlled experiments, not defaults: ROC-AUC is
   rank-based and imbalance remedies can degrade probability ranking. Retain them only on repeatable
   OOF gains.

## 3. Evaluation and iteration

1. Use overall OOF ROC-AUC as the primary selection score, with fold mean/std and average precision
   as secondary diagnostics. Compare candidate predictions by fold and inspect OOF correlations.
2. Run ablations for feature groups and validation strategy before hyperparameter tuning. Reject
   changes smaller than fold/seed noise unless they improve consistently.
3. Tune the leading GBDTs conservatively: learning rate/iterations, depth or leaves, minimum child
   size, column/row sampling, and L1/L2 regularization. Use a small search budget and the same folds.
4. Blend probability or rank predictions only when OOF proves complementary. Start with equal
   weights; optionally optimize simple non-negative weights against OOF ROC-AUC without fitting the
   test set or public leaderboard.
5. Investigate any competition-provided original dataset only after a strong internal baseline.
   Validate schema mapping and source/license, mark source rows, and keep the external-data result as
   a separate experiment.

## 4. Submission and reproducibility

1. `src/submit.py` must align test IDs exactly to `sample_submission.csv`, verify two columns
   (`id,is_fraud`), full row count, unique IDs, finite probabilities in `[0,1]`, and no index column.
2. Submit a simple baseline first to validate the pipeline, then the best OOF-selected model/blend.
   Include model names, feature version, folds, and OOF ROC-AUC in each Kaggle message.
3. Record every experiment and leaderboard result in `experiments/README.md` and
   `reports/RESULTS.md`. Use the public leaderboard only as a sanity check; final selection remains
   based on validation because the private leaderboard determines the result.
4. Tests will cover schema validation, target encoding, fold disjointness, leakage-safe transforms,
   ROC-AUC computation, saved prediction alignment, blend shapes, and submission order/range.

## Acceptance criteria

- `uv sync` and `uv run pytest` pass from the project root.
- EDA findings document validation choice and leakage risks from the real schema.
- Every full learner produces reproducible OOF/test artifacts and metadata.
- At least logistic regression plus three GBDTs are compared on identical folds.
- The submitted CSV passes all local checks and Kaggle accepts and scores it.
