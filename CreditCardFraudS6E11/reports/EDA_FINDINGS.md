# EDA findings

## Dataset

- Train: 699,960 rows and 26 columns; test: 299,984 rows and 25 columns.
- Target `is_fraud`: 11,689 positive rows (1.66995%) and 688,271 negative rows.
- 24 predictors: 13 numeric, 5 categorical, and 6 boolean flags. There are no missing values,
  duplicate feature rows, or categorical levels present in only one split.
- No entity ID or full timestamp exists. `id` is a row identifier and must not be a feature.

## Signal checks

Strongest single numeric rankings are `merchant_risk_score` (direction-free AUC 0.749),
`velocity_score` (0.711), `cvv_retry_count` (0.708), `customer_age` (0.625), and
`txn_count_last_24h` (0.620). The largest boolean target-rate changes occur for
`ip_country_mismatch` (16.5% fraud when true), AI-generated scam attempt (15.2%), billing/shipping
mismatch (13.0%), and foreign transaction (12.6%), versus 1.67% overall.

`merchant_category` and `auth_method` also vary meaningfully in fraud rate. `channel`,
`device_type`, and `card_type` show weaker marginal separation but may contribute interactions.

## Validation and next checks

Use five-fold shuffled `StratifiedKFold` with fixed seed 42 because no time sequence or entity group
is available. Optimize official ROC-AUC and report average precision as the rare-event diagnostic.
Before feature experiments, quantify train/test distribution drift for every feature and compare a
fold-safe logistic baseline against native-categorical LightGBM, XGBoost, and CatBoost.
