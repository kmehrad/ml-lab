# Initial Problem Analysis

## Problem Overview

The goal of the Home Credit Default Risk competition is to estimate the
probability that a loan applicant will experience payment difficulties.

This is a supervised, tabular, binary-classification problem:

- One row in `application_train.csv` represents one current loan application.
- `TARGET = 1` indicates payment difficulties.
- `TARGET = 0` indicates no recorded payment difficulties.
- Predictions for `application_test.csv` must be probabilities, not only
  binary decisions.

## Dataset

The training set contains 307,511 applications and 122 columns: 121 input
columns plus the target. The test set contains 48,744 applications and the
same inputs.

Only 24,825 training cases, approximately 8.1%, have `TARGET = 1`. This makes
the problem substantially imbalanced. Accuracy is misleading because always
predicting the majority class would be about 91.9% accurate while providing no
useful risk ranking.

## Main Feature Groups

The application table contains:

- Loan details: credit amount, annuity, goods price, and contract type.
- Financial data: income, employment duration, and income type.
- Applicant profile: age, education, occupation, family status, and household
  size.
- Assets and housing: car and real-estate ownership, housing type, and building
  characteristics.
- Location and contact indicators: regional ratings, address consistency, and
  phone or email flags.
- External risk scores: `EXT_SOURCE_1`, `EXT_SOURCE_2`, and `EXT_SOURCE_3`.
- Credit inquiries and supplied-document indicators.

The external risk scores have some of the strongest direct relationships with
the target. Useful engineered features may include credit-to-income,
annuity-to-income, credit-to-annuity, employment-to-age, and income-per-family-
member ratios.

## Historical Credit Tables

Six supporting tables provide one-to-many historical records:

- `bureau.csv`: loans reported by other financial institutions.
- `bureau_balance.csv`: monthly status of bureau loans.
- `previous_application.csv`: earlier Home Credit applications.
- `POS_CASH_balance.csv`: monthly point-of-sale and cash-loan history.
- `credit_card_balance.csv`: monthly credit-card activity.
- `installments_payments.csv`: scheduled and actual installment payments.

These tables must be aggregated to the applicant level. Useful summaries
include record counts, averages, recency, overdue frequency, payment
shortfalls, and maximum delinquency.

## Data-Quality Considerations

Missing data is significant: 41 application columns are at least 50% missing,
and several housing fields are approximately 60–70% missing.
`EXT_SOURCE_1` is about 56% missing. Missingness may itself contain predictive
information, so incomplete columns should not be dropped automatically.

Time columns are generally negative day offsets relative to the application
date. They require careful conversion, and known sentinel values—particularly
in employment duration—must be identified before modeling.

## Evaluation and Modeling Approach

The competition metric is ROC AUC. It measures whether the model ranks risky
applicants above non-risky applicants across all classification thresholds:

- `0.5` represents random ranking.
- `1.0` represents perfect ranking.
- Higher values are better.

ROC AUC is appropriate for the imbalanced target and probability-ranking
objective. Validation should use stratified cross-validation and report
out-of-fold ROC AUC.

Gradient-boosted decision trees, particularly LightGBM or CatBoost, are a
strong fit because the data is tabular, nonlinear, mixed-type, sparse, and
contains substantial missingness. A simple linear or logistic-regression
baseline is still useful for validating preprocessing and establishing an
initial benchmark.
