# EDA Findings

Generated from the competition files by `notebooks/eda_build.py`.

## Data

- Train: 691,369 rows; test: 296,302 rows; 12 predictors.
- Positive target rate: 70.9424%.
- Duplicate predictor rows in train: 0. IDs remain unique and are never model features.
- Numeric ranges were inspected through `describe`; categorical cardinalities are recorded below.

## Signal and drift

- Strongest univariate numeric AUCs: daily_screen_time_hours=0.8646, weekend_screen_time=0.8513, social_media_hours=0.8168, work_study_hours=0.6495, gaming_hours=0.6075.
- Highest mutual-information variables: daily_screen_time_hours, weekend_screen_time, social_media_hours, notifications_per_day, app_opens_per_day.
- Missingness is separated from value drift. Largest absolute missing-rate shifts: social_media_hours, app_opens_per_day, daily_screen_time_hours, academic_work_impact, work_study_hours.
- Missing flags and missing count are isolated in the `missing` ablation; they are not part of `base`.
- Two-dimensional target-rate surfaces cover daily/social and daily/weekend interactions. Model permutation importance is saved alongside them.

## Feature decisions

- Evaluate `base`, behavioral interactions, the seven-column `compact` set, and the neutral-on-missing `source_rule` independently on shared folds.
- Preserve native missing values where the estimator supports them; otherwise imputation is learned inside each fold.
- Treat demographic/context predictors as optional weak features, not assumed signal.

## Schema and missingness

```
                           dtype  cardinality  train_missing  test_missing  missing_drift
age                      float64           18       0.041843      0.057840       0.015997
daily_screen_time_hours  float64         1389       0.138644      0.110657      -0.027986
social_media_hours       float64          721       0.193811      0.159962      -0.033849
gaming_hours             float64          401       0.183435      0.200539       0.017104
work_study_hours         float64          600       0.074516      0.093746       0.019230
sleep_hours              float64          451       0.064336      0.075784       0.011448
notifications_per_day    float64          231       0.097754      0.115494       0.017740
app_opens_per_day        float64          166       0.116739      0.086753      -0.029987
weekend_screen_time      float64         1437       0.162089      0.171099       0.009011
gender                       str            3       0.041995      0.047965       0.005970
stress_level                 str            3       0.079766      0.066236      -0.013530
academic_work_impact         str            2       0.063966      0.086807       0.022841
```

## Categorical target rates

```
{'gender': {'Female': {'mean': 0.7038, 'size': 221595}, 'Male': {'mean': 0.7232, 'size': 223662}, 'Other': {'mean': 0.701, 'size': 217078}, nan: {'mean': 0.7088, 'size': 29034}}, 'stress_level': {'High': {'mean': 0.7114, 'size': 220873}, 'Low': {'mean': 0.7112, 'size': 207783}, 'Medium': {'mean': 0.7057, 'size': 207565}, nan: {'mean': 0.709, 'size': 55148}}, 'academic_work_impact': {'No': {'mean': 0.711, 'size': 316579}, 'Yes': {'mean': 0.7079, 'size': 330566}, nan: {'mean': 0.7095, 'size': 44224}}}
```
