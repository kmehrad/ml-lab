# EDA findings

Train rows: 668,665; test rows: 286,571.
Features: 13 (7 numeric, 6 categorical).
Positive class share: 17.4645%.
Duplicate feature rows in train: 0.

## Missingness

| Feature | Train | Test |
|---|---:|---:|
| Age | 0.00% | 0.00% |
| Annual_Income_USD | 0.00% | 0.00% |
| Daily_Commute_km | 0.00% | 0.00% |
| Number_of_Cars_Owned | 0.00% | 0.00% |
| Charging_Stations_Near_Home | 0.00% | 0.00% |
| Charging_Stations_Near_Work | 0.00% | 0.00% |
| Environmental_Concern_Level | 0.00% | 0.00% |
| Gender | 0.00% | 0.00% |
| City_Type | 0.00% | 0.00% |
| Current_Car_Type | 0.00% | 0.00% |
| Home_Charging_Possible | 0.00% | 0.00% |
| Subsidy_Available | 0.00% | 0.00% |
| Range_Anxiety_Level | 0.00% | 0.00% |

## Categorical cardinality and target rate

### Gender (3 levels)

| Gender   |   count |     mean |
|:---------|--------:|---------:|
| Male     |  367954 | 0.172253 |
| Female   |  295427 | 0.177641 |
| Other    |    5284 | 0.173732 |

### City_Type (3 levels)

| City_Type   |   count |     mean |
|:------------|--------:|---------:|
| Urban       |  289305 | 0.161058 |
| Suburban    |  255377 | 0.180936 |
| Rural       |  123983 | 0.193389 |

### Current_Car_Type (4 levels)

| Current_Car_Type   |   count |     mean |
|:-------------------|--------:|---------:|
| Sedan              |  303459 | 0.171967 |
| SUV                |  246545 | 0.180953 |
| Hatchback          |   79438 | 0.174299 |
| Truck              |   39223 | 0.156413 |

### Home_Charging_Possible (2 levels)

| Home_Charging_Possible   |   count |     mean |
|:-------------------------|--------:|---------:|
| Yes                      |  462677 | 0.195819 |
| No                       |  205988 | 0.127085 |

### Subsidy_Available (2 levels)

| Subsidy_Available   |   count |       mean |
|:--------------------|--------:|-----------:|
| Yes                 |  419909 | 0.274695   |
| No                  |  248756 | 0.00575665 |

### Range_Anxiety_Level (3 levels)

| Range_Anxiety_Level   |   count |       mean |
|:----------------------|--------:|-----------:|
| Low                   |  603972 | 0.189027   |
| Medium                |   62499 | 0.0417447  |
| High                  |    2194 | 0.00136737 |

## Train/test numeric shifts

| Feature | Train mean | Test mean |
|---|---:|---:|
| Age | 47.04 | 47.07 |
| Annual_Income_USD | 8.477e+04 | 8.48e+04 |
| Daily_Commute_km | 32.16 | 32.16 |
| Number_of_Cars_Owned | 1.713 | 1.716 |
| Charging_Stations_Near_Home | 4.96 | 4.951 |
| Charging_Stations_Near_Work | 7.176 | 7.156 |
| Environmental_Concern_Level | 2.935 | 2.935 |

Review these shifts and class rates before creating features; target-derived encoders must fit within folds.
