# EnergyAnomaly — Large-scale Energy Anomaly Detection (LEAD)

Solution to the Kaggle competition
[`energy-anomaly-detection`](https://www.kaggle.com/competitions/energy-anomaly-detection):
classify each hourly electricity meter reading as anomalous (`1`) or normal (`0`).

- **Metric:** AUC-ROC (submit probabilities). **Train:** 200 buildings (~1.75M rows).
  **Test:** 206 *disjoint* buildings. Full-year 2016 hourly data.
- **CV:** `GroupKFold` by `building_id` (test buildings are unseen — random KFold leaks).

## Setup (uv)

```bash
uv sync                                   # create env from pyproject.toml / uv.lock
export KAGGLE_API_TOKEN="$(cat ~/.kaggle/access_token)"
uv run kaggle competitions download -c energy-anomaly-detection -p data/raw
unzip -o data/raw/energy-anomaly-detection.zip -d data/raw
```

The competition ships `train.csv`/`test.csv` (minimal) and `train_features.csv`/`test_features.csv`
(pre-engineered: metadata, weather, weather lag-aggregates, calendar, and `gte_*` target encodings).

## EDA

- Notebook: [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) (executed, figures in `reports/figures/`)
- Findings: [`reports/EDA_FINDINGS.md`](reports/EDA_FINDINGS.md)

Headline findings: anomaly rate 2.13%; `meter_reading == 1.0` → anomaly 99.97% (stuck-meter
flatlines, ~46% of all anomalies); train/test buildings disjoint; linear feature↔target
correlations are negligible (tree models required).

## Layout

`src/` pipeline · `notebooks/` EDA · `reports/` findings & figures · `experiments/` log & artifacts
· `data/` (git-ignored) · `models/` `outputs/` (git-ignored).
