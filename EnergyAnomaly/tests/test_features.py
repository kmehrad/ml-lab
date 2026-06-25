import numpy as np
import pandas as pd

from src.features import add_features, feature_columns, _run_length


def _toy():
    # two buildings, ordered hourly; building 2 has a stuck-at-1.0 flatline run
    ts = pd.date_range("2016-01-01", periods=6, freq="h")
    rows = []
    for b, vals in [(1, [10, 12, 11, 13, 14, 15]), (2, [5, 1, 1, 1, 20, 21])]:
        for t, v in zip(ts, vals):
            rows.append({"building_id": b, "timestamp": t, "meter_reading": float(v),
                         "site_id": 0, "primary_use": "Office", "square_feet": 1000,
                         "year_built": 2000, "floor_count": 1,
                         "air_temperature": 5.0, "cloud_coverage": 255, "dew_temperature": 1.0,
                         "precip_depth_1_hr": 0, "sea_level_pressure": 1000.0,
                         "wind_direction": 90.0, "wind_speed": 3.0,
                         **{c: 0.0 for c in feature_columns() if c.endswith(("lag7", "lag73"))},
                         "hour": t.hour, "weekday": 4, "month": 1,
                         "hour_x": 0.0, "hour_y": 1.0, "month_x": 1.0, "month_y": 0.0,
                         "weekday_x": 0.0, "weekday_y": 1.0, "is_holiday": 1, "anomaly": 0})
    return pd.DataFrame(rows)


def test_run_length_counts_consecutive_constants():
    s = pd.Series([1, 1, 1, 2, 2, 3])
    assert _run_length(s).tolist() == [1, 2, 3, 1, 2, 1]


def test_flatline_and_flags():
    out = add_features(_toy())
    b2 = out[out.building_id == 2].reset_index(drop=True)
    # the three consecutive 1.0 readings form a run of length 1,2,3
    assert b2.loc[b2.meter_reading == 1.0, "flatline_run"].tolist() == [1, 2, 3]
    assert b2["is_reading_one"].sum() == 3


def test_lags_do_not_cross_buildings():
    out = add_features(_toy())
    # first row of each building has no within-building lag_1
    firsts = out.groupby("building_id").head(1)
    assert firsts["lag_1"].isna().all()


def test_weather_encoding():
    out = add_features(_toy())
    assert (out["cloud_missing"] == 1).all()          # 255 sentinel flagged
    assert out["cloud_coverage"].isna().all()          # ...and restored to NaN
    assert np.allclose(out["wind_dir_sin"], 1.0)       # 90 degrees -> sin=1
    assert np.allclose(out["wind_dir_cos"], 0.0, atol=1e-9)
