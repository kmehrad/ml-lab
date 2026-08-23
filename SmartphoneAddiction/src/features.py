from __future__ import annotations

import numpy as np
import pandas as pd
from .data import CATEGORICAL, FEATURES

COMPACT = ["daily_screen_time_hours", "weekend_screen_time", "social_media_hours",
           "gaming_hours", "work_study_hours", "notifications_per_day", "app_opens_per_day"]


def _ratio(a, b):
    return a / b.replace(0, np.nan)


def make_features(df: pd.DataFrame, group: str = "base") -> pd.DataFrame:
    if group not in {"base", "behavior", "compact", "source_rule", "missing"}:
        raise ValueError(f"unknown feature group: {group}")
    x = df[FEATURES].copy()
    if group == "compact":
        return x[COMPACT]
    if group in {"behavior", "source_rule", "missing"}:
        d, w = x.daily_screen_time_hours, x.weekend_screen_time
        components = x.social_media_hours + x.gaming_hours + x.work_study_hours
        x["screen_weekend_delta"] = w - d
        x["screen_weekend_ratio"] = _ratio(w, d)
        x["component_hours_total"] = components
        x["component_hours_residual"] = d - components
        x["social_share"] = _ratio(x.social_media_hours, d)
        x["gaming_share"] = _ratio(x.gaming_hours, d)
        x["opens_per_notification"] = _ratio(x.app_opens_per_day, x.notifications_per_day)
        x["notifications_per_open"] = _ratio(x.notifications_per_day, x.app_opens_per_day)
        x["screen_per_waking_hour"] = _ratio(d, 24 - x.sleep_hours)
    if group == "source_rule":
        valid = d.notna() & x.social_media_hours.notna()
        score = (d >= 7).astype(float) + (x.social_media_hours >= 3).astype(float)
        x["source_usage_band"] = score.where(valid, 1.0)
    if group == "missing":
        raw = df[FEATURES]
        for c in FEATURES:
            x[f"{c}__missing"] = raw[c].isna().astype("int8")
        x["missing_count"] = raw.isna().sum(axis=1).astype("int8")
    numeric = x.select_dtypes(include=np.number).columns
    x[numeric] = x[numeric].replace([np.inf, -np.inf], np.nan)
    return x
