"""
features.py — Shared feature engineering for South Sudan Flood Prediction

Single source of truth for the feature formulas used by both train.py
(offline training) and app.py (live inference), so the two can never drift
out of sync with each other.
"""

import numpy as np
import pandas as pd

TARGET = "flood"

RAW_FEATURES = [
    "rainfall_mm", "soil_moisture_mm",
    "max_temperature_celsius", "min_temperature_celsius",
    "vapor_pressure_deficit_kPa",
    "wetland_fraction", "elevation_m", "slope_deg", "ndvi",
    "flood_prev_month",
]

ENGINEERED_FEATURES = [
    "temp_range", "wetness_index", "rain_wetland", "month_sin", "month_cos",
]

FEATURES = [
    # Core climate
    "rainfall_mm", "soil_moisture_mm",
    "max_temperature_celsius", "min_temperature_celsius",
    "vapor_pressure_deficit_kPa",
    # Terrain & land cover  (water_fraction deliberately excluded — label leakage)
    "wetland_fraction", "elevation_m", "slope_deg", "ndvi",
    # Temporal lag
    "flood_prev_month",
    # Engineered
    "temp_range", "wetness_index", "rain_wetland",
    "month_sin", "month_cos",
]

STATIC_GEO = ["wetland_fraction", "elevation_m", "slope_deg"]

# Additional lag/rolling features (Phase C — richer temporal signal beyond
# the single flood_prev_month persistence flag). Computed per-county,
# chronologically, so no cross-county leakage occurs.
LAG_FEATURES = [
    "rainfall_lag1", "rainfall_roll3", "soil_moisture_roll3", "flood_count_last_12mo",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the 5 engineered climate/temporal features to a raw feature frame.

    Mutates and returns df. Requires columns: max_temperature_celsius,
    min_temperature_celsius, rainfall_mm, soil_moisture_mm, wetland_fraction,
    month (1-12).
    """
    df["temp_range"]    = df["max_temperature_celsius"] - df["min_temperature_celsius"]
    df["wetness_index"] = (df["rainfall_mm"] * df["soil_moisture_mm"]) / 1000.0
    df["rain_wetland"]  = df["rainfall_mm"] * df["wetland_fraction"]
    df["month_sin"]     = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]     = np.cos(2 * np.pi * df["month"] / 12)
    return df


def engineer_features_row(raw: dict) -> dict:
    """Same formulas as engineer_features(), for a single dict of raw inputs
    (used at inference time by the API, one county/month at a time).
    """
    r = raw.copy()
    r["temp_range"]    = r["max_temperature_celsius"] - r["min_temperature_celsius"]
    r["wetness_index"] = (r["rainfall_mm"] * r["soil_moisture_mm"]) / 1000.0
    r["rain_wetland"]  = r["rainfall_mm"] * r["wetland_fraction"]
    r["month_sin"]     = np.sin(2 * np.pi * r["month"] / 12)
    r["month_cos"]     = np.cos(2 * np.pi * r["month"] / 12)
    return r


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-county chronological lag/rolling climate features.

    Must be called on a frame containing every county's full monthly history
    (not a pre-filtered train/test split) so lag/rolling windows are computed
    over real consecutive months before the split happens. Grouping by county
    and sorting by (year, month) prevents one county's rolling window from
    picking up another county's values.
    """
    df = df.sort_values(["county", "year", "month"]).reset_index(drop=True)
    g = df.groupby("county")

    df["rainfall_lag1"] = g["rainfall_mm"].shift(1)
    df["rainfall_roll3"] = (
        g["rainfall_mm"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    )
    df["soil_moisture_roll3"] = (
        g["soil_moisture_mm"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    )
    df["flood_count_last_12mo"] = (
        g["flood"].transform(lambda s: s.shift(1).rolling(12, min_periods=1).sum())
    )

    # First observed month per county has no prior history — fall back to the
    # county's own current-row climate values (best available proxy) and 0 floods.
    df["rainfall_lag1"] = df["rainfall_lag1"].fillna(df["rainfall_mm"])
    df["rainfall_roll3"] = df["rainfall_roll3"].fillna(df["rainfall_mm"])
    df["soil_moisture_roll3"] = df["soil_moisture_roll3"].fillna(df["soil_moisture_mm"])
    df["flood_count_last_12mo"] = df["flood_count_last_12mo"].fillna(0.0)
    return df
