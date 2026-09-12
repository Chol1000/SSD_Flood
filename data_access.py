"""
data_access.py — Artifact loading and dataframe filtering for the API.
"""

import json
import pickle

import pandas as pd


def load_model_bundle(path: str = "model/best_model.pkl") -> dict:
    """Load the {nowcast_model, outlook_model} bundle written by train.py."""
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    return bundle


def load_artifacts(model_dir: str = "model"):
    with open(f"{model_dir}/metadata.json")              as f: meta        = json.load(f)
    with open(f"{model_dir}/counties.json")              as f: counties    = json.load(f)
    with open(f"{model_dir}/feature_stats.json")         as f: fstats      = json.load(f)
    with open(f"{model_dir}/county_defaults.json")       as f: cdefaults   = json.load(f)
    with open(f"{model_dir}/county_climate_percentiles.json") as f: cpercentiles = json.load(f)
    hist    = pd.read_csv(f"{model_dir}/county_flood_history.csv")
    monthly = pd.read_csv(f"{model_dir}/monthly_flood_data.csv")
    return meta, counties, fstats, cdefaults, cpercentiles, hist, monthly


def filter_monthly(monthly_df: pd.DataFrame, counties: list, year_range: tuple, months: list) -> pd.DataFrame:
    """Slice monthly_df (columns: county, year, month, flood) by a set of
    counties, an inclusive (min_year, max_year) range, and a set of month
    numbers (1-12). Used by every chart on the Historical Analysis tab so
    the filters stay in sync across the whole tab.
    """
    out = monthly_df[
        monthly_df["county"].isin(counties)
        & monthly_df["year"].between(year_range[0], year_range[1])
        & monthly_df["month"].isin(months)
    ]
    return out
