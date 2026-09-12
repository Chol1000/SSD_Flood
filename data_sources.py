"""
data_sources.py — Live daily climate data for the flood nowcast.

Pulls daily rainfall/temperature/soil-moisture/VPD from Open-Meteo (primary)
for a county's coordinates, and derives the month-to-date / lag / rolling
inputs the model needs. If Open-Meteo is unreachable (e.g. its shared
anonymous rate limit is exhausted), falls back to NASA POWER — a free,
keyless, agriculture-oriented reanalysis API with generous, non-shared
limits — before finally falling back to that county's historical median from
county_defaults.json. Anything neither live source can supply (NDVI,
wetland_fraction, elevation_m, slope_deg, flood_prev_month — see README for
why) always comes from the historical median, tagged so the UI can show which
fields are live vs. historical.

Set OPEN_METEO_API_KEY to use Open-Meteo's free API-key tier instead of the
anonymous endpoint — quota is then tied to the key, not the shared IP, which
is what anonymous requests are rate-limited by. Get one at
https://open-meteo.com/en/pricing.

No data source here is daily-resolution flood *ground truth* — none exists.
This module only replaces frozen training-time climate inputs with live ones.
"""

import datetime as dt
import os
import time

import numpy as np
import pandas as pd
import requests

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_CUSTOMER_URL = "https://customer-api.open-meteo.com/v1/forecast"
OPEN_METEO_API_KEY = os.environ.get("OPEN_METEO_API_KEY")

NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
NASA_POWER_LAG_DAYS = 2  # NASA POWER's reanalysis isn't published same-day

REQUEST_TIMEOUT = 12  # seconds — fail fast and fall back rather than hang the UI
REQUEST_RETRIES = 2   # transient network blips shouldn't surface as a user-visible failure
CACHE_TTL_SECONDS = 900  # 15 min — climate data doesn't change fast enough to justify refetching more often

# Fields neither live source can supply; always sourced from historical medians.
STATIC_FALLBACK_FIELDS = ["ndvi", "wetland_fraction", "elevation_m", "slope_deg", "flood_prev_month"]

_cache: dict[tuple, tuple[float, pd.DataFrame]] = {}
_nasa_cache: dict[tuple, tuple[float, pd.DataFrame]] = {}


def fetch_open_meteo_daily(lat: float, lon: float, past_days: int = 92) -> pd.DataFrame:
    """Fetch ~`past_days` of daily climate data (blended reanalysis + forecast)
    for one lat/lon. Raises requests.RequestException / KeyError on failure —
    callers must catch and fall back.

    Cached for CACHE_TTL_SECONDS per (lat, lon, past_days) — scanning 79
    counties (or repeat visits to the same county) would otherwise re-hit
    Open-Meteo hard enough to trip its rate limiting (HTTP 429).
    """
    cache_key = (round(lat, 3), round(lon, 3), past_days)
    cached = _cache.get(cache_key)
    if cached and (time.monotonic() - cached[0]) < CACHE_TTL_SECONDS:
        return cached[1]

    params = {
        "latitude": lat, "longitude": lon,
        "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min",
        "hourly": "soil_moisture_0_to_7cm,vapour_pressure_deficit",
        "past_days": past_days,
        "forecast_days": 1,
        "timezone": "auto",
    }
    url = OPEN_METEO_URL
    if OPEN_METEO_API_KEY:
        url = OPEN_METEO_CUSTOMER_URL
        params["apikey"] = OPEN_METEO_API_KEY
    last_exc = None
    for attempt in range(REQUEST_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 429:
                # Respect the server's own backoff hint rather than hammering it.
                wait = float(resp.headers.get("Retry-After", 2 * (attempt + 1)))
                if attempt < REQUEST_RETRIES:
                    time.sleep(min(wait, 4))
                    continue
            resp.raise_for_status()
            data = resp.json()
            break
        except requests.RequestException as exc:
            last_exc = exc
    else:
        raise last_exc

    daily = pd.DataFrame({
        "date": pd.to_datetime(data["daily"]["time"]),
        "rainfall_mm": data["daily"]["precipitation_sum"],
        "max_temperature_celsius": data["daily"]["temperature_2m_max"],
        "min_temperature_celsius": data["daily"]["temperature_2m_min"],
    })

    hourly = pd.DataFrame({
        "time": pd.to_datetime(data["hourly"]["time"]),
        "soil_moisture_frac": data["hourly"]["soil_moisture_0_to_7cm"],
        "vapor_pressure_deficit_kPa": data["hourly"]["vapour_pressure_deficit"],
    })
    hourly["date"] = hourly["time"].dt.floor("D")
    hourly_daily = (hourly.groupby("date")[["soil_moisture_frac", "vapor_pressure_deficit_kPa"]]
                     .mean().reset_index())

    daily = daily.merge(hourly_daily, on="date", how="left")
    # Open-Meteo's soil_moisture_0_to_7cm is volumetric water content (m3/m3),
    # a different measurement than the training dataset's TerraClimate-derived
    # soil moisture (mm of water depth). This is an approximate proxy scaled
    # to a comparable range (assumes ~700mm effective root-zone depth), not an
    # exact unit match — documented as a known limitation, not silently hidden.
    daily["soil_moisture_mm"] = daily["soil_moisture_frac"] * 700
    result_df = daily.drop(columns=["soil_moisture_frac"]).dropna(subset=["rainfall_mm"])
    _cache[cache_key] = (time.monotonic(), result_df)
    return result_df


def fetch_nasa_power_daily(lat: float, lon: float, past_days: int = 92) -> pd.DataFrame:
    """Fallback climate source, used only when Open-Meteo fails. NASA POWER
    (power.larc.nasa.gov) is free, keyless, and rate-limited generously enough
    for this app's traffic — but it lags ~2 days behind real time (a
    reanalysis product, not a same-day forecast blend like Open-Meteo), so
    it's a fallback, not a replacement.

    Raises requests.RequestException / KeyError on failure — callers must
    catch and fall back further (to historical medians).
    """
    cache_key = (round(lat, 3), round(lon, 3), past_days)
    cached = _nasa_cache.get(cache_key)
    if cached and (time.monotonic() - cached[0]) < CACHE_TTL_SECONDS:
        return cached[1]

    end = dt.date.today() - dt.timedelta(days=NASA_POWER_LAG_DAYS)
    start = end - dt.timedelta(days=past_days)
    params = {
        "parameters": "PRECTOTCORR,T2M_MAX,T2M_MIN,T2M,RH2M,GWETROOT",
        "community": "AG",
        "longitude": lon, "latitude": lat,
        "start": start.strftime("%Y%m%d"), "end": end.strftime("%Y%m%d"),
        "format": "JSON",
    }
    resp = requests.get(NASA_POWER_URL, params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    param_data = resp.json()["properties"]["parameter"]

    dates = sorted(param_data["PRECTOTCORR"].keys())
    series = {name: [param_data[name].get(d) for d in dates] for name in
              ("PRECTOTCORR", "T2M_MAX", "T2M_MIN", "T2M", "RH2M", "GWETROOT")}

    daily = pd.DataFrame({
        "date": pd.to_datetime(dates, format="%Y%m%d"),
        "rainfall_mm": series["PRECTOTCORR"],
        "max_temperature_celsius": series["T2M_MAX"],
        "min_temperature_celsius": series["T2M_MIN"],
        "_t2m_mean": series["T2M"],
        "_rh2m": series["RH2M"],
        "_gwetroot": series["GWETROOT"],
    })
    daily = daily.replace(-999, np.nan)  # NASA POWER's missing-value sentinel

    # No direct VPD field from NASA POWER — derive it from mean temperature and
    # relative humidity via the Tetens saturation-vapor-pressure formula.
    svp = 0.6108 * np.exp(17.27 * daily["_t2m_mean"] / (daily["_t2m_mean"] + 237.3))
    daily["vapor_pressure_deficit_kPa"] = svp * (1 - daily["_rh2m"] / 100)

    # GWETROOT is root-zone wetness as a fraction of field capacity (0-1), a
    # different measurement than Open-Meteo's volumetric water content — reuse
    # the same ~700mm proxy scaling used above for a comparable range, not an
    # exact unit match.
    daily["soil_moisture_mm"] = daily["_gwetroot"] * 700

    result_df = daily.drop(columns=["_t2m_mean", "_rh2m", "_gwetroot"]).dropna(subset=["rainfall_mm"])
    _nasa_cache[cache_key] = (time.monotonic(), result_df)
    return result_df


def get_live_county_inputs(county: str, county_coords: dict, county_defaults: dict,
                            full_history: bool = True) -> dict:
    """Fetch live daily data for `county` and derive nowcast model inputs.

    Returns:
      inputs: dict of raw feature values (climate fields live where available,
        everything else — including lag/rolling fields when full_history=False
        — from historical medians).
      field_source: dict tagging each field "live" or "historical_fallback".
      live_data_available: bool.
      last_updated: ISO date string of the latest live observation used, or None.
      error: human-readable message if live data could not be used, else None.
    """
    defaults = dict(county_defaults.get(county, {}))
    result = {
        "inputs": dict(defaults),
        "field_source": {f: "historical_fallback" for f in defaults},
        "live_data_available": False,
        "last_updated": None,
        "source": None,
        "error": None,
    }

    if county not in county_coords:
        result["error"] = f"No coordinates known for county '{county}' — showing historical medians."
        return result

    lat, lon = county_coords[county]
    past_days = 92 if full_history else 35
    daily = None
    source = None
    try:
        daily = fetch_open_meteo_daily(lat, lon, past_days=past_days)
        source = "open-meteo"
    except Exception as om_exc:
        # Full detail (including the request URL) goes server-side only — the
        # user-facing message stays short so it never breaks page layout.
        print(f"[live-fetch] {county} open-meteo failed: {om_exc}")
        try:
            daily = fetch_nasa_power_daily(lat, lon, past_days=past_days)
            source = "nasa-power"
        except Exception as power_exc:
            print(f"[live-fetch] {county} nasa-power fallback failed: {power_exc}")
            status = getattr(getattr(om_exc, "response", None), "status_code", None)
            if status == 429:
                result["error"] = "Live weather provider is rate-limited right now — showing historical medians."
            else:
                result["error"] = "Live data temporarily unavailable — showing historical medians."
            return result

    if daily.empty:
        result["error"] = "Live data source returned no rows — showing historical medians."
        return result

    today_row = daily.iloc[-1]
    this_month_mask = ((daily["date"].dt.year == today_row["date"].year) &
                        (daily["date"].dt.month == today_row["date"].month))
    month_to_date = daily[this_month_mask]

    live_fields = {
        "rainfall_mm": month_to_date["rainfall_mm"].sum(),
        "max_temperature_celsius": month_to_date["max_temperature_celsius"].mean(),
        "min_temperature_celsius": month_to_date["min_temperature_celsius"].mean(),
        "soil_moisture_mm": month_to_date["soil_moisture_mm"].mean(),
        "vapor_pressure_deficit_kPa": month_to_date["vapor_pressure_deficit_kPa"].mean(),
    }
    got_any_live = False
    for field, value in live_fields.items():
        if pd.notnull(value):
            result["inputs"][field] = round(float(value), 4)
            result["field_source"][field] = "live"
            got_any_live = True

    if full_history:
        prior_period = today_row["date"].to_period("M") - 1
        prior_mask = daily["date"].dt.to_period("M") == prior_period
        if prior_mask.any():
            result["inputs"]["rainfall_lag1"] = round(float(daily.loc[prior_mask, "rainfall_mm"].sum()), 4)
            result["field_source"]["rainfall_lag1"] = "live"

        three_mo_cutoff = today_row["date"] - pd.Timedelta(days=90)
        recent = daily[(daily["date"] >= three_mo_cutoff) & (daily["date"] < today_row["date"].normalize())]
        if len(recent) >= 30:  # require at least ~1 month of history for a meaningful average
            result["inputs"]["rainfall_roll3"] = round(float(recent["rainfall_mm"].sum() / (len(recent) / 30.0)), 4)
            result["inputs"]["soil_moisture_roll3"] = round(float(recent["soil_moisture_mm"].mean()), 4)
            result["field_source"]["rainfall_roll3"] = "live"
            result["field_source"]["soil_moisture_roll3"] = "live"

    # flood_count_last_12mo requires satellite flood detection, not available
    # live — always historical (already the default above).

    result["live_data_available"] = got_any_live
    if got_any_live:
        result["last_updated"] = today_row["date"].strftime("%Y-%m-%d")
        result["source"] = source
    else:
        result["error"] = "Live fetch returned no usable values for this month — showing historical medians."

    return result
