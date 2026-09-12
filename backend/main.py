"""
backend/main.py — FastAPI service for the South Sudan Flood EWS React dashboard.

Wraps the same model/data logic used by train.py — features.py, models.py,
data_access.py, data_sources.py, geo.py — so there is exactly one
implementation of feature engineering and inference, with the React dashboard
as its single front end.
"""

import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import datetime as dt
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import data_access
import data_sources
import weather_live
from features import engineer_features_row, LAG_FEATURES
from geo import COUNTY_COORDS
from models import load_outlook_bundle, predict_outlook, predict_outlook_multistep, SEQ_LEN, SEQ_FEATURES, STATIC_FEATURES_SEQ

from .report import build_county_report, build_national_report

# ── Background live-data refresh ───────────────────────────────────────────────
# On-demand fetching (one Open-Meteo call per user request) is what was
# tripping the 429 rate limit and making requests hang waiting on retries.
# Instead, a background thread refreshes all 79 counties on a timer, gently
# (a handful at a time), and every request just reads the in-memory result —
# instant and never dependent on Open-Meteo being fast or under quota right
# at the moment a user clicks something.
LIVE_REFRESH_INTERVAL_SECONDS = 20 * 60  # 20 min — climate inputs don't need to be fresher than this
_live_cache: dict[str, dict] = {}
_live_cache_lock = threading.Lock()
_live_cache_updated_at: Optional[float] = None


def _refresh_live_cache_once():
    art = _artifacts()
    with ThreadPoolExecutor(max_workers=5) as pool:  # gentle — this runs unattended, no rush
        def _fetch(c):
            return c, data_sources.get_live_county_inputs(c, COUNTY_COORDS, art["county_defaults"], full_history=True)

        results = dict(pool.map(_fetch, art["counties"]))
    global _live_cache_updated_at
    with _live_cache_lock:
        for c, result in results.items():
            # A transient provider outage shouldn't discard a real reading from
            # 20 minutes ago — keep the last known-good live value until a new
            # fetch actually succeeds, rather than overwriting it with the
            # historical-median fallback on every failed cycle.
            prior = _live_cache.get(c)
            if not result.get("live_data_available") and prior and prior.get("live_data_available"):
                continue
            _live_cache[c] = result
        _live_cache_updated_at = time.monotonic()


def _live_refresh_loop():
    while True:
        try:
            _refresh_live_cache_once()
        except Exception as exc:  # background loop must never die
            print(f"[live-refresh] failed: {exc}")
        time.sleep(LIVE_REFRESH_INTERVAL_SECONDS)


# Same pattern for the OpenWeatherMap "current conditions" grid (all 79
# counties) — one background refresh instead of the frontend triggering 79
# on-demand calls, which would risk the free tier's per-minute limit.
WEATHER_REFRESH_INTERVAL_SECONDS = 10 * 60
WEATHER_TICKER_HISTORY_LEN = 40
_weather_grid_cache: dict[str, dict] = {}
_weather_history_cache: dict[str, list[dict]] = {}
_weather_grid_lock = threading.Lock()


def _refresh_weather_grid_once():
    if not weather_live.available():
        return
    art = _artifacts()
    with ThreadPoolExecutor(max_workers=5) as pool:
        def _fetch(c):
            lat, lon = COUNTY_COORDS[c]
            try:
                return c, weather_live.get_current_only(lat, lon)
            except Exception:
                return c, None

        results = {c: v for c, v in pool.map(_fetch, art["counties"]) if v is not None}
    with _weather_grid_lock:
        _weather_grid_cache.update(results)
        # Same OpenWeatherMap reading also feeds a rolling per-county history —
        # this is the Live Updates ticker's actual data source. Unlike
        # Open-Meteo, OpenWeatherMap's free tier is keyed to this app's own
        # API key rather than shared across every anonymous caller on this
        # network, so it isn't exposed to the same rate-limit risk.
        for c, reading in results.items():
            arr = _weather_history_cache.setdefault(c, [])
            arr.append({"t": time.time(), **reading})
            del arr[:-WEATHER_TICKER_HISTORY_LEN]


def _weather_refresh_loop():
    while True:
        try:
            _refresh_weather_grid_once()
        except Exception as exc:
            print(f"[weather-refresh] failed: {exc}")
        time.sleep(WEATHER_REFRESH_INTERVAL_SECONDS)


# ── Live Updates terminal ticker (Open-Meteo, world-open, no key) ────────────
# Fetched server-side once per interval for all 79 counties in a single
# batched request, so 79 concurrent browser tabs never each hammer Open-Meteo
# themselves and trip its rate limit — the failure mode this replaced.
OM_TICKER_REFRESH_SECONDS = 5 * 60
OM_TICKER_HISTORY_LEN = 40
OM_TICKER_VARS = "temperature_2m,relative_humidity_2m,apparent_temperature,weathercode,windspeed_10m,surface_pressure,is_day"
_om_ticker_cache: dict[str, list[dict]] = {}
_om_ticker_lock = threading.Lock()


def _seed_om_ticker_once():
    """Runs once at startup: backfills ~18 real past hours per county (not
    fabricated) so the terminal already shows a moving trend for the very
    first visitor, instead of an empty chart that fills in over hours."""
    art = _artifacts()
    counties = art["counties"]
    lats = ",".join(str(COUNTY_COORDS[c][0]) for c in counties)
    lons = ",".join(str(COUNTY_COORDS[c][1]) for c in counties)
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lats}&longitude={lons}&current={OM_TICKER_VARS}"
        f"&hourly={OM_TICKER_VARS}&past_days=1&forecast_days=1&timezone=auto"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    rows = resp.json()
    if isinstance(rows, dict):
        rows = [rows]
    now = time.time()
    seeded_cache: dict[str, list[dict]] = {}
    for county, row in zip(counties, rows):
        cur, hourly = row.get("current"), row.get("hourly")
        if not cur or not hourly:
            continue
        past_idx = [i for i, t in enumerate(hourly["time"]) if t <= cur["time"]]
        last_n = past_idx[-(OM_TICKER_HISTORY_LEN - 1):]
        seeded = [
            {
                "t": now - (len(last_n) - k) * 3600, "temp": hourly["temperature_2m"][idx],
                "humidity": hourly["relative_humidity_2m"][idx], "feels_like": hourly["apparent_temperature"][idx],
                "wind": hourly["windspeed_10m"][idx], "pressure": hourly["surface_pressure"][idx],
                "code": hourly["weathercode"][idx], "is_day": bool(hourly["is_day"][idx]),
            }
            for k, idx in enumerate(last_n)
        ]
        seeded.append({
            "t": now, "temp": cur["temperature_2m"], "humidity": cur["relative_humidity_2m"],
            "feels_like": cur["apparent_temperature"], "wind": cur["windspeed_10m"],
            "pressure": cur["surface_pressure"], "code": cur["weathercode"], "is_day": bool(cur["is_day"]),
        })
        seeded_cache[county] = seeded
    with _om_ticker_lock:
        _om_ticker_cache.update(seeded_cache)


def _refresh_om_ticker_once():
    """Lightweight periodic tick: one 'current conditions' point per county,
    appended to the seeded history."""
    art = _artifacts()
    counties = art["counties"]
    lats = ",".join(str(COUNTY_COORDS[c][0]) for c in counties)
    lons = ",".join(str(COUNTY_COORDS[c][1]) for c in counties)
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lats}&longitude={lons}&current={OM_TICKER_VARS}&timezone=auto"
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    rows = resp.json()
    if isinstance(rows, dict):
        rows = [rows]
    now = time.time()
    with _om_ticker_lock:
        for county, row in zip(counties, rows):
            cur = row.get("current")
            if not cur:
                continue
            reading = {
                "t": now, "temp": cur["temperature_2m"], "humidity": cur["relative_humidity_2m"],
                "feels_like": cur["apparent_temperature"], "wind": cur["windspeed_10m"],
                "pressure": cur["surface_pressure"], "code": cur["weathercode"], "is_day": bool(cur["is_day"]),
            }
            arr = _om_ticker_cache.setdefault(county, [])
            arr.append(reading)
            del arr[:-OM_TICKER_HISTORY_LEN]


def _om_ticker_loop():
    # Retry the initial seed with backoff — if Open-Meteo is mid rate-limit
    # window at boot, recover in under a minute rather than waiting a full
    # OM_TICKER_REFRESH_SECONDS cycle.
    for delay in (5, 15, 30, 60, 120):
        try:
            _seed_om_ticker_once()
            break
        except Exception as exc:
            print(f"[om-ticker-seed] failed, retrying in {delay}s: {exc}")
            time.sleep(delay)
    while True:
        time.sleep(OM_TICKER_REFRESH_SECONDS)
        try:
            _refresh_om_ticker_once()
        except Exception as exc:
            print(f"[om-ticker-refresh] failed: {exc}")


# ── Live flood-risk trend (deployed nowcast model, all 79 counties) ─────────
# Same idea as the Open-Meteo ticker above: computed server-side on a timer
# and shared across every client, so the trend is already flowing — hours
# deep — the moment any visitor opens the page, instead of resetting to
# empty per browser tab/session.
RISK_TREND_REFRESH_SECONDS = 10 * 60
RISK_TREND_HISTORY_LEN = 60
_risk_trend_cache: dict[str, list[dict]] = {}
_risk_trend_lock = threading.Lock()


def _refresh_risk_trend_once():
    art = _artifacts()
    month = dt.date.today().month
    now = time.time()
    readings: dict[str, dict] = {}
    for c in art["counties"]:
        try:
            live = get_cached_live(c)
            inputs = dict(live["inputs"])
            inputs["month"] = month
            p = run_prediction(inputs, c)
            readings[c] = {"t": now, "probability": p, "tier": risk_tier(p)}
        except Exception as exc:
            print(f"[risk-trend] {c} failed: {exc}")
    with _risk_trend_lock:
        for c, reading in readings.items():
            arr = _risk_trend_cache.setdefault(c, [])
            arr.append(reading)
            del arr[:-RISK_TREND_HISTORY_LEN]


def _risk_trend_loop():
    while True:
        try:
            _refresh_risk_trend_once()
        except Exception as exc:
            print(f"[risk-trend-refresh] failed: {exc}")
        time.sleep(RISK_TREND_REFRESH_SECONDS)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    threading.Thread(target=_live_refresh_loop, daemon=True).start()
    threading.Thread(target=_weather_refresh_loop, daemon=True).start()
    threading.Thread(target=_om_ticker_loop, daemon=True).start()
    threading.Thread(target=_risk_trend_loop, daemon=True).start()
    yield


app = FastAPI(title="SSD Flood EWS API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_cached_live(county: str) -> dict:
    """Live data for one county from the background-refreshed cache. Falls
    back to a direct (blocking) fetch only if the cache hasn't populated yet
    (e.g. right after a cold start) or genuinely never had this county."""
    with _live_cache_lock:
        cached = _live_cache.get(county)
    if cached is not None:
        return cached
    art = _artifacts()
    return data_sources.get_live_county_inputs(county, COUNTY_COORDS, art["county_defaults"], full_history=True)


# ── Artifacts (loaded once at process start) ──────────────────────────────────
@lru_cache(maxsize=1)
def _artifacts():
    bundle = data_access.load_model_bundle(str(REPO_ROOT / "model" / "best_model.pkl"))
    nowcast_model = bundle["nowcast_model"]
    outlook_bundle = bundle["outlook_model"]
    outlook_model = load_outlook_bundle(outlook_bundle)
    meta, counties, fstats, county_defaults, county_climate_percentiles, hist_df, monthly_df = data_access.load_artifacts(
        str(REPO_ROOT / "model")
    )
    return {
        "nowcast_model": nowcast_model,
        "outlook_bundle": outlook_bundle,
        "outlook_model": outlook_model,
        "meta": meta,
        "counties": counties,
        "fstats": fstats,
        "county_defaults": county_defaults,
        "county_climate_percentiles": county_climate_percentiles,
        "hist_df": hist_df,
        "monthly_df": monthly_df,
    }


def risk_tier(p: float):
    if p < 0.25:
        return "Low"
    elif p < 0.50:
        return "Moderate"
    elif p < 0.75:
        return "High"
    return "Critical"


def historical_risk_tier(rate: float):
    if rate >= 0.12:
        return "Critical"
    elif rate >= 0.06:
        return "High"
    elif rate >= 0.03:
        return "Moderate"
    return "Low"


def _fill_lag_defaults(raw: dict, county: str, county_defaults: dict) -> dict:
    cd = county_defaults.get(county, {})
    r = dict(raw)
    for f in LAG_FEATURES:
        r.setdefault(f, cd.get(f, 0.0))
    return r


def run_prediction(raw: dict, county: str) -> float:
    art = _artifacts()
    r = _fill_lag_defaults(raw, county, art["county_defaults"])
    r = engineer_features_row(r)
    X = np.array([[r[f] for f in art["meta"]["features"]]])
    return float(art["nowcast_model"].predict_proba(X)[0, 1])


def run_outlook(county: str) -> Optional[float]:
    art = _artifacts()
    g = art["monthly_df"][art["monthly_df"]["county"] == county].sort_values(["year", "month"])
    if len(g) < SEQ_LEN:
        return None
    cd = art["county_defaults"].get(county, {})
    seq_window = [
        {**{f: cd.get(f, 0.0) for f in SEQ_FEATURES}, "flood": float(row.flood)}
        for row in g.tail(SEQ_LEN).itertuples()
    ]
    static_row = {f: cd.get(f, 0.0) for f in STATIC_FEATURES_SEQ}
    return predict_outlook(art["outlook_bundle"], art["outlook_model"], seq_window, static_row)


def run_outlook_multistep(county: str, n_months: int) -> Optional[list]:
    art = _artifacts()
    g = art["monthly_df"][art["monthly_df"]["county"] == county].sort_values(["year", "month"])
    if len(g) < SEQ_LEN:
        return None
    cd = art["county_defaults"].get(county, {})
    seq_window = [
        {**{f: cd.get(f, 0.0) for f in SEQ_FEATURES}, "flood": float(row.flood)}
        for row in g.tail(SEQ_LEN).itertuples()
    ]
    static_row = {f: cd.get(f, 0.0) for f in STATIC_FEATURES_SEQ}
    return predict_outlook_multistep(art["outlook_bundle"], art["outlook_model"], seq_window, static_row, n_months)


# ── Schemas ────────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    county: str
    month: int
    rainfall_mm: float
    soil_moisture_mm: float
    max_temperature_celsius: float
    min_temperature_celsius: float
    vapor_pressure_deficit_kPa: float
    wetland_fraction: float
    elevation_m: float
    slope_deg: float
    ndvi: float
    flood_prev_month: int


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/model-info")
def model_info():
    art = _artifacts()
    return art["meta"]


@app.get("/api/counties")
def list_counties():
    art = _artifacts()
    hr = art["hist_df"].set_index("county")["flood_rate"].to_dict()
    ev = art["hist_df"].set_index("county")["flood_events"].to_dict()
    out = []
    for c in art["counties"]:
        if c not in COUNTY_COORDS:
            continue
        lat, lon = COUNTY_COORDS[c]
        rate = hr.get(c, 0.0)
        out.append({
            "county": c, "lat": lat, "lon": lon,
            "flood_rate": rate,
            "flood_events": int(ev.get(c, 0)),
            "risk_tier": historical_risk_tier(rate),
        })
    return out


# Declared before the bare /api/county/{county:path} route below: the path
# convertor is greedy, so that route would otherwise swallow this suffix
# and report the county as unknown.
@app.get("/api/county/{county:path}/climate-percentiles")
def county_climate_percentiles(county: str):
    """Historical (2011-2025) min/p10/p25/median/p75/p90/max for this county's
    five live climate inputs, precomputed from the training dataset (see
    model/county_climate_percentiles.json) — lets the UI show today's live
    reading against the county's own historical distribution, not just a
    single median point, using the same percentile-anomaly language (e.g.
    "90th percentile rainfall") humanitarian early-warning bulletins use.
    """
    art = _artifacts()
    if county not in art["counties"]:
        raise HTTPException(404, f"Unknown county: {county}")
    return art["county_climate_percentiles"].get(county, {})


@app.get("/api/county/{county:path}")
def county_detail(county: str):
    art = _artifacts()
    if county not in art["counties"]:
        raise HTTPException(404, f"Unknown county: {county}")
    cd = art["county_defaults"].get(county, {})
    crow = art["hist_df"][art["hist_df"]["county"] == county]
    rank_df = art["hist_df"].sort_values("flood_rate", ascending=False).reset_index(drop=True)
    rank = int(rank_df[rank_df["county"] == county].index[0]) + 1 if county in rank_df["county"].values else None
    lat, lon = COUNTY_COORDS.get(county, (None, None))
    return {
        "county": county, "lat": lat, "lon": lon,
        "defaults": cd,
        "flood_rate": float(crow["flood_rate"].iloc[0]) if not crow.empty else 0.0,
        "flood_events": int(crow["flood_events"].iloc[0]) if not crow.empty else 0,
        "rank": rank,
        "n_counties": len(art["counties"]),
    }


@app.get("/api/live/{county:path}")
def live_data(county: str):
    art = _artifacts()
    if county not in art["counties"]:
        raise HTTPException(404, f"Unknown county: {county}")
    return get_cached_live(county)


@app.get("/api/weather/{county:path}")
def weather(county: str):
    """Real-time 'right now' conditions + short rain outlook (OpenWeatherMap).
    Separate from /api/live — this never feeds the model, it's a human-facing
    display. 503s cleanly if no API key is configured rather than pretending
    to have data."""
    if county not in COUNTY_COORDS:
        raise HTTPException(404, f"Unknown county: {county}")
    if not weather_live.available():
        raise HTTPException(503, "OPENWEATHER_API_KEY not configured on the server")
    lat, lon = COUNTY_COORDS[county]
    try:
        return weather_live.get_live_weather(lat, lon)
    except Exception as exc:
        raise HTTPException(502, f"Weather fetch failed: {exc}")


@app.get("/api/weather-grid")
def weather_grid():
    """All 79 counties' current conditions from the background-refreshed
    cache — instant, never triggers a live OpenWeatherMap call itself."""
    art = _artifacts()
    with _weather_grid_lock:
        snapshot = dict(_weather_grid_cache)
    out = []
    for c in art["counties"]:
        w = snapshot.get(c)
        if not w:
            continue
        lat, lon = COUNTY_COORDS[c]
        out.append({"county": c, "lat": lat, "lon": lon, **w})
    return out


@app.get("/api/weather-ticker")
def weather_ticker():
    """Live Updates terminal feed, OpenWeatherMap-backed — a rolling history
    per county built from the same background refresh as /api/weather-grid.
    Used instead of Open-Meteo for the ticker since OpenWeatherMap's quota is
    tied to this app's own API key, not shared across every caller on
    whatever network this server happens to run on."""
    art = _artifacts()
    with _weather_grid_lock:
        snapshot = {c: list(v) for c, v in _weather_history_cache.items()}
    out = []
    for c in art["counties"]:
        hist = snapshot.get(c)
        if not hist:
            continue
        lat, lon = COUNTY_COORDS[c]
        out.append({"county": c, "lat": lat, "lon": lon, "history": hist})
    return out


@app.get("/api/live-ticker")
def live_ticker():
    """Live Updates terminal feed — Open-Meteo current-conditions history for
    all 79 counties, seeded with real past hours at server startup and
    appended to every 5 minutes. Server-side and shared across every client,
    so no browser ever calls Open-Meteo directly (that per-client pattern is
    what tripped its rate limit before)."""
    art = _artifacts()
    with _om_ticker_lock:
        snapshot = {c: list(v) for c, v in _om_ticker_cache.items()}
    out = []
    for c in art["counties"]:
        hist = snapshot.get(c)
        if not hist:
            continue
        lat, lon = COUNTY_COORDS[c]
        out.append({"county": c, "lat": lat, "lon": lon, "history": hist})
    return out


@app.post("/api/predict")
def predict(req: PredictRequest):
    art = _artifacts()
    if req.county not in art["counties"]:
        raise HTTPException(404, f"Unknown county: {req.county}")
    raw = req.model_dump()
    raw.pop("county")
    prob = run_prediction(raw, req.county)
    outlook_prob = run_outlook(req.county)
    return {
        "probability": prob,
        "risk_tier": risk_tier(prob),
        "threshold": art["meta"]["threshold"],
        "above_threshold": prob >= art["meta"]["threshold"],
        "outlook_probability": outlook_prob,
        "outlook_risk_tier": risk_tier(outlook_prob) if outlook_prob is not None else None,
        "model_name": art["meta"]["best_model_name"],
        "outlook_model_name": art["meta"].get("outlook_model_name"),
    }


@app.get("/api/risk-trend")
def risk_trend():
    """Live flood-risk trend for all 79 counties — same deployed nowcast
    model as /api/scan, but pre-computed server-side on a timer and shared
    across every client, so a new visitor sees hours of real trend
    immediately instead of an empty chart that only fills in per-session."""
    art = _artifacts()
    with _risk_trend_lock:
        snapshot = {c: list(v) for c, v in _risk_trend_cache.items()}
    out = []
    for c in art["counties"]:
        hist = snapshot.get(c)
        if not hist:
            continue
        out.append({"county": c, "history": hist})
    return out


@app.get("/api/scan")
def scan(
    month: int,
    use_live: bool = False,
    rainfall_mm: float = 80.0,
    soil_moisture_mm: float = 25.0,
    max_temperature_celsius: float = 35.0,
    min_temperature_celsius: float = 21.0,
    vapor_pressure_deficit_kPa: float = 2.3,
    flood_prev_month: int = 0,
):
    art = _artifacts()

    live_by_county = {}
    if use_live:
        # Reads the background-refreshed cache — never hits Open-Meteo
        # directly from a request, so this can't be slow or rate-limited.
        for c in art["counties"]:
            live_by_county[c] = get_cached_live(c)

    results = []
    for c in art["counties"]:
        cd = art["county_defaults"].get(c, {})
        if use_live:
            live = live_by_county[c]
            inputs = dict(live["inputs"])
            inputs["month"] = month
        else:
            inputs = {
                "rainfall_mm": rainfall_mm,
                "soil_moisture_mm": soil_moisture_mm,
                "max_temperature_celsius": max_temperature_celsius,
                "min_temperature_celsius": min_temperature_celsius,
                "vapor_pressure_deficit_kPa": vapor_pressure_deficit_kPa,
                "wetland_fraction": float(min(cd.get("wetland_fraction", 0.10), 0.92)),
                "elevation_m": float(min(max(cd.get("elevation_m", 513.0), 392.0), 1145.0)),
                "slope_deg": float(min(max(cd.get("slope_deg", 1.7), 0.9), 8.3)),
                "ndvi": float(min(max(cd.get("ndvi", 0.55), 0.19), 0.85)),
                "flood_prev_month": flood_prev_month,
                "month": month,
            }
        p = run_prediction(inputs, c)
        lat, lon = COUNTY_COORDS.get(c, (None, None))
        results.append({
            "county": c, "lat": lat, "lon": lon,
            "probability": p, "risk_tier": risk_tier(p),
        })
    return sorted(results, key=lambda r: r["probability"], reverse=True)


@app.get("/api/outlook/{county:path}/multistep")
def outlook_multistep(county: str, months: int = 6):
    """Recursive GRU rollout. `months` can go out to 48 (4 calendar years
    ahead) — uncertainty compounds heavily that far out (each step's forecast
    feeds the next), which is why the response includes a 0-1 `confidence`
    that decays with distance: treat months 24+ as an indicative trend only,
    never as a precise monthly forecast."""
    art = _artifacts()
    if county not in art["counties"]:
        raise HTTPException(404, f"Unknown county: {county}")
    months = max(1, min(months, 48))
    probs = run_outlook_multistep(county, months)
    if probs is None:
        raise HTTPException(422, f"Not enough recorded history for {county} to run the outlook model.")
    today = dt.date.today()
    out = []
    for i, p in enumerate(probs, start=1):
        # month index rolls forward from the current calendar month
        m = (today.month - 1 + i) % 12 + 1
        y = today.year + (today.month - 1 + i) // 12
        # Confidence decays with each recursive step — halves roughly every
        # 12 months as compounded forecast error accumulates. This is a
        # heuristic communication device, not a calibrated statistical bound.
        confidence = 0.5 ** (i / 12)
        out.append({"month": m, "year": y, "probability": p, "risk_tier": risk_tier(p), "confidence": round(confidence, 3)})
    return out


NOWCAST_ADVICE = {
    "Low": "No immediate action required. Maintain routine monitoring.",
    "Moderate": "Alert county officials. Pre-position emergency supplies and review evacuation routes.",
    "High": "Issue a public warning. Activate emergency response teams and prepare for evacuations.",
    "Critical": "Immediate action. Initiate evacuations and notify national emergency management.",
}
OUTLOOK_ADVICE = {
    "Low": "No elevated risk expected next month based on recent history — routine monitoring is sufficient.",
    "Moderate": "Risk is above baseline for next month. Begin reviewing preparedness plans and supply positions.",
    "High": "Historical pattern points to elevated risk next month. Start preparedness planning and resource staging now.",
    "Critical": "Strong signal of flood risk next month. Begin pre-positioning resources and briefing response teams ahead of time.",
}


@app.get("/api/report/{county:path}")
def report(county: str, month: Optional[int] = None, year: Optional[int] = None,
           range_from: Optional[int] = None, range_to: Optional[int] = None):
    art = _artifacts()
    if county not in art["counties"]:
        raise HTTPException(404, f"Unknown county: {county}")
    cd = art["county_defaults"].get(county, {})
    today = dt.date.today()
    report_month = month or today.month
    report_year = year or today.year
    range_from = range_from or 4
    range_to = range_to or 12

    live = get_cached_live(county)
    inputs = {
        "rainfall_mm": cd.get("rainfall_mm", 80.0), "soil_moisture_mm": cd.get("soil_moisture_mm", 25.0),
        "max_temperature_celsius": cd.get("max_temperature_celsius", 35.0),
        "min_temperature_celsius": cd.get("min_temperature_celsius", 21.0),
        "vapor_pressure_deficit_kPa": cd.get("vapor_pressure_deficit_kPa", 2.3),
        "wetland_fraction": cd.get("wetland_fraction", 0.1), "elevation_m": cd.get("elevation_m", 500.0),
        "slope_deg": cd.get("slope_deg", 1.5), "ndvi": cd.get("ndvi", 0.5),
        "flood_prev_month": cd.get("flood_prev_month", 0),
    }
    if live.get("live_data_available"):
        inputs.update(live["inputs"])
    inputs["month"] = report_month
    p_now = run_prediction(inputs, county)
    p_out = run_outlook(county)

    # Months needed (from real today) to cover Jan-Dec of report_year.
    months_to_cover = max(1, (report_year - today.year) * 12 + (12 - today.month) + 1)
    months_to_cover = min(months_to_cover, 48)
    multistep_raw = run_outlook_multistep(county, months_to_cover) or []
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    multistep = []
    for i, p in enumerate(multistep_raw, start=1):
        m = (today.month - 1 + i) % 12 + 1
        y = today.year + (today.month - 1 + i) // 12
        if y == report_year:
            multistep.append({"label": f"{month_names[m - 1]} {y}", "probability": p})

    hist = art["hist_df"]
    crow = hist[hist["county"] == county]
    rank_df = hist.sort_values("flood_rate", ascending=False).reset_index(drop=True)
    rank = int(rank_df[rank_df["county"] == county].index[0]) + 1 if county in rank_df["county"].values else None

    range_months = (list(range(range_from, range_to + 1)) if range_from <= range_to
                    else list(range(range_from, 13)) + list(range(1, range_to + 1)))
    seasonal = [
        {"month": month_names[m - 1], "probability": run_prediction({**inputs, "month": m}, county)}
        for m in range_months
    ]

    pdf_bytes = build_county_report(
        county=county,
        generated_at=dt.datetime.utcnow(),
        for_period=f"{month_names[report_month - 1]} {report_year}",
        nowcast={"probability": p_now, "risk_tier": risk_tier(p_now)},
        outlook={"probability": p_out, "risk_tier": risk_tier(p_out)} if p_out is not None else None,
        historical={
            "flood_rate": float(crow["flood_rate"].iloc[0]) if not crow.empty else 0.0,
            "flood_events": int(crow["flood_events"].iloc[0]) if not crow.empty else 0,
            "rank": rank, "n_counties": len(art["counties"]),
        },
        seasonal=seasonal,
        nowcast_advice=NOWCAST_ADVICE[risk_tier(p_now)],
        outlook_advice=OUTLOOK_ADVICE[risk_tier(p_out)] if p_out is not None else None,
        model_name=art["meta"]["best_model_name"],
        outlook_model_name=art["meta"].get("outlook_model_name", "GRU (Outlook)"),
        climate_inputs=inputs,
        field_source=live.get("field_source", {}),
        multistep=multistep,
        model_metrics=art["meta"]["test_metrics"],
    )
    return Response(
        content=pdf_bytes, media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{county}_flood_report_{today.isoformat()}.pdf"'},
    )


@app.get("/api/overview")
def overview():
    art = _artifacts()
    hist = art["hist_df"]
    tiers = {"Critical": 0, "High": 0, "Moderate": 0, "Low": 0}
    for rate in hist["flood_rate"]:
        tiers[historical_risk_tier(rate)] += 1

    watchlist = (
        hist.sort_values("flood_rate", ascending=False).head(10)
        [["county", "flood_rate", "flood_events"]]
        .assign(risk_tier=lambda d: d["flood_rate"].map(historical_risk_tier))
        .to_dict(orient="records")
    )

    calendar = (
        art["monthly_df"].groupby(["year", "month"])["flood"].mean().reset_index()
        .to_dict(orient="records")
    )

    return {
        "total_flood_events": int(hist["flood_events"].sum()),
        "national_mean_rate": float(hist["flood_rate"].mean()),
        "n_counties": len(art["counties"]),
        "risk_tier_counts": tiers,
        "highest_risk_county": hist.loc[hist["flood_rate"].idxmax(), "county"],
        "watchlist": watchlist,
        "calendar": calendar,
        "period": {
            "year_min": int(art["monthly_df"]["year"].min()),
            "year_max": int(art["monthly_df"]["year"].max()),
        },
    }


DATA_QUALITY_NOTE = (
    "The satellite water-extent signal underlying flood labels has a growing rate of missing values for "
    "2022-2025 (roughly 4.6% missing in 2022 rising to 7.9% by 2025, vs. 0% missing every year 2011-2021), "
    "while rainfall shows no corresponding anomaly — consistent with a processing-lag gap in the satellite "
    "source rather than a real decline in flooding. Historical rates for 2022-2025 likely undercount real "
    "events; predictions themselves are always forward-looking from the current date, independent of this."
)


@app.get("/api/report-national")
def report_national():
    art = _artifacts()
    hist = art["hist_df"]
    tiers = {"Critical": 0, "High": 0, "Moderate": 0, "Low": 0}
    for rate in hist["flood_rate"]:
        tiers[historical_risk_tier(rate)] += 1
    watchlist = (
        hist.sort_values("flood_rate", ascending=False).head(15)
        [["county", "flood_rate", "flood_events"]]
        .assign(risk_tier=lambda d: d["flood_rate"].map(historical_risk_tier))
        .to_dict(orient="records")
    )
    calendar = art["monthly_df"].groupby(["year", "month"])["flood"].mean().reset_index().to_dict(orient="records")

    pdf_bytes = build_national_report(
        generated_at=dt.datetime.utcnow(),
        total_flood_events=int(hist["flood_events"].sum()),
        national_mean_rate=float(hist["flood_rate"].mean()),
        n_counties=len(art["counties"]),
        risk_tier_counts=tiers,
        highest_risk_county=hist.loc[hist["flood_rate"].idxmax(), "county"],
        watchlist=watchlist,
        calendar=calendar,
        model_name=art["meta"]["best_model_name"],
        outlook_model_name=art["meta"].get("outlook_model_name", "GRU (Outlook)"),
        model_metrics=art["meta"]["test_metrics"],
        model_selection_criterion=art["meta"]["model_selection_criterion"],
        data_quality_note=DATA_QUALITY_NOTE,
    )
    today = dt.date.today()
    return Response(
        content=pdf_bytes, media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="SSD_Flood_EWS_National_Summary_{today.isoformat()}.pdf"'},
    )


@app.get("/api/historical")
def historical(counties: str, year_min: int, year_max: int, months: str):
    art = _artifacts()
    county_list = counties.split(",") if counties else []
    month_list = [int(m) for m in months.split(",")] if months else list(range(1, 13))
    filtered = data_access.filter_monthly(art["monthly_df"], county_list, (year_min, year_max), month_list)
    return {
        "records": filtered.to_dict(orient="records"),
        "national": data_access.filter_monthly(
            art["monthly_df"], art["counties"], (year_min, year_max), month_list
        ).groupby("year")["flood"].mean().reset_index().to_dict(orient="records"),
    }


# ── Serve the built React frontend, if present (single-container deploy) ─────
FRONTEND_DIST = REPO_ROOT / "frontend" / "dist"
if FRONTEND_DIST.exists():
    # Real build output (/assets/..., /geo/...) is served from disk.
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="assets")
    for _extra in ("geo",):
        _dir = FRONTEND_DIST / _extra
        if _dir.is_dir():
            app.mount(f"/{_extra}", StaticFiles(directory=str(_dir)), name=_extra)

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa_fallback(full_path: str):
        """Hand any non-API path to the React app.

        Routing is client-side, so paths like /alerts or /county/Malakal exist
        only once index.html has loaded — there is no file behind them. A plain
        StaticFiles mount 404s on those, which breaks every deep link and every
        page refresh away from the root. Serve the real file when one exists,
        and index.html otherwise so the router can take over.

        /api/* is excluded: an unknown API path should stay a 404 rather than
        silently return HTML, which is far harder to debug from the client.
        """
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not found")

        candidate = (FRONTEND_DIST / full_path).resolve()
        # Only serve inside the build directory — `full_path` is attacker
        # controlled, and without this check `../` escapes it.
        if (
            full_path
            and FRONTEND_DIST.resolve() in candidate.parents
            and candidate.is_file()
        ):
            return FileResponse(str(candidate))

        return FileResponse(str(FRONTEND_DIST / "index.html"))
