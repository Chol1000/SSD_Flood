"""
weather_live.py — Real-time "right now" weather via OpenWeatherMap.

Distinct in purpose from data_sources.py: that module feeds the flood
model's climate inputs (month-to-date aggregates, lag/rolling features) from
Open-Meteo. This module is purely for the human-facing "what's the weather
doing in this county right now / over the next few days" display — current
conditions plus a short rain-probability forecast. It never feeds the model.

Requires OPENWEATHER_API_KEY (see .env, gitignored — never hardcode the key
in a committed file).
"""

import os
import time

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY")
CURRENT_URL = "https://api.openweathermap.org/data/2.5/weather"
FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"
REQUEST_TIMEOUT = 8
CACHE_TTL_SECONDS = 600  # 10 min — this is a "right now" display, not a model input

_cache: dict[tuple, tuple[float, dict]] = {}


def available() -> bool:
    return bool(OPENWEATHER_API_KEY)


def _cached_get(url: str, params: dict, cache_key: tuple) -> dict:
    cached = _cache.get(cache_key)
    if cached and (time.monotonic() - cached[0]) < CACHE_TTL_SECONDS:
        return cached[1]
    resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    _cache[cache_key] = (time.monotonic(), data)
    return data


def get_current_only(lat: float, lon: float) -> dict:
    """Lighter single-call version for the national grid view (79 counties
    at once) — skips the forecast call to keep quota usage low."""
    if not OPENWEATHER_API_KEY:
        raise RuntimeError("OPENWEATHER_API_KEY not configured")
    cur = _cached_get(CURRENT_URL, {
        "lat": lat, "lon": lon, "appid": OPENWEATHER_API_KEY, "units": "metric",
    }, ("current", round(lat, 3), round(lon, 3)))
    return {
        "temp_c": cur["main"]["temp"],
        "humidity_pct": cur["main"]["humidity"],
        "wind_speed_ms": cur["wind"]["speed"],
        "description": cur["weather"][0]["description"],
        "icon": cur["weather"][0]["icon"],
        "observed_at": cur["dt"],
    }


def _day_key(unix_s: int, tz_offset_s: int) -> str:
    import datetime as _dt
    return _dt.datetime.utcfromtimestamp(unix_s + tz_offset_s).strftime("%Y-%m-%d")


def _day_label(unix_s: int, tz_offset_s: int) -> str:
    import datetime as _dt
    return _dt.datetime.utcfromtimestamp(unix_s + tz_offset_s).strftime("%a %d")


def get_live_weather(lat: float, lon: float) -> dict:
    """Current conditions + full 5-day/3-hour outlook for one point, plus a
    server-side daily rollup (min/max/dominant icon per day) so the frontend
    can render a proper "today + next several days" view instead of a flat
    48h window. Raises if the API key is missing or the request fails —
    callers should catch and degrade gracefully (this is a bonus display,
    not core to the flood model).
    """
    if not OPENWEATHER_API_KEY:
        raise RuntimeError("OPENWEATHER_API_KEY not configured")

    cur = _cached_get(CURRENT_URL, {
        "lat": lat, "lon": lon, "appid": OPENWEATHER_API_KEY, "units": "metric",
    }, ("current", round(lat, 3), round(lon, 3)))

    fc = _cached_get(FORECAST_URL, {
        "lat": lat, "lon": lon, "appid": OPENWEATHER_API_KEY, "units": "metric", "cnt": 40,
    }, ("forecast", round(lat, 3), round(lon, 3)))

    tz_offset_s = cur.get("timezone", 0)

    forecast = [
        {
            "time": item["dt_txt"],
            "dt": item["dt"],
            "day_key": _day_key(item["dt"], tz_offset_s),
            "temp_c": item["main"]["temp"],
            "rain_probability": item.get("pop", 0.0),
            "rain_mm_3h": item.get("rain", {}).get("3h", 0.0),
            "description": item["weather"][0]["description"],
            "icon": item["weather"][0]["icon"],
        }
        for item in fc.get("list", [])
    ]

    daily_buckets: dict[str, list[dict]] = {}
    for item in forecast:
        daily_buckets.setdefault(item["day_key"], []).append(item)

    daily = []
    for day_key, items in daily_buckets.items():
        temps = [it["temp_c"] for it in items]
        # prefer the ~midday reading for the representative icon/description
        rep = min(items, key=lambda it: abs((it["dt"] + tz_offset_s) % 86400 - 12 * 3600))
        daily.append({
            "day_key": day_key,
            "label": _day_label(items[0]["dt"], tz_offset_s),
            "temp_min": min(temps),
            "temp_max": max(temps),
            "icon": rep["icon"],
            "description": rep["description"],
            "rain_probability_max": max(it["rain_probability"] for it in items),
        })

    return {
        "temp_c": cur["main"]["temp"],
        "feels_like_c": cur["main"]["feels_like"],
        "humidity_pct": cur["main"]["humidity"],
        "pressure_hpa": cur["main"]["pressure"],
        "wind_speed_ms": cur["wind"]["speed"],
        "description": cur["weather"][0]["description"],
        "icon": cur["weather"][0]["icon"],
        "observed_at": cur["dt"],
        "sunrise": cur.get("sys", {}).get("sunrise"),
        "sunset": cur.get("sys", {}).get("sunset"),
        "timezone_offset_s": tz_offset_s,
        "forecast": forecast,
        "daily": daily,
        "source": "OpenWeatherMap",
    }
