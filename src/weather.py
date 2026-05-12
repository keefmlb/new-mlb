"""Weather data for MLB games.

Two paths:
  - Historical (for backtest features): Open-Meteo Archive API, hourly observed values.
  - Forecast (for upcoming games): Open-Meteo Forecast API.

Both are free and need no API key. We always pull the hour closest to first pitch
in the park's local time (the API accepts ISO timestamps in UTC).

For domes / closed retractable roofs, we still pull the data but the model
will down-weight wind/temp via the `roof` flag from parks.py.
"""
from __future__ import annotations
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
import requests

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "weather"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
FORECAST = "https://api.open-meteo.com/v1/forecast"


def _cache_key(lat: float, lon: float, when: datetime) -> Path:
    return CACHE_DIR / f"w_{lat:.3f}_{lon:.3f}_{when.strftime('%Y%m%dT%H')}.json"


def _fetch(url: str, params: dict) -> dict:
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def get_weather(lat: float, lon: float, when: datetime, retries: int = 2) -> dict:
    """Return weather at (lat, lon) at the hour containing `when` (UTC).

    Auto-routes to forecast vs archive based on whether `when` is in the past.
    Returned dict keys: temp_f, wind_mph, wind_dir_deg, humidity, precip_in,
                        pressure_inhg, cloud_cover_pct.
    """
    when = when.astimezone(timezone.utc)
    cache = _cache_key(lat, lon, when)
    if cache.exists():
        try:
            return json.loads(cache.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass

    now = datetime.now(timezone.utc)
    is_future = when > now

    target_iso = when.strftime("%Y-%m-%dT%H:00")
    target_date = when.strftime("%Y-%m-%d")

    common = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join([
            "temperature_2m", "relative_humidity_2m", "dew_point_2m",
            "precipitation",
            "wind_speed_10m", "wind_direction_10m", "surface_pressure",
            "cloud_cover",
        ]),
        "timezone": "UTC",
        "wind_speed_unit": "mph",
        "temperature_unit": "fahrenheit",
        "precipitation_unit": "inch",
    }

    if is_future:
        params = {**common, "start_date": target_date, "end_date": target_date}
        url = FORECAST
    else:
        params = {**common, "start_date": target_date, "end_date": target_date}
        url = ARCHIVE

    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            data = _fetch(url, params)
            break
        except requests.RequestException as e:
            last_err = e
            time.sleep(0.5 * (attempt + 1))
    else:
        # Fall back to neutral weather rather than failing the whole pipeline.
        return _neutral_weather()

    hourly = data.get("hourly") or {}
    times = hourly.get("time") or []
    if not times:
        return _neutral_weather()

    # Find the index for our target hour. Open-Meteo returns times in UTC ISO without tz suffix.
    idx = None
    for i, t in enumerate(times):
        if t.startswith(target_iso):
            idx = i
            break
    if idx is None:
        idx = min(range(len(times)), key=lambda i: abs(_diff_hours(times[i], target_iso)))

    out = {
        "temp_f":          _val(hourly.get("temperature_2m"),     idx, default=70.0),
        "humidity":        _val(hourly.get("relative_humidity_2m"), idx, default=50.0),
        "dew_point_f":     _val(hourly.get("dew_point_2m"),       idx, default=55.0),
        "precip_in":       _val(hourly.get("precipitation"),       idx, default=0.0),
        "wind_mph":        _val(hourly.get("wind_speed_10m"),      idx, default=5.0),
        "wind_dir_deg":    _val(hourly.get("wind_direction_10m"),  idx, default=0.0),
        "pressure_hpa":    _val(hourly.get("surface_pressure"),    idx, default=1013.0),
        "cloud_cover_pct": _val(hourly.get("cloud_cover"),         idx, default=50.0),
    }
    cache.write_text(json.dumps(out), encoding="utf-8")
    return out


def _val(arr, idx, default):
    if arr is None or idx is None or idx >= len(arr) or arr[idx] is None:
        return default
    return float(arr[idx])


def _diff_hours(a: str, b: str) -> float:
    fa = datetime.fromisoformat(a).replace(tzinfo=timezone.utc)
    fb = datetime.fromisoformat(b).replace(tzinfo=timezone.utc)
    return (fa - fb).total_seconds() / 3600.0


def _neutral_weather() -> dict:
    return {
        "temp_f": 70.0, "humidity": 50.0, "precip_in": 0.0,
        "wind_mph": 5.0, "wind_dir_deg": 0.0, "pressure_hpa": 1013.0,
        "cloud_cover_pct": 50.0,
    }


def wind_component_to_cf(wind_mph: float, wind_dir_deg: float, cf_bearing_deg: float) -> float:
    """Wind component blowing toward CF (positive = out toward CF, negative = blowing in).

    `wind_dir_deg` is the direction the wind is *coming from* (meteorological convention).
    We convert to the direction it's *going to* (+ 180), then take the cosine of the
    angle between that vector and the CF bearing (home plate -> CF).
    """
    going_to = (wind_dir_deg + 180.0) % 360.0
    angle = math.radians(going_to - cf_bearing_deg)
    return wind_mph * math.cos(angle)
