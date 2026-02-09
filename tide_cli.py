#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

NOAA_API = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
NOAA_STATIONS_API = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.json?type=tidepredictions"
NOAA_STATION_DETAIL_TEMPLATE = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations/{station_id}.json"
ZIP_API_TEMPLATE = "https://api.zippopotam.us/us/{zip_code}"
OPEN_METEO_FORECAST_API = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_MARINE_API = "https://marine-api.open-meteo.com/v1/marine"


@dataclass
class Prediction:
    t: datetime
    v: float


@dataclass
class TideEvent:
    t: datetime
    v: float
    kind: str


@dataclass
class Observation:
    t: datetime
    v: float


@dataclass
class Station:
    station_id: str
    name: str
    lat: float
    lon: float
    state: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show current tide schedule and a simple CLI visualization from NOAA CO-OPS data.",
    )
    parser.add_argument(
        "--station",
        default=os.getenv("TIDE_STATION"),
        help="NOAA station id (example: 9414290 for San Francisco). Can also be set via TIDE_STATION.",
    )
    parser.add_argument(
        "--zip",
        dest="zip_code",
        help="US ZIP code. If set, finds nearest NOAA tide station automatically.",
    )
    parser.add_argument(
        "--units",
        choices=["english", "metric"],
        default="english",
        help="Display units for tide height.",
    )
    parser.add_argument(
        "--events",
        type=int,
        default=4,
        help="How many upcoming high/low events to show.",
    )
    parser.add_argument(
        "--weather-source",
        choices=["noaa", "open-meteo"],
        default="noaa",
        help="Source for air/water temperatures.",
    )
    return parser.parse_args()


def get_json_url(url: str) -> dict:
    req = urlopen(url, timeout=20)
    with req:
        raw = req.read().decode("utf-8")
    return json.loads(raw)


def get_json(params: dict[str, str]) -> dict:
    query = urlencode(params)
    url = f"{NOAA_API}?{query}"
    return get_json_url(url)


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r_miles = 3958.7613
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r_miles * c


def resolve_zip_to_lat_lon(zip_code: str) -> tuple[float, float, str, str]:
    payload = get_json_url(ZIP_API_TEMPLATE.format(zip_code=zip_code))
    places = payload.get("places", [])
    if not places:
        raise ValueError(f"No location found for ZIP {zip_code}")
    place = places[0]
    lat = float(place["latitude"])
    lon = float(place["longitude"])
    city = place.get("place name", "")
    state = place.get("state abbreviation", "")
    return lat, lon, city, state


def fetch_tide_stations() -> list[Station]:
    payload = get_json_url(NOAA_STATIONS_API)
    rows = payload.get("stations", [])
    out: list[Station] = []
    for row in rows:
        lat = row.get("lat")
        lon = row.get("lng")
        station_id = str(row.get("id", "")).strip()
        if not station_id or lat is None or lon is None:
            continue
        out.append(
            Station(
                station_id=station_id,
                name=row.get("name", "").strip(),
                lat=float(lat),
                lon=float(lon),
                state=row.get("state", "").strip(),
            )
        )
    return out


def fetch_station_by_id(station_id: str) -> Station | None:
    payload = get_json_url(NOAA_STATION_DETAIL_TEMPLATE.format(station_id=station_id))
    rows = payload.get("stations", [])
    if not rows:
        return None
    row = rows[0]
    lat = row.get("lat")
    lon = row.get("lng")
    if lat is None or lon is None:
        return None
    return Station(
        station_id=str(row.get("id", station_id)).strip(),
        name=row.get("name", "").strip(),
        lat=float(lat),
        lon=float(lon),
        state=row.get("state", "").strip(),
    )


def find_nearest_stations(zip_code: str) -> tuple[list[tuple[Station, float]], str, str]:
    lat, lon, city, state = resolve_zip_to_lat_lon(zip_code)
    stations = fetch_tide_stations()
    if not stations:
        raise ValueError("No NOAA tide stations available")
    ranked = sorted(
        ((station, haversine_miles(lat, lon, station.lat, station.lon)) for station in stations),
        key=lambda x: x[1],
    )
    return ranked, city, state


def fetch_predictions(station: str, units: str, start: datetime, end: datetime) -> list[Prediction]:
    params = {
        "product": "predictions",
        "application": "tide-cli",
        "begin_date": start.strftime("%Y%m%d"),
        "end_date": end.strftime("%Y%m%d"),
        "datum": "MLLW",
        "station": station,
        "time_zone": "lst_ldt",
        "units": units,
        "format": "json",
    }
    payload = get_json(params)
    if "error" in payload:
        raise ValueError(payload["error"].get("message", "Unknown NOAA API error"))
    series = payload.get("predictions", [])
    out: list[Prediction] = []
    for row in series:
        out.append(Prediction(datetime.strptime(row["t"], "%Y-%m-%d %H:%M"), float(row["v"])))
    return out


def fetch_hilo_events(station: str, units: str, start: datetime, end: datetime) -> list[TideEvent]:
    params = {
        "product": "predictions",
        "application": "tide-cli",
        "begin_date": start.strftime("%Y%m%d"),
        "end_date": end.strftime("%Y%m%d"),
        "datum": "MLLW",
        "station": station,
        "time_zone": "lst_ldt",
        "units": units,
        "interval": "hilo",
        "format": "json",
    }
    payload = get_json(params)
    if "error" in payload:
        raise ValueError(payload["error"].get("message", "Unknown NOAA API error"))
    series = payload.get("predictions", [])
    out: list[TideEvent] = []
    for row in series:
        out.append(TideEvent(datetime.strptime(row["t"], "%Y-%m-%d %H:%M"), float(row["v"]), row["type"]))
    return out


def fetch_observations(product: str, station: str, units: str, start: datetime, end: datetime) -> list[Observation]:
    params = {
        "product": product,
        "application": "tide-cli",
        "begin_date": start.strftime("%Y%m%d"),
        "end_date": end.strftime("%Y%m%d"),
        "station": station,
        "time_zone": "lst_ldt",
        "units": units,
        "format": "json",
    }
    payload = get_json(params)
    if "error" in payload:
        raise ValueError(payload["error"].get("message", "Unknown NOAA API error"))
    rows = payload.get("data", [])
    out: list[Observation] = []
    for row in rows:
        raw = str(row.get("v", "")).strip()
        if not raw:
            continue
        try:
            out.append(Observation(datetime.strptime(row["t"], "%Y-%m-%d %H:%M"), float(raw)))
        except (KeyError, ValueError):
            continue
    return out


def fetch_open_meteo_temperatures(
    lat: float, lon: float, units: str, start: datetime, end: datetime
) -> tuple[list[Observation], list[Observation]]:
    temp_unit = "fahrenheit" if units == "english" else "celsius"
    weather_params = urlencode(
        {
            "latitude": f"{lat:.6f}",
            "longitude": f"{lon:.6f}",
            "hourly": "temperature_2m",
            "temperature_unit": temp_unit,
            "timezone": "auto",
            "start_date": start.strftime("%Y-%m-%d"),
            "end_date": end.strftime("%Y-%m-%d"),
        }
    )
    marine_params = urlencode(
        {
            "latitude": f"{lat:.6f}",
            "longitude": f"{lon:.6f}",
            "hourly": "sea_surface_temperature",
            "temperature_unit": temp_unit,
            "timezone": "auto",
            "start_date": start.strftime("%Y-%m-%d"),
            "end_date": end.strftime("%Y-%m-%d"),
        }
    )

    weather_payload = get_json_url(f"{OPEN_METEO_FORECAST_API}?{weather_params}")
    marine_payload = get_json_url(f"{OPEN_METEO_MARINE_API}?{marine_params}")

    def parse_hourly(payload: dict, key: str) -> list[Observation]:
        hourly = payload.get("hourly", {})
        times = hourly.get("time", [])
        values = hourly.get(key, [])
        out: list[Observation] = []
        for ts, raw in zip(times, values):
            if raw is None:
                continue
            try:
                dt = datetime.fromisoformat(str(ts))
                if dt.tzinfo is not None:
                    dt = dt.astimezone().replace(tzinfo=None)
                out.append(Observation(dt, float(raw)))
            except ValueError:
                continue
        return out

    return parse_hourly(weather_payload, "temperature_2m"), parse_hourly(
        marine_payload, "sea_surface_temperature"
    )


def nearest_prediction(points: Iterable[Prediction], target: datetime) -> Prediction:
    return min(points, key=lambda p: abs((p.t - target).total_seconds()))


def nearest_observation(points: Iterable[Observation], target: datetime) -> Observation:
    return min(points, key=lambda p: abs((p.t - target).total_seconds()))


def format_height(value: float, units: str) -> str:
    suffix = "ft" if units == "english" else "m"
    return f"{value:.2f} {suffix}"


def format_temp(value: float, units: str) -> str:
    suffix = "F" if units == "english" else "C"
    return f"{value:.1f} {suffix}"


def sample_to_width(values: list[float], width: int) -> list[float]:
    if not values:
        return []
    if len(values) <= width:
        return values
    sampled: list[float] = []
    for i in range(width):
        pos = (i / max(width - 1, 1)) * (len(values) - 1)
        lo = int(math.floor(pos))
        hi = min(lo + 1, len(values) - 1)
        frac = pos - lo
        sampled.append(values[lo] + (values[hi] - values[lo]) * frac)
    return sampled


def render_ascii_graph(
    values: list[float],
    start: datetime,
    end: datetime,
    now: datetime,
    units: str,
    width: int = 72,
    height: int = 14,
) -> list[str]:
    points = sample_to_width(values, width)
    if not points:
        return []

    lo = min(points)
    hi = max(points)
    if math.isclose(lo, hi):
        hi = lo + 1.0
    span = hi - lo

    canvas = [[" " for _ in range(width)] for _ in range(height)]
    y_axis_width = 8

    # Curve points.
    y_positions: list[int] = []
    for x, value in enumerate(points):
        normalized = (value - lo) / span
        y = int(round((height - 1) - normalized * (height - 1)))
        y = max(0, min(height - 1, y))
        y_positions.append(y)
        canvas[y][x] = "*"

    # Connect sparse columns for a smoother line.
    for x in range(1, width):
        y0 = y_positions[x - 1]
        y1 = y_positions[x]
        if y0 == y1:
            continue
        step = 1 if y1 > y0 else -1
        for y in range(y0 + step, y1, step):
            if canvas[y][x] == " ":
                canvas[y][x] = "|"

    # Mark current time column.
    total_seconds = max((end - start).total_seconds(), 1.0)
    now_ratio = (now - start).total_seconds() / total_seconds
    now_x = int(round(now_ratio * (width - 1)))
    now_x = max(0, min(width - 1, now_x))
    for y in range(height):
        if canvas[y][now_x] == " ":
            canvas[y][now_x] = ":"
    canvas[y_positions[now_x]][now_x] = "O"

    # Add a horizontal midline for easier reading.
    mid_value = (lo + hi) / 2
    mid_y = int(round((height - 1) - ((mid_value - lo) / span) * (height - 1)))
    mid_y = max(0, min(height - 1, mid_y))
    for x in range(width):
        if canvas[mid_y][x] == " ":
            canvas[mid_y][x] = "-"

    # Build labeled output.
    lines: list[str] = []
    for y in range(height):
        value_at_row = hi - (y / max(height - 1, 1)) * span
        label = f"{value_at_row:>6.2f}"
        lines.append(f"{label} |{''.join(canvas[y])}")

    unit_suffix = "ft" if units == "english" else "m"
    axis = " " * y_axis_width + "+" + "-" * width
    time_labels = (
        " " * y_axis_width
        + f"{start.strftime('%H:%M')}"
        + " " * max(width - 15, 0)
        + f"{end.strftime('%H:%M')}"
    )
    marker = " " * y_axis_width + " " * now_x + "^ now"
    lines.append(axis)
    lines.append(time_labels)
    lines.append(marker + f" ({now.strftime('%H:%M')})")
    lines.append(f"{' ' * y_axis_width}Units: {unit_suffix}")
    return lines


def interpolate_series(points: list[Observation], width: int, start: datetime, end: datetime) -> list[float]:
    if not points:
        return []
    points = sorted(points, key=lambda p: p.t)
    out: list[float] = []
    idx = 0
    total_seconds = max((end - start).total_seconds(), 1.0)
    for i in range(width):
        target = start + timedelta(seconds=(i / max(width - 1, 1)) * total_seconds)
        while idx + 1 < len(points) and points[idx + 1].t <= target:
            idx += 1
        if idx + 1 >= len(points):
            out.append(points[idx].v)
            continue
        left = points[idx]
        right = points[idx + 1]
        gap = max((right.t - left.t).total_seconds(), 1.0)
        frac = max(0.0, min(1.0, (target - left.t).total_seconds() / gap))
        out.append(left.v + (right.v - left.v) * frac)
    return out


def render_temperature_graph(
    air: list[Observation],
    water: list[Observation],
    start: datetime,
    end: datetime,
    now: datetime,
    units: str,
    width: int = 72,
    height: int = 10,
) -> list[str]:
    if not air and not water:
        return []

    air_points = interpolate_series(air, width, start, end) if air else []
    water_points = interpolate_series(water, width, start, end) if water else []
    combined = (air_points or []) + (water_points or [])
    if not combined:
        return []

    lo = min(combined)
    hi = max(combined)
    if math.isclose(lo, hi):
        hi = lo + 1.0
    span = hi - lo

    canvas = [[" " for _ in range(width)] for _ in range(height)]
    y_axis_width = 8

    def y_for(value: float) -> int:
        y = int(round((height - 1) - ((value - lo) / span) * (height - 1)))
        return max(0, min(height - 1, y))

    if water_points:
        for x, value in enumerate(water_points):
            y = y_for(value)
            canvas[y][x] = "W" if canvas[y][x] == " " else "X"

    if air_points:
        for x, value in enumerate(air_points):
            y = y_for(value)
            canvas[y][x] = "A" if canvas[y][x] == " " else "X"

    total_seconds = max((end - start).total_seconds(), 1.0)
    now_ratio = (now - start).total_seconds() / total_seconds
    now_x = int(round(now_ratio * (width - 1)))
    now_x = max(0, min(width - 1, now_x))
    for y in range(height):
        if canvas[y][now_x] == " ":
            canvas[y][now_x] = ":"

    lines: list[str] = []
    for y in range(height):
        value_at_row = hi - (y / max(height - 1, 1)) * span
        label = f"{value_at_row:>6.1f}"
        lines.append(f"{label} |{''.join(canvas[y])}")

    unit_suffix = "F" if units == "english" else "C"
    axis = " " * y_axis_width + "+" + "-" * width
    time_labels = (
        " " * y_axis_width
        + f"{start.strftime('%H:%M')}"
        + " " * max(width - 15, 0)
        + f"{end.strftime('%H:%M')}"
    )
    marker = " " * y_axis_width + " " * now_x + "^ now"
    lines.append(axis)
    lines.append(time_labels)
    lines.append(marker + f" ({now.strftime('%H:%M')})")
    lines.append(f"{' ' * y_axis_width}Units: {unit_suffix} | A=Air W=Water X=Overlap")
    return lines


def main() -> int:
    args = parse_args()
    if not args.station and not args.zip_code:
        print("error: provide --station, --zip, or set TIDE_STATION", file=sys.stderr)
        return 2

    now = datetime.now()
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    day_end = day_start + timedelta(days=1)

    try:
        station_id = args.station
        station_label = ""
        station_obj: Station | None = None
        predictions: list[Prediction] = []
        events: list[TideEvent] = []

        if station_id:
            predictions = fetch_predictions(station_id, args.units, day_start, day_end)
            events = fetch_hilo_events(station_id, args.units, day_start, day_end + timedelta(days=1))
            station_obj = fetch_station_by_id(station_id)
        elif args.zip_code:
            ranked, city, state = find_nearest_stations(args.zip_code)
            errors: list[str] = []
            for candidate, miles in ranked[:40]:
                try:
                    predictions = fetch_predictions(candidate.station_id, args.units, day_start, day_end)
                    events = fetch_hilo_events(candidate.station_id, args.units, day_start, day_end + timedelta(days=1))
                    station_id = candidate.station_id
                    station_label = (
                        f"Resolved from ZIP {args.zip_code} ({city}, {state}) -> "
                        f"{candidate.station_id} ({candidate.name}, {candidate.state}) [{miles:.1f} mi]"
                    )
                    station_obj = candidate
                    break
                except ValueError as exc:
                    errors.append(str(exc))
            if not station_id:
                detail = errors[0] if errors else "No nearby station had tide predictions"
                raise ValueError(f"Could not resolve station for ZIP {args.zip_code}: {detail}")
        else:
            print("error: unable to determine station id", file=sys.stderr)
            return 2
    except (ValueError, HTTPError, URLError, TimeoutError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not predictions:
        print("error: no prediction data returned", file=sys.stderr)
        return 1

    current = nearest_prediction(predictions, now)
    today_values = [p.v for p in predictions if day_start <= p.t < day_end]
    upcoming = [e for e in events if e.t >= now][: max(args.events, 0)]
    air_temps: list[Observation] = []
    water_temps: list[Observation] = []
    if args.weather_source == "open-meteo":
        try:
            if station_obj is None:
                station_obj = fetch_station_by_id(station_id)
            if station_obj is None:
                raise ValueError("Could not determine station coordinates for Open-Meteo")
            air_temps, water_temps = fetch_open_meteo_temperatures(
                station_obj.lat, station_obj.lon, args.units, day_start, day_end
            )
        except (ValueError, HTTPError, URLError, TimeoutError):
            air_temps, water_temps = [], []
    else:
        try:
            air_temps = fetch_observations("air_temperature", station_id, args.units, day_start, day_end)
        except (ValueError, HTTPError, URLError, TimeoutError):
            air_temps = []
        try:
            water_temps = fetch_observations("water_temperature", station_id, args.units, day_start, day_end)
        except (ValueError, HTTPError, URLError, TimeoutError):
            water_temps = []

    min_v = min(today_values)
    max_v = max(today_values)

    print(f"Tide Station: {station_id}")
    if station_label:
        print(station_label)
    print(f"As of: {now.strftime('%Y-%m-%d %H:%M')}")
    print(f"Weather source: {args.weather_source}")
    print(f"Current predicted tide: {format_height(current.v, args.units)}")
    print(f"Today's range: {format_height(min_v, args.units)} to {format_height(max_v, args.units)}")
    if air_temps:
        current_air = nearest_observation(air_temps, now)
        print(f"Current air temperature: {format_temp(current_air.v, args.units)}")
    else:
        print("Current air temperature: N/A")
    if water_temps:
        current_water = nearest_observation(water_temps, now)
        print(f"Current water temperature: {format_temp(current_water.v, args.units)}")
    else:
        print("Current water temperature: N/A")
    print()
    print("Today (00:00-24:00) tide graph")
    graph_lines = render_ascii_graph(today_values, day_start, day_end, now, args.units)
    if not graph_lines:
        print("(no graph data)")
    else:
        for line in graph_lines:
            print(line)
    print()
    print("Today (00:00-24:00) temperature graph")
    temp_graph_lines = render_temperature_graph(air_temps, water_temps, day_start, day_end, now, args.units)
    if not temp_graph_lines:
        print("(no temperature data for this station)")
    else:
        for line in temp_graph_lines:
            print(line)
    print()
    print("Upcoming high/low tides")
    if not upcoming:
        print("No upcoming events available.")
    else:
        for event in upcoming:
            label = "High" if event.kind.upper() == "H" else "Low"
            print(f"- {event.t.strftime('%a %Y-%m-%d %H:%M')}  {label:4}  {format_height(event.v, args.units)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
