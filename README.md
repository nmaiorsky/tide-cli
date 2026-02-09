# tide-cli

Simple macOS terminal app to view current tide predictions and upcoming high/low tide events.
It also attempts to show air and water temperature (when the chosen station provides those sensors).

## Requirements

- macOS with `python3`
- Internet access (uses NOAA CO-OPS API)

## Quick start

```bash
cd /Users/nickmaiorsky/dev/tide-cli
./tides --station 9414290
```

Or set your default station once:

```bash
export TIDE_STATION=9414290
./tides
```

## Using US ZIP codes

```bash
./tides --zip 95501
```

Notes:
- `--zip` accepts US ZIP codes only (for example `95501`, `10001`, `33139`).
- ZIP resolution uses a US ZIP geocoding service, then finds the nearest NOAA tide station.
- If the closest station does not provide tide predictions, the CLI tries nearby stations automatically.

## Using outside the US

`--zip` is US-only. For non-US locations, use an explicit station id:

```bash
./tides --station <station_id>
```

Recommended workflow:
1. Find a tide station id for your area from the data provider you want to use.
2. Run the CLI with `--station`.
3. Optionally set a default:

```bash
export TIDE_STATION=<station_id>
./tides
```

Important:
- Tide predictions in this app currently come from NOAA CO-OPS, so coverage is best in US/coastal NOAA station regions.
- `--weather-source open-meteo` can still provide air/water temperature data for many non-US coordinates, but tide prediction availability still depends on the station id.

## Useful flags

- `--station <id>` NOAA station id
- `--zip <us_zip>` Find nearest NOAA tide station from ZIP code
- `--units english|metric` Height units (default: english)
- `--events <n>` Number of upcoming high/low events (default: 4)
- `--weather-source noaa|open-meteo` Temperature provider (default: noaa)

## Example stations (US)

- `9414290` San Francisco, CA
- `9447130` Seattle, WA
- `8720218` Mayport, FL

You can find station ids at NOAA Tides & Currents station pages.
