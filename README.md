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

Use ZIP code auto-selection:

```bash
./tides --zip 95501
```

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
