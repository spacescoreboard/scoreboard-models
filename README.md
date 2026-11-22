# Weather Scoreboard forecast models

Twice a day, a GitHub Actions job downloads three forecast models:

- the American GFS (NOAA, public domain);
- the European IFS (ECMWF open data, CC BY 4.0);
- the Canadian GDPS/GEM (Environment and Climate Change Canada, Open Government Licence – Canada).

It reads each model's forecast for every Weather Scoreboard city for the next 72 hours. When a storm is coming, it
also draws an animated snowfall map comparing the three models. Everything is sent to WeatherScoreboard.com.

- `models.py`: downloads the models, extracts the city forecasts, draws the map and uploads it.
- `cities.json`: the 363 cities, from the Weather Scoreboard plugin.
- `fonts/`: Atkinson Hyperlegible and Bricolage Grotesque, under the SIL Open Font License (license files included).
- `.github/workflows/models.yml`: the schedule, at 4:20 AM and 4:20 PM Eastern.

Snow is each model's snowfall as liquid times 10, the common 10-to-1 ratio, so the three models are measured the
same way. Where a model's snowfall isn't available, snow is the precipitation that falls at about 33°F or colder.

## Settings (Settings › Secrets and variables › Actions)

- `SITE_URL`: `https://www.weatherscoreboard.com`
- `WIS_MODELS_KEY`: the key shown in the plugin under Settings › Forecast models

## Testing

Actions tab › Forecast models › Run workflow. Tick "Draw the map even if no storm is coming" to see a map any day.
Each run's page offers the map to download for a week.
