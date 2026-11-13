# London 24-Hour Temperature Forecast — Linear Regression

Predicts the air temperature in London 24 hours ahead from current weather conditions, using ordinary least squares linear regression.

## Data

- Source: [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) (ERA5 reanalysis)
- Location: South Kensington, London (51.4984° N, 0.1769° W)
- 5 years of hourly observations: temperature, humidity, wind speed, pressure, cloud cover, rainfall

## Method

**Target:** temperature at the same location 24 hours later.

**Features (13):**
- Current conditions: temperature, humidity, wind speed, pressure, cloud cover, rainfall
- Cyclical time features: sin/cos of day of year and hour of day, so that 31 December sits next to 1 January and 23:00 next to 00:00
- 24-hour tendencies of temperature, pressure and humidity, since falling pressure tends to signal an incoming front

Cross terms and their gradients were also tested, but they were highly correlated with existing features, so they did not improve the model and only changed the coefficients.

**Evaluation:** a chronological 80/20 train/test split (oldest 80% for training, most recent 20% for testing), so the model never learns from future data. Features are standardised using statistics from the training set only.

## Results

Compared against a persistence baseline, which predicts that the temperature in 24 hours will equal the current temperature:

| Model | RMSE | R² |
|---|---|---|
| Persistence baseline | 2.81 °C | 0.845 |
| Linear regression | 2.52 °C | 0.875 |

The linear regression reduces RMSE by 10.1% relative to persistence.

## Limitations

- A linear model cannot capture non-linear interactions between variables
- Trained and tested on a single location
- Uses only current and 24-hour-old conditions, with no information from forecasts or surrounding areas

## How to run

```bash
pip install -r requirements.txt
python weather_model.py
```
