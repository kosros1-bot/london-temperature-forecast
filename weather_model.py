import datetime as dt
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1) COORDINATES
lat = 51.4984
lon = -0.1769

# 2) DATE RANGE (last 5 years from today)
end_date   = dt.date.today()
start_date = end_date - dt.timedelta(days=365 * 5)

start_str = start_date.isoformat()   # "YYYY-MM-DD"
end_str   = end_date.isoformat()

# 3) VARIABLES TO DOWNLOAD
hourly_vars = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "pressure_msl",
    "cloud_cover",
    "dew_point_2m",
    "rain"
]

# 4) CALL THE API
base_url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude":   lat,
    "longitude":  lon,
    "start_date": start_str,
    "end_date":   end_str,
    "hourly":     ",".join(hourly_vars),
    "timezone":   "GMT"
}

print(f"Downloading hourly data from {start_str} to {end_str} ...")
response = requests.get(base_url, params=params, timeout=60)
response.raise_for_status()
data = response.json()

# 5) BUILD DATAFRAME
hourly = data["hourly"]
df = pd.DataFrame({
    "DateTime":    pd.to_datetime(hourly["time"]),
    "Temperature": hourly["temperature_2m"],
    "Humidity":    hourly["relative_humidity_2m"],
    "Wind Speed":  hourly["wind_speed_10m"],
    "Pressure":    hourly["pressure_msl"],
    "Cloud Cover": hourly["cloud_cover"],
    "Rainfall":    hourly["rain"],
})

print(f"\nRaw rows downloaded : {len(df)}")
print(f"Missing values per column:\n{df.isna().sum()}\n")

# 6) TARGET: temperature 24 hours ahead
df["Target_Temp"] = df["Temperature"].shift(-24)
df["Previous_Temp"] = df["Temperature"].shift(24)
df["Previous_Pressure"] = df["Pressure"].shift(24)
df["Previous_Humidity"] = df["Humidity"].shift(24)

# 7) FEATURE ENGINEERING
day_of_year   = df["DateTime"].dt.dayofyear
df["Sin_Day"] = np.sin(2 * np.pi * day_of_year / 365.25)
df["Cos_Day"] = np.cos(2 * np.pi * day_of_year / 365.25)
hour_of_day = df["DateTime"].dt.hour
df["Sin_Hour"] = np.sin(2 * np.pi * hour_of_day / 24)
df["Cos_Hour"] = np.cos(2 * np.pi * hour_of_day / 24)

df["Temperature Gradient"] = (df["Temperature"] - df["Previous_Temp"])/24
df["Pressure Gradient"] = (df["Pressure"] - df["Previous_Pressure"])/24
df["Humidity Gradient"] = (df["Humidity"] - df["Previous_Humidity"])/24

# Tried cross terms and cross term gradients, but they were highly correlated with existing features, so they did not improve the model and only changed the coefficients

# 8) DROP ROWS WITH NO VALUES DUE TO THE SHIFT
df = df.dropna().reset_index(drop=True)
print(f"Rows after dropping NaN rows : {len(df)}")

# 9) CHRONOLOGICAL TRAIN / TEST SPLIT
n         = len(df)
train_end = int(n * 0.80)   # 80% training (oldest data)
                            # 20% test     (most recent data)

train_df = df.iloc[:train_end].copy()
test_df  = df.iloc[train_end:].copy()

print(f"\nSplit summary:")
print(f"  Training : {len(train_df):>6} rows  ({train_df['DateTime'].iloc[0].date()} → {train_df['DateTime'].iloc[-1].date()})")
print(f"  Test     : {len(test_df):>6} rows  ({test_df['DateTime'].iloc[0].date()} → {test_df['DateTime'].iloc[-1].date()})")

# 10) SAVE TO CSV
df.to_csv("london_weather.csv", index=False)
print(f"\nFull dataset saved to 'london_weather.csv'")
print(df[["DateTime","Temperature","Target_Temp","Sin_Day","Cos_Day"]].head())

# 11) DEFINE FEATURES AND TARGET
# Select the 13 columns the model will use as inputs (X)
# and the column we want to predict (y)
feature_cols = ["Temperature","Humidity","Wind Speed","Pressure",
                "Cloud Cover", "Rainfall","Sin_Day","Cos_Day", "Sin_Hour", "Cos_Hour", "Temperature Gradient", "Pressure Gradient", "Humidity Gradient"]

X_train = train_df[feature_cols].to_numpy()   # inputs for training
y_train = train_df["Target_Temp"].to_numpy()  # targets for training
X_test  = test_df[feature_cols].to_numpy()    # inputs for testing
y_test  = test_df["Target_Temp"].to_numpy()   # targets for testing

# 12) SCALE THE FEATURES
# Transforms each feature to mean=0, std=1 using: z = (x - mean) / std
# We fit ONLY on training data to avoid the model peeking at future data
scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)  # learns mean & std, then scales
X_test_s  = scaler.transform(X_test)       # applies the same mean & std

# 13) FIT THE MODEL
# Ordinary least squares: finds the theta that minimises the sum of squared errors
# .fit() is the "learning" step — it finds the best weights (theta values)
model = LinearRegression()
model.fit(X_train_s, y_train)

# 14) PRINT THE HYPOTHESIS FUNCTION
# The full equation the model learned:
# y_hat = intercept + theta_1*Temperature + theta_2*Humidity + ...
print("=" * 50)
print("LEARNED HYPOTHESIS FUNCTION")
print("=" * 50)
print(f"y_hat = {model.intercept_:+.4f}")
for name, coef in zip(feature_cols, model.coef_):
    direction = "↑ warmer" if coef > 0 else "↓ cooler"
    print(f"        {coef:+.6f}  ×  {name:<15}  ({direction})")

# 15) PREDICT ON TEST SET
# Applies the learned equation: y_hat = X @ theta + intercept
# We use the TEST set here — data the model has never seen during training
y_pred_test = model.predict(X_test_s)

# 16) EVALUATE ON TEST SET
# RMSE: average error in °C — lower is better
# R²  : fraction of variance explained — closer to 1.0 is better
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2_test   = r2_score(y_test, y_pred_test)
print(f"\nTest RMSE : {rmse_test:.3f} °C")
print(f"Test R²   : {r2_test:.4f}")

y_pred_persist = test_df["Temperature"].to_numpy()
rmse_persist   = np.sqrt(mean_squared_error(y_test, y_pred_persist))
r2_persist     = r2_score(y_test, y_pred_persist)
improvement    = (rmse_persist - rmse_test) / rmse_persist * 100
print(f"\nPersistence RMSE : {rmse_persist:.3f} °C")
print(f"Persistence R²   : {r2_persist:.4f}")
print(f"Improvement over persistence : {improvement:.1f}%")

# 17) PLOT RESIDUALS ON TEST SET
# Residuals = actual - predicted (our estimate of epsilon)
# Left plot  : histogram — should look like a bell curve centred at 0
# Right plot : residuals vs predicted — should show no pattern (random scatter)
residuals = y_test - y_pred_test

fig, ax = plt.subplots(1, 2, figsize=(12, 4))

ax[0].hist(residuals, bins=60, edgecolor="black", color="steelblue")
ax[0].axvline(0, color="red", linestyle="--", label="zero")
ax[0].set_title("Test Residuals (Histogram)")
ax[0].set_xlabel("Residual (°C)")
ax[0].set_ylabel("Count")
ax[0].legend()

ax[1].scatter(y_pred_test, residuals, s=6, alpha=0.15, color="steelblue")
ax[1].axhline(0, color="red", linestyle="--")
ax[1].set_title("Residuals vs Predicted Temperature")
ax[1].set_xlabel("Predicted Temp (°C)")
ax[1].set_ylabel("Residual (°C)")

plt.tight_layout()
plt.show()
