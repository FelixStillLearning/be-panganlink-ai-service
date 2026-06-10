import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error

# 1. Load data
df = pd.read_csv("data/raw/komoditas_beras_2022_2026.csv")
df["ds"] = pd.to_datetime(df["Date_Param"])
df["y"] = pd.to_numeric(df["Price"], errors="coerce")
df = df.dropna(subset=["ds", "y"])

# 2. Group by date (average across provinces)
df = df.groupby("ds")["y"].mean().reset_index()
df = df.sort_values("ds")

# 3. Resample
df = df.set_index("ds").resample("D").interpolate(method="linear").reset_index()

print(f"Total data points: {len(df)}")

# 4. Train/Test split (last 15% for test)
test_size = int(len(df) * 0.15)
train_df = df.iloc[:-test_size]
test_df = df.iloc[-test_size:]

print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

# 5. Train model
m = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    changepoint_prior_scale=0.5,
    seasonality_mode='multiplicative'
)
m.add_country_holidays(country_name='ID')
m.fit(train_df)

# 6. Predict on test
future = test_df[["ds"]]
forecast = m.predict(future)

y_true = test_df["y"].values
y_pred = forecast["yhat"].values

mae = mean_absolute_error(y_true, y_pred)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print(f"MAE: {mae:.2f}")
print(f"MAPE: {mape:.2f}%")

# 7. Plot
plt.figure(figsize=(10, 5))
plt.plot(train_df["ds"], train_df["y"], label="Train Actual")
plt.plot(test_df["ds"], test_df["y"], label="Test Actual")
plt.plot(forecast["ds"], forecast["yhat"], label="Test Predicted")
plt.title("Beras Forecast")
plt.legend()
plt.savefig("beras_test_plot.png")
print("Saved beras_test_plot.png")
