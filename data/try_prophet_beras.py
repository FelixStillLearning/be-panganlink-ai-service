import pandas as pd
from prophet import Prophet

csv_path = "data/raw/komoditas_beras_2022_2026.csv"

df = pd.read_csv(csv_path)

df = (
    df.groupby("Date_Param")["Price"]
    .mean()
    .reset_index()
    .rename(columns={"Date_Param": "ds", "Price": "y"})
)

df["ds"] = pd.to_datetime(df["ds"])
df = df.sort_values("ds")

model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
)
model.fit(df)

future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

print(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(30))