import pandas as pd
from prophet import Prophet
from sqlalchemy import text
from app.core.database import engine

def fetch_historical_data(komoditas_id: str) -> pd.DataFrame:
    query = text("SELECT tanggal AS ds, harga AS y FROM harga_pasar WHERE komoditas_id = :komoditas_id ORDER BY tanggal ASC")
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"komoditas_id": komoditas_id})
    return df

def generate_forecast(komoditas_id: str, periods: int = 30) -> list:
    df = fetch_historical_data(komoditas_id)
    if df.empty or len(df) < 2:
        return []

    # Initialize and fit Prophet
    m = Prophet(daily_seasonality=True, yearly_seasonality=False, weekly_seasonality=True)
    m.fit(df)

    # Generate future dates
    future = m.make_future_dataframe(periods=periods)
    forecast = m.predict(future)

    # Filter to only return future predictions (tail)
    future_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
    
    # Format to dictionary
    result = []
    for _, row in future_forecast.iterrows():
        result.append({
            "tanggal": row['ds'].strftime('%Y-%m-%d'),
            "prediksi_harga": round(row['yhat'], 2),
            "batas_bawah": round(row['yhat_lower'], 2),
            "batas_atas": round(row['yhat_upper'], 2)
        })
    return result

def get_recommendation(komoditas_id: str) -> list:
    # Recommend is essentially forecasting for 1 period (next day)
    return generate_forecast(komoditas_id, periods=1)
