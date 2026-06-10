from pathlib import Path
from typing import Dict, Union

import joblib
import pandas as pd
from prophet import Prophet
from sqlalchemy import text

from app.core.database import engine

COMMODITIES: Dict[int, str] = {
    1: "beras",
    2: "cabai_merah",
    3: "bawang_merah",
}

MODEL_DIR = Path(__file__).resolve().parents[2] / "ml" / "models"


def _to_int_kid(komoditas_id: Union[str, int]) -> int:
    try:
        return int(komoditas_id)
    except (TypeError, ValueError):
        raise ValueError(f"komoditas_id tidak valid: {komoditas_id}")


def _model_path(komoditas_id: int) -> Path:
    name = COMMODITIES.get(komoditas_id)
    if not name:
        raise ValueError(f"komoditas_id tidak dikenal: {komoditas_id}")
        
    if MODEL_DIR.exists():
        runs = sorted([d for d in MODEL_DIR.iterdir() if d.is_dir() and d.name.startswith("run_")])
        if runs:
            latest_run = runs[-1]
            return latest_run / name / f"prophet_{name}.joblib"
            
    return MODEL_DIR / f"prophet_{name}.joblib"


def fetch_historical_data(komoditas_id: int) -> pd.DataFrame:
    name = COMMODITIES.get(komoditas_id)
    if not name:
        return pd.DataFrame()
    
    csv_path = Path(__file__).resolve().parents[2] / "data" / "raw" / f"komoditas_{name}_2022_2026.csv"
    if not csv_path.exists():
        raise ValueError(f"File CSV tidak ditemukan: {csv_path}")
        
    df = pd.read_csv(csv_path)
    df["ds"] = pd.to_datetime(df["Date_Param"])
    df["y"] = pd.to_numeric(df["Price"], errors="coerce")
    df = df.dropna(subset=["ds", "y"])
    
    df = df.groupby("ds")["y"].mean().reset_index()
    df = df.sort_values("ds")
    
    if not df.empty:
        df = df.set_index("ds").resample("D").interpolate(method="linear").reset_index()
        
    return df


def _load_or_train_model(komoditas_id: int) -> Prophet:
    path = _model_path(komoditas_id)
    if path.exists():
        return joblib.load(path)

    df = fetch_historical_data(komoditas_id)
    if df.empty or len(df) < 2:
        raise ValueError("Data historis tidak cukup untuk training model.")

    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
    )
    m.add_country_holidays(country_name='ID')
    m.fit(df)

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(m, path)
    return m


def generate_forecast(komoditas_id: Union[str, int], periods: int = 30) -> dict:
    kid = _to_int_kid(komoditas_id)
    if periods <= 0:
        return {"komoditas_id": komoditas_id, "prediksi": []}

    model = _load_or_train_model(kid)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    
    forecast["yhat"] = np.exp(forecast["yhat"])
    forecast["yhat_lower"] = np.exp(forecast["yhat_lower"])
    forecast["yhat_upper"] = np.exp(forecast["yhat_upper"])

    forecast["yhat"] = forecast["yhat"].apply(lambda x: max(0.0, x))
    forecast["yhat_lower"] = forecast["yhat_lower"].apply(lambda x: max(0.0, x))
    forecast["yhat_upper"] = forecast["yhat_upper"].apply(lambda x: max(0.0, x))

    future_forecast = forecast.tail(periods)

    result = []
    for _, row in future_forecast.iterrows():
        result.append(
            {
                "tanggal": row["ds"].strftime("%Y-%m-%d"),
                "prediksi_harga": round(row["yhat"], 2),
                "batas_bawah": round(row["yhat_lower"], 2),
                "batas_atas": round(row["yhat_upper"], 2),
            }
        )
    return {"komoditas_id": komoditas_id, "prediksi": result}


def get_recommendation(komoditas_id: Union[str, int]) -> list:
    return generate_forecast(komoditas_id, periods=1)