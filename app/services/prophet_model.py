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

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


def _to_int_kid(komoditas_id: Union[str, int]) -> int:
    try:
        return int(komoditas_id)
    except (TypeError, ValueError):
        raise ValueError(f"komoditas_id tidak valid: {komoditas_id}")


def _model_path(komoditas_id: int) -> Path:
    name = COMMODITIES.get(komoditas_id)
    if not name:
        raise ValueError(f"komoditas_id tidak dikenal: {komoditas_id}")
    return MODEL_DIR / f"prophet_{name}.joblib"


def fetch_historical_data(komoditas_id: int) -> pd.DataFrame:
    query = text(
        """
        SELECT tanggal AS ds, harga AS y
        FROM harga_pasar
        WHERE komoditas_id = :komoditas_id
        ORDER BY tanggal ASC
        """
    )
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"komoditas_id": komoditas_id})

    df["ds"] = pd.to_datetime(df["ds"])
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna().sort_values("ds").reset_index(drop=True)
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
    m.fit(df)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(m, path)
    return m


def generate_forecast(komoditas_id: Union[str, int], periods: int = 30) -> list:
    kid = _to_int_kid(komoditas_id)
    if periods <= 0:
        return []

    model = _load_or_train_model(kid)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    future_forecast = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods)

    result = []
    for _, row in future_forecast.iterrows():
        result.append(
            {
                "tanggal": row["ds"].strftime("%Y-%m-%d"),
                "prediksi_harga": round(float(row["yhat"]), 2),
                "batas_bawah": round(float(row["yhat_lower"]), 2),
                "batas_atas": round(float(row["yhat_upper"]), 2),
            }
        )
    return result


def get_recommendation(komoditas_id: Union[str, int]) -> list:
    return generate_forecast(komoditas_id, periods=1)