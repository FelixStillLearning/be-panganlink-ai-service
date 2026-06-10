from pathlib import Path
from typing import Dict, Union

import joblib
import pandas as pd
import numpy as np

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
        runs = sorted([d for d in MODEL_DIR.iterdir() if d.is_dir() and d.name.startswith("run_huber_")])
        if runs:
            latest_run = runs[-1]
            return latest_run / name / f"huber_{name}.joblib"
            
    return MODEL_DIR / f"huber_{name}.joblib"


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


def _load_model(komoditas_id: int):
    path = _model_path(komoditas_id)
    if path.exists():
        return joblib.load(path)
    raise FileNotFoundError(f"Model Huber belum ditraining untuk komoditas {komoditas_id}. Silakan jalankan 'python ml/train_huber.py'.")


def generate_forecast(komoditas_id: Union[str, int], periods: int = 30) -> dict:
    kid = _to_int_kid(komoditas_id)
    if periods <= 0:
        return {"komoditas_id": komoditas_id, "prediksi": []}

    model = _load_model(kid)
    
    df = fetch_historical_data(kid)
    if df.empty or len(df) < 7:
        raise ValueError("Data historis tidak cukup untuk membuat lag fitur.")
        
    last_date = df["ds"].max()
    historical_y = df["y"].tolist()
    
    result = []
    current_date = last_date
    
    for _ in range(periods):
        current_date += pd.Timedelta(days=1)
        
        # Calculate features (Lag & Rolling)
        lag_1 = historical_y[-1]
        lag_2 = historical_y[-2]
        lag_3 = historical_y[-3]
        lag_4 = historical_y[-4]
        lag_5 = historical_y[-5]
        lag_6 = historical_y[-6]
        lag_7 = historical_y[-7]
        
        rolling_mean_7 = np.mean(historical_y[-7:])
        rolling_std_7 = np.std(historical_y[-7:], ddof=1) if len(historical_y) >= 7 else 0
        
        dayofweek = current_date.dayofweek
        month = current_date.month
        
        feature_df = pd.DataFrame([{
            "lag_1": lag_1, "lag_2": lag_2, "lag_3": lag_3, 
            "lag_4": lag_4, "lag_5": lag_5, "lag_6": lag_6, "lag_7": lag_7,
            "rolling_mean_7": rolling_mean_7,
            "rolling_std_7": rolling_std_7,
            "dayofweek": dayofweek,
            "month": month
        }])
        
        pred = model.predict(feature_df)[0]
        
        # Append to historical to use for next prediction (Recursive Forecasting)
        historical_y.append(pred)
        
        # Estimate bounds using rolling standard deviation as a proxy
        margin = 1.96 * rolling_std_7
        batas_bawah = max(0.0, pred - margin)
        batas_atas = max(0.0, pred + margin)
        
        result.append({
            "tanggal": current_date.strftime("%Y-%m-%d"),
            "prediksi_harga": round(pred, 2),
            "batas_bawah": round(batas_bawah, 2),
            "batas_atas": round(batas_atas, 2),
        })
        
    return {"komoditas_id": komoditas_id, "prediksi": result}


def get_recommendation(komoditas_id: Union[str, int]) -> list:
    res = generate_forecast(komoditas_id, periods=1)
    return res.get("prediksi", [])