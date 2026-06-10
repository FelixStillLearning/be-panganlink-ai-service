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
    """Fetch historical data from CSV files"""
    global COMMODITIES
    komoditas_name = COMMODITIES.get(komoditas_id)
    if not komoditas_name:
        return pd.DataFrame()
        
    csv_path = Path(__file__).resolve().parents[2] / "data" / "raw" / f"komoditas_{komoditas_name}_2022_2026.csv"
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

def append_historical_data(komoditas_id: int, tanggal: str, harga_aktual: float) -> bool:
    """Append new actual data from admin to the CSV file"""
    global COMMODITIES
    komoditas_name = COMMODITIES.get(komoditas_id)
    if not komoditas_name:
        raise ValueError(f"Komoditas ID {komoditas_id} tidak valid.")
        
    csv_path = Path(__file__).resolve().parents[2] / "data" / "raw" / f"komoditas_{komoditas_name}_2022_2026.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"File data untuk komoditas {komoditas_name} tidak ditemukan.")
        
    try:
        # Validate date format first
        try:
            parsed_date = pd.to_datetime(tanggal)
            formatted_date = parsed_date.strftime('%Y-%m-%d')
        except Exception:
            raise ValueError(f"Format tanggal '{tanggal}' tidak valid. Gunakan format YYYY-MM-DD.")

        df = pd.read_csv(csv_path)
        # Check if date already exists
        if formatted_date in df['Date_Param'].values:
            df.loc[df['Date_Param'] == formatted_date, 'Price'] = harga_aktual
        else:
            new_row = pd.DataFrame([{'Date_Param': formatted_date, 'Price': harga_aktual}])
            df = pd.concat([df, new_row], ignore_index=True)
            
        df['Date_Param'] = pd.to_datetime(df['Date_Param']).dt.strftime('%Y-%m-%d')
        df = df.sort_values('Date_Param')
        df.to_csv(csv_path, index=False)
        return True
    except ValueError as ve:
        raise ve
    except Exception as e:
        print(f"Error updating data: {e}")
        raise RuntimeError(f"Gagal memperbarui data: {str(e)}")

def remove_historical_data(komoditas_id: int, tanggal: str) -> bool:
    """Remove historical data from the CSV file"""
    global COMMODITIES
    komoditas_name = COMMODITIES.get(komoditas_id)
    if not komoditas_name:
        raise ValueError(f"Komoditas ID {komoditas_id} tidak valid.")
        
    csv_path = Path(__file__).resolve().parents[2] / "data" / "raw" / f"komoditas_{komoditas_name}_2022_2026.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"File data untuk komoditas {komoditas_name} tidak ditemukan.")
        
    try:
        # Validate date format
        try:
            parsed_date = pd.to_datetime(tanggal)
            formatted_date = parsed_date.strftime('%Y-%m-%d')
        except Exception:
            raise ValueError(f"Format tanggal '{tanggal}' tidak valid. Gunakan format YYYY-MM-DD.")

        df = pd.read_csv(csv_path)
        # Hapus baris yang tanggalnya sama
        df = df[df['Date_Param'] != formatted_date]
        df.to_csv(csv_path, index=False)
        return True
    except ValueError as ve:
        raise ve
    except Exception as e:
        print(f"Error deleting data: {e}")
        raise RuntimeError(f"Gagal menghapus data: {str(e)}")


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
        
    historical_output = []
    # Kembalikan 365 hari (1 tahun) terakhir untuk grafik historis
    recent_history = df.tail(365)
    for _, row in recent_history.iterrows():
        historical_output.append({
            "tanggal": row["ds"].strftime("%Y-%m-%d"),
            "harga_aktual": round(row["y"], 2)
        })
        
    return {"komoditas_id": komoditas_id, "prediksi": result, "historical": historical_output}


def get_recommendation(komoditas_id: Union[str, int]) -> dict:
    kid = _to_int_kid(komoditas_id)
    res = generate_forecast(kid, periods=7)
    
    historical = res.get("historical", [])
    predictions = res.get("prediksi", [])
    
    if not historical or not predictions:
        return {}
        
    last_actual = historical[-1]["harga_aktual"]
    next_week_pred = predictions[-1]["prediksi_harga"] # Prediksi H+7
    tomorrow_pred = predictions[0]["prediksi_harga"]
    
    # Hitung persentase perubahan dari harga terakhir ke H+7
    diff = next_week_pred - last_actual
    pct = (diff / last_actual) * 100 if last_actual > 0 else 0
    
    direction = "up" if pct > 0 else "down" if pct < 0 else "stable"
    icon = "trending_up" if direction == "up" else "trending_down" if direction == "down" else "trending_flat"
    
    # Buat narasi rekomendasi berdasarkan Huber Regression output
    if pct > 5:
        trend_text = f"+{abs(pct):.1f}% (7 Hari ke depan)"
        recommendation = "Harga diprediksi naik signifikan dalam seminggu ke depan. Tahan stok Anda jika memungkinkan untuk dijual saat harga puncak."
    elif pct < -5:
        trend_text = f"-{abs(pct):.1f}% (7 Hari ke depan)"
        recommendation = "Tren harga menunjukkan penurunan tajam. Segera jual stok Anda sekarang sebelum harga semakin jatuh."
    elif pct > 0:
        trend_text = f"+{abs(pct):.1f}% (Stabil Naik)"
        recommendation = "Harga diprediksi akan stabil dengan sedikit kenaikan. Anda dapat menjual stok secara bertahap."
    else:
        trend_text = f"-{abs(pct):.1f}% (Stabil Turun)"
        recommendation = "Harga akan mengalami sedikit koreksi turun namun tetap stabil. Jual sesuai kebutuhan operasional."
        
    return {
        "komoditas_id": kid,
        "komoditas_name": COMMODITIES.get(kid, "Komoditas"),
        "current_price": last_actual,
        "predicted_price": tomorrow_pred,
        "direction": direction,
        "trend_text": trend_text,
        "icon": icon,
        "recommendation": recommendation
    }