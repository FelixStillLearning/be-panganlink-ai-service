from pathlib import Path
from typing import Dict

import pandas as pd
from sqlalchemy import text

from app.core.database import engine

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
MODEL_DIR = PROJECT_DIR / "ml" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

COMMODITIES: Dict[int, str] = {
    1: "beras",
    2: "cabai_merah",
    3: "bawang_merah",
}


def fetch_historical_data(komoditas_id: int) -> pd.DataFrame:
    name = COMMODITIES.get(komoditas_id)
    if not name:
        return pd.DataFrame()
    
    csv_path = PROJECT_DIR / "data" / "raw" / f"komoditas_{name}_2022_2026.csv"
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