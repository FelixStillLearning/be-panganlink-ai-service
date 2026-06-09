from pathlib import Path
from typing import Dict

import pandas as pd
from sqlalchemy import text

from app.core.database import engine

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
MODEL_DIR = PROJECT_DIR / "app" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

COMMODITIES: Dict[int, str] = {
    1: "beras",
    2: "cabai_merah",
    3: "bawang_merah",
}


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