from pydantic import BaseModel
from typing import List, Dict, Any

class RecommendRequest(BaseModel):
    komoditas_id: str

class ForecastRequest(BaseModel):
    komoditas_id: str
    periods: int = 30

class AIResponse(BaseModel):
    komoditas_id: str
    predictions: List[Dict[str, Any]]
    historical: List[Dict[str, Any]] = []

class UpdateDataRequest(BaseModel):
    komoditas_id: str
    tanggal: str
    harga_aktual: float
