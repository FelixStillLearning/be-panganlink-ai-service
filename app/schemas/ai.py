from pydantic import BaseModel
from typing import List, Dict, Any

class RecommendRequest(BaseModel):
    komoditas_id: str

class ForecastRequest(BaseModel):
    komoditas_id: str

class AIResponse(BaseModel):
    komoditas_id: str
    predictions: List[Dict[str, Any]]
