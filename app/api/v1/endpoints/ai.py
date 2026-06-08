from fastapi import APIRouter, HTTPException
from app.schemas.ai import RecommendRequest, ForecastRequest, AIResponse
from app.services.prophet_model import get_recommendation, generate_forecast

router = APIRouter()

@router.post("/recommend", response_model=AIResponse)
def recommend_price(request: RecommendRequest):
    try:
        preds = get_recommendation(request.komoditas_id)
        if not preds:
            raise HTTPException(status_code=404, detail="Not enough data to generate recommendation")
        return AIResponse(komoditas_id=request.komoditas_id, predictions=preds)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/forecast", response_model=AIResponse)
def forecast_price(request: ForecastRequest):
    try:
        preds = generate_forecast(request.komoditas_id, periods=30)
        if not preds:
            raise HTTPException(status_code=404, detail="Not enough data to generate forecast")
        return AIResponse(komoditas_id=request.komoditas_id, predictions=preds)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
