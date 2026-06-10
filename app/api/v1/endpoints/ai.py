from fastapi import APIRouter, HTTPException
from app.schemas.ai import RecommendRequest, ForecastRequest, AIResponse, UpdateDataRequest
from app.services.prophet_model import get_recommendation, generate_forecast, append_historical_data, remove_historical_data

router = APIRouter()

@router.post("/recommend")
def recommend_price(request: RecommendRequest):
    try:
        preds = get_recommendation(request.komoditas_id)
        if not preds:
            raise HTTPException(status_code=404, detail="Not enough data to generate recommendation")
        return preds
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/forecast", response_model=AIResponse)
def forecast_price(request: ForecastRequest):
    try:
        preds = generate_forecast(request.komoditas_id, periods=request.periods)
        if not preds:
            raise HTTPException(status_code=404, detail="Not enough data to generate forecast")
        return AIResponse(
            komoditas_id=request.komoditas_id, 
            predictions=preds.get("prediksi", []),
            historical=preds.get("historical", [])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/update_data")
def update_data(request: UpdateDataRequest):
    try:
        success = append_historical_data(
            komoditas_id=int(request.komoditas_id),
            tanggal=request.tanggal,
            harga_aktual=request.harga_aktual
        )
        if not success:
            raise HTTPException(status_code=400, detail="Gagal menyimpan data baru. Pastikan komoditas ID valid.")
        return {"message": "Data berhasil ditambahkan dan AI akan mulai menggunakannya untuk prediksi berikutnya."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/delete_data")
def delete_data(komoditas_id: str, tanggal: str):
    try:
        success = remove_historical_data(
            komoditas_id=int(komoditas_id),
            tanggal=tanggal
        )
        if not success:
            raise HTTPException(status_code=400, detail="Gagal menghapus data historis.")
        return {"message": "Data berhasil dihapus dari sistem AI."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
