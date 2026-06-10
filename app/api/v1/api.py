from fastapi import APIRouter

from app.api.v1.endpoints import health, placeholder, ai

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(placeholder.router, prefix="/placeholder", tags=["placeholder"])
api_router.include_router(ai.router, prefix="/ai", tags=["ai"])
