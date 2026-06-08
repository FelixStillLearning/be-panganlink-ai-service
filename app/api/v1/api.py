from fastapi import APIRouter

from app.api.v1.endpoints import health, placeholder

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(placeholder.router, prefix="/placeholder", tags=["placeholder"])
