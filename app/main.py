from fastapi import FastAPI

from app.api.v1.api import api_router

app = FastAPI(
    title="Template Service",
    version="0.1.0",
    description="FastAPI service template",
)

app.include_router(api_router, prefix="/api/v1")
