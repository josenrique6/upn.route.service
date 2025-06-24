from fastapi import APIRouter
from app.api.api_v1.endpoints import optimizer

api_router = APIRouter()
api_router.include_router(optimizer.router, prefix="", tags=["optimizer"])