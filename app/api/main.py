from fastapi import APIRouter
from fastapi import FastAPI

from app.api.routes import utils
from app.core.config import settings
from app.endpoints import optimizer

app = FastAPI(debug=True)
api_router = APIRouter()
api_router.include_router(utils.router)
api_router.include_router(optimizer.router)
