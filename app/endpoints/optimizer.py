from http.client import HTTPException
from fastapi import APIRouter
from app.entities.optimizer import RouteRequest, RouteResponse
from app.services.optimizer import optimize_route_logic, optimize_with_kmeans_capacity
from app.api.deps import CurrentUser, SessionDep

router = APIRouter()

@router.post("/optimize/route", response_model=RouteResponse, tags=["optimizer"])
def optimize_route(
    request: RouteRequest,
    session: SessionDep):
    #current_user: CurrentUser):
    #if not current_user.is_superuser:
    #    raise HTTPException(status_code=403, detail="No tienes permisos para optimizar rutas.")
    return optimize_with_kmeans_capacity(request)
