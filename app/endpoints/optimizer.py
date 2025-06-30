from fastapi import APIRouter

from app.api.deps import SessionDep
from app.entities.optimizer import RouteRequest, RouteResponse
from app.services.optimizer import optimize_with_kmeans_capacity

router = APIRouter()

@router.post("/optimize/route", response_model=RouteResponse, tags=["optimizer"])
def optimize_route(
    request: RouteRequest,
    _session: SessionDep,
):
    """Retorna la ruta optimizada empleando la lógica de K-Means."""
    #current_user: CurrentUser):
    #if not current_user.is_superuser:
    #    raise HTTPException(status_code=403, detail="No tienes permisos para optimizar rutas.")
    return optimize_with_kmeans_capacity(request)
