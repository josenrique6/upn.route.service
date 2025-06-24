from datetime import datetime
from pydantic import BaseModel
from typing import List, Optional, Any, Dict


class Location(BaseModel):
    id: str 
    coordinates: List[float]  # [lon, lat]
    time_window: Optional[List[int]] = None  # [inicio, fin]
    service_time: int = 600  # tiempo de atención en segundos 10mi
    # Datos nuevos para cubicaje:
    weight_kg: float                 # peso real en kg
    volume_m3: float                 # volumen en metros cúbicos
    demand: Optional[float] = None   # este campo ya no se envía desde cliente; lo llenaremos calculando max(weight_kg, peso_volum)
    priority: str = "baja"           # prioridad de la entrega: "baja", "media", "alta"

class Truck(BaseModel):
    id: str                # un identificador único por camión
    capacity: float        # capacidad de carga (en volumen, peso o ítems)
    axleload: float = 10
    height: float = 6
    length: float = 8.35
    weight: float = 18
    width: float = 2.3
    hazmat: bool = False    

class TruckOptions(BaseModel):
    axleload: float = 10
    height: float = 6
    length: float = 8.35
    weight: float = 18
    width: float = 2.3
    hazmat: bool = False


class RouteRequest(BaseModel):
    locations: List[Location]
    start_datetime: datetime  
    # -----------------------------------------------------------------
    # Reemplazamos truck_options por una lista de camiones
    trucks: List[Truck]
    # -----------------------------------------------------------------
    round_trip: bool = True
    optimize_by: str = "time"  # "time" o "distance"
    depot_index: int = 0
    avoid_polygons: Optional[List[List[List[List[float]]]]] = None
    use_dbscan: Optional[bool] = True
    dbscan_eps: Optional[float] = 0.01
    dbscan_min_samples: Optional[int] = 3

class EtaResult(BaseModel):
    point: int
    point_id: str                 # ← Incluimos también el id aquí
    eta_seconds: int
    eta_formatted: str

class TruckResponse(BaseModel):
    id: str
    capacity: float
    axleload: float
    height: float
    length: float
    weight: float
    width: float
    hazmat: bool

class EtaItem(BaseModel):
    vehicle_id: str
    point: int
    point_id: str
    eta_seconds: int
    eta_formatted: str

class RouteSegment(BaseModel):
    segment_index: int
    from_index: int
    to_index: int
    polyline: str

class RouteInfo(BaseModel):
    vehicle_id: str
    route_nodes: List[int]
    route_location_ids: List[str]
    route_geometry: str
    route_segments: List[RouteSegment]
    vehicle_distance: float
    vehicle_duration: float
class RouteResponse(BaseModel):
    locations: List[Location]
    routes: List[RouteInfo]
    etas: List[EtaItem]
    total_distance: float
    total_duration: float
    truck_parameters: List[TruckResponse]


