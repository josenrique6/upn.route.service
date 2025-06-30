from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from typing import Any

import numpy as np
import requests
from fastapi import HTTPException
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from sklearn.cluster import DBSCAN, KMeans

from app.entities.optimizer import (
    EtaItem,
    Location,
    RouteInfo,
    RouteRequest,
    RouteResponse,
    TruckResponse,
)

# Definimos el divisor volumétrico (kg por m³)
FACTOR_VOLUM_TO_KG = 167.0

def fetch_cost_pair(i: int, j: int, coords: list[list[float]], truck_options: dict[str, Any], headers: dict[str, str], route_url: str, preference: str) -> tuple[int, int, int, int]:
    """Realiza una sola petición a ORS Directions y retorna (i, j, distancia, duración)."""
    body = {
        "preference": preference,
        "coordinates": [coords[i], coords[j]],
        "profile": "driving-hgv",
        "options": truck_options,
        "instructions": True,
        "geometry": False
    }
    resp = requests.post(route_url, json=body, headers=headers)
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    summary = resp.json().get("routes", [{}])[0].get("summary", {})
    dist = int(round(summary.get("distance", 0)))
    dur = int(round(summary.get("duration", 0)))
    return i, j, dist, dur

def build_cost_matrices(
    coordinates: list[list[float]],
    truck_options: dict[str, Any],
    headers: dict[str, str],
    route_url: str,
    max_workers: int = 20,
    preference: str = "fastest"
) -> tuple[list[list[int]], list[list[int]]]:
    """
    Construye matrices de distancia y tiempo usando llamadas paralelas a ORS Directions,
    respetando exclusiones definidas en truck_options.
    """
    n = len(coordinates)
    distance_matrix = [[0] * n for _ in range(n)]
    time_matrix = [[0] * n for _ in range(n)]

    # Lanzar peticiones concurrentes
    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for i in range(n):
            for j in range(n):
                if i != j:
                    futures.append(pool.submit(fetch_cost_pair, i, j, coordinates, truck_options, headers, route_url, preference))
        for future in as_completed(futures):
            i, j, dist, dur = future.result()
            distance_matrix[i][j] = dist
            time_matrix[i][j] = dur

    return distance_matrix, time_matrix

def optimize_route_logic(request: RouteRequest) -> dict[str, Any]:
    """Resuelve el VRP para todas las ubicaciones y camiones recibidos."""

    coordinates = [loc.coordinates for loc in request.locations]
    service_times = [loc.service_time for loc in request.locations]
    location_ids = [loc.id for loc in request.locations]
    preference = "recommended" if request.optimize_by.lower() == "time" else "shortest"
    n = len(coordinates)
    start_time_dt = request.start_datetime
    prio_map = {"baja": 0.0, "media": 0.5, "alta": 1.0}
    PENALTY_FACTOR = 300

    # Construimos un solo truck_options “genérico” usando el primer camión
    first_truck = request.trucks[0]
    common_truck_opts = {
        "vehicle_type": "delivery",
        "profile_params": {
            "restrictions": {
                "axleload": first_truck.axleload,
                "height": first_truck.height,
                "length": first_truck.length,
                "weight": first_truck.weight,
                "width": first_truck.width,
                "hazmat": first_truck.hazmat,
            }
        }
    }

    #print(request.avoid_polygons)

    if request.avoid_polygons:
        common_truck_opts["avoid_polygons"] = {
            "type": "MultiPolygon",
            "coordinates": request.avoid_polygons
        }

    route_url = "http://localhost:8080/ors/v2/directions/driving-hgv"
    headers = {"Content-Type": "application/json"}
    distance_matrix, time_matrix_original = build_cost_matrices(
        coordinates, common_truck_opts, headers, route_url, preference = preference
    )

    # 1. Parámetros previos
    depot = request.depot_index if 0 <= request.depot_index < n else 0

    # 2. Filtrar solo aquellos sin time_window (reservado para futuras mejoras)

    osm_speed_profiles = {
        "motorway": 45,
        "trunk": 35,
        "primary": 25,
        "secondary": 15,
        "tertiary": 5,
        "residential": 5,
        "unclassified": 20,
    }

    road_type_mapping = {
        (0, 1): "motorway",
        (0, 2): "trunk",
        (0, 3): "primary",
        (1, 0): "motorway",
        (1, 2): "primary",
        (1, 3): "secondary",
        (2, 0): "trunk",
        (2, 1): "primary",
        (2, 3): "secondary",
        (3, 0): "primary",
        (3, 1): "secondary",
        (3, 2): "tertiary",
    }

    adjusted_time_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                adjusted_time_matrix[i][j] = 0
            else:
                dist = distance_matrix[i][j]
                road_type = road_type_mapping.get((i, j), "primary")
                speed_m_s = (osm_speed_profiles[road_type] * 1000) / 3600.0
                adjusted_time = dist / speed_m_s if speed_m_s > 0 else time_matrix_original[i][j]
                adjusted_time_matrix[i][j] = adjusted_time

    adjusted_time_matrix_list = np.rint(adjusted_time_matrix).astype(int).tolist()

    depot = request.depot_index if 0 <= request.depot_index < n else 0
    # 2.1. Determinar número de vehículos y depósitos
    num_vehicles = len(request.trucks)
    depot = request.depot_index if 0 <= request.depot_index < n else 0

    # Si es ruta cerrada (round_trip=True), cada vehículo parte y retorna al mismo depot.
    # Sino, pueden tener destino final distinto (aquí simplificamos: parten desde depot y terminan en depot last_point).
    if request.round_trip:
        manager = pywrapcp.RoutingIndexManager(n, num_vehicles, depot)
    else:
        # Si queremos que cada camión termine en el último nodo (por ejemplo), usar:
        manager = pywrapcp.RoutingIndexManager(
            n,
            num_vehicles,
            [depot] * num_vehicles,      # todos parten desde el mismo depot
            [n - 1] * num_vehicles       # todos terminan en la última ubicación
        )
    routing = pywrapcp.RoutingModel(manager)
    # ------------------------------------------------------------------------------------

    # Lista de penalidades por nodo
    penalties = [int(prio_map[loc.priority] * PENALTY_FACTOR)
                 for loc in request.locations]

    # Construimos la matriz (n×n) de tiempo penalizado
    n = len(request.locations)
    time_prio_matrix = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            # adjusted_time_matrix_list ya incluye solo travel_time + service_time[i]
            # añadimos solo la penalidad del nodo destino j
            time_prio_matrix[i][j] = (
                adjusted_time_matrix_list[i][j]
                + penalties[j]
            )

    def time_prio_lookup(from_index: int, to_index: int) -> int:
        i = manager.IndexToNode(from_index)
        j = manager.IndexToNode(to_index)
        return time_prio_matrix[i][j]

    def adjusted_time_callback(from_index: int, to_index: int) -> int:
        # Índices de OR-Tools → índices de nodo
        from_node = manager.IndexToNode(from_index)
        to_node   = manager.IndexToNode(to_index)
        travel_time = adjusted_time_matrix_list[from_node][to_node]
        # sumar tiempo de atención en el nodo de origen
        return travel_time + service_times[from_node]

    def distance_callback(from_index, to_index):
        return distance_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

    # Callback para tiempo ajustado con penalización por prioridad
    def time_with_priority_callback(from_index, to_index) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node   = manager.IndexToNode(to_index)
        base_tt   = adjusted_time_matrix_list[from_node][to_node] + service_times[from_node]
        priority  = prio_map.get(request.locations[to_node].priority, 0.0)
        penalty   = int(priority * PENALTY_FACTOR)
        return base_tt + penalty

    distance_callback_index = routing.RegisterTransitCallback(distance_callback)
    #time_prio_cb_index  = routing.RegisterTransitCallback(time_with_priority_callback)
    time_callback_index     = routing.RegisterTransitCallback(time_prio_lookup)

    '''if request.optimize_by.lower() == "distance":
        routing.SetArcCostEvaluatorOfAllVehicles(distance_callback_index)
    else:
        routing.SetArcCostEvaluatorOfAllVehicles(time_callback_index)'''

    if request.optimize_by.lower() == "distance":
        routing.SetArcCostEvaluatorOfAllVehicles(distance_callback_index)
    else:
        routing.SetArcCostEvaluatorOfAllVehicles(time_callback_index)

    time_name = "Time"
    horizon = 500_000
    slack_max = 1800    # tu slack original
    routing.AddDimension(time_callback_index, slack_max, horizon, True, time_name)
    #time_dimension = routing.GetMutableDimension(time_name)
    time_dimension = routing.GetDimensionOrDie(time_name)

    # Definimos las ventanas por prioridad (inician en 0 siempre, y terminan antes según nivel)

    for i, loc in enumerate(request.locations):
        if loc.time_window:
            # impongo la ventana sobre el nodo real, no sobre End(0)
            node_index = manager.NodeToIndex(i)
            start, end = loc.time_window
            time_dimension.CumulVar(node_index).SetRange(start, end)

    for idx in range(routing.Size()):
        routing.AddVariableMinimizedByFinalizer(
            time_dimension.SlackVar(idx))

    # 2.2. Callback para la demanda (peso/volumen)
    demands = [loc.demand for loc in request.locations]
    def demand_callback(from_index):
        node = manager.IndexToNode(from_index)
        return demands[node]

    #demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)

    # Agregar dimensión de "Capacidad" usando como frontera la lista de capacidades de cada camión
    '''capacities = [int(truck.capacity) for truck in request.trucks]
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,               # no hay slack
        capacities,      # capacidad máxima para cada vehículo
        True,            # nódos de “inicio” (depot) tienen carga=0
        "Capacity"
    )'''
    # ------------------------------------------------------------------------------------

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.log_search = True
    search_parameters.time_limit.seconds = 30

    try:
        solution = routing.SolveWithParameters(search_parameters)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=400, detail=f"Error interno durante la optimización: {str(e)}")

    if not solution:
        raise HTTPException(status_code=400, detail="No se encontró una solución factible.")

    # Recorrer cada vehículo para armar su ruta
    all_routes = []
    eta_results = []
    for veh_id in range(num_vehicles):
        index = routing.Start(veh_id)
        route = []
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            arrival = solution.Min(time_dimension.CumulVar(index))
            eta_results.append({
                "vehicle_id": request.trucks[veh_id].id,
                "point": node,
                "point_id": location_ids[node],
                "eta_seconds": arrival
            })
            route.append(node)
            index = solution.Value(routing.NextVar(index))
        # último nodo
        node = manager.IndexToNode(index)
        arrival = solution.Min(time_dimension.CumulVar(index))
        eta_results.append({
            "vehicle_id": request.trucks[veh_id].id,
            "point": node,
            "point_id": location_ids[node],
            "eta_seconds": arrival
        })
        route.append(node)
        all_routes.append({"vehicle_id": request.trucks[veh_id].id, "route_nodes": route})

    # Supongamos que, antes de este bloque, ya has resuelto el VRP y tienes:
    #   all_routes: List[ {"vehicle_id": str, "route_nodes": List[int]} ]
    #   eta_results: List[ { "vehicle_id": str, "point": int, "point_id": str, "eta_seconds": int } ]
    #   coordinates: List[[lon, lat]]  (todas las coords originales)
    #   location_ids: List[str]        (los IDs originales de cada ubicación)
    #   start_time_dt: datetime        (el datetime de inicio)

    # 1. Formatear cada ETA (añadir un campo "eta_formatted" con ISO)
    for item in eta_results:
        eta_dt = start_time_dt + timedelta(seconds=item["eta_seconds"])
        item["eta_formatted"] = eta_dt.isoformat()

    # Ahora, en lugar de trabajar con un solo 'optimized_route', iteramos sobre 'all_routes'.
    # Vamos a construir, para cada vehículo:
    #   - Una lista de IDs de ubicaciones en orden optimizado (route_ids)
    #   - Su geometría completa (route_geometry)
    #   - Sus segmentos individuales (route_segments)
    #   - Su distancia y duración totales (vehicle_distance, vehicle_duration)

    route_url = "http://localhost:8080/ors/v2/directions/driving-hgv"
    common_options = common_truck_opts  # el perfil de restricción que ya definiste

    # Aquí crearemos listas que contengan la información por cada camión:
    all_route_infos = []   # contendrá un dict por cada vehículo
    sum_total_distance = 0
    max_total_duration = 0

    for veh_route in all_routes:
        veh_id = veh_route["vehicle_id"]
        node_sequence = veh_route["route_nodes"]  # p. ej. [0, 3, 7, 0]

        # 2.1. Construir optimized_route_ids para este vehículo
        optimized_route_ids = [location_ids[i] for i in node_sequence]

        # 2.2. Llamar a ORS para obtener la geometría completa de esa ruta
        optimized_locations = [coordinates[i] for i in node_sequence]
        directions_body = {
            "coordinates": optimized_locations,
            "preference": preference,
            "options": common_options,
        }
        resp = requests.post(route_url, json=directions_body, headers=headers)
        if resp.status_code != 200:
            raise HTTPException(
                status_code=resp.status_code,
                detail=f"Error al obtener dirección de ruta para vehículo {veh_id}"
            )
        data = resp.json()
        route_geometry = data["routes"][0]["geometry"]
        summary = data["routes"][0].get("summary", {})
        vehicle_distance = summary.get("distance", 0)    # metros
        vehicle_duration = summary.get("duration", 0)    # segundos

        # 2.3. Desglosar 'route_segments' internamente
        vehicle_segments = []
        for i in range(len(node_sequence) - 1):
            pt1 = coordinates[node_sequence[i]]
            pt2 = coordinates[node_sequence[i + 1]]
            body_seg = {
                "coordinates": [pt1, pt2],
                "profile": "driving-hgv",
                "format": "json",
                "preference": preference,
                "options": common_options,
            }
            seg_resp = requests.post(route_url, json=body_seg, headers=headers)
            if seg_resp.status_code != 200:
                # Si falla un segment, podrías decidir omitirlo o lanzar excepción:
                raise HTTPException(
                    status_code=seg_resp.status_code,
                    detail=f"Error al obtener segmento {i} para vehículo {veh_id}"
                )
            seg_data = seg_resp.json()
            encoded_poly = seg_data["routes"][0]["geometry"]
            vehicle_segments.append({
                "segment_index": i,
                "from_index": node_sequence[i],
                "to_index": node_sequence[i + 1],
                "polyline": encoded_poly,
            })

        # 2.4. Acumular distancias y duraciones para métricas globales
        sum_total_distance += vehicle_distance
        max_total_duration = max(max_total_duration, vehicle_duration)

        # 2.5. Guardamos todo en un dict para este vehículo
        all_route_infos.append({
            "vehicle_id": veh_id,
            "route_nodes": node_sequence,
            "route_location_ids": optimized_route_ids,
            "route_geometry": route_geometry,
            "route_segments": vehicle_segments,
            "vehicle_distance": vehicle_distance,
            "vehicle_duration": vehicle_duration,
        })

    # 3. Finalmente, devolvemos la respuesta completa.
    # En lugar de un único optimized_route, retornamos la lista 'all_route_infos'
    return {
        "locations": [loc.dict() for loc in request.locations],
        "routes": all_route_infos,           # Lista con la info de cada vehículo
        "etas": eta_results,
        "total_distance": sum_total_distance,
        "total_duration": max_total_duration,
        "truck_parameters": [truck.dict() for truck in request.trucks],
    }

def merge_responses(responses: list[RouteResponse]) -> RouteResponse:
    """
    Combina múltiples RouteResponse en una sola.
    """
    merged_locations, merged_routes, merged_etas, merged_trucks = [], [], [], []
    total_distance, total_duration = 0.0, 0.0
    for r in responses:
        merged_locations.extend(r.locations)
        merged_routes.extend(r.routes)
        merged_etas.extend(r.etas)
        merged_trucks.extend(r.truck_parameters)
        total_distance += r.total_distance
        total_duration = max(total_duration, r.total_duration)
    return RouteResponse(
        locations=merged_locations,
        routes=merged_routes,
        etas=merged_etas,
        total_distance=total_distance,
        total_duration=total_duration,
        truck_parameters=merged_trucks
    )

def optimize_with_kmeans_capacity(request: RouteRequest) -> RouteResponse:
    """Agrupa ubicaciones por demanda y optimiza cada cluster por separado."""

    # -----------------------------------------------------------
    # 0. Preprocesamiento DBSCAN (Opción A)
    # -----------------------------------------------------------
    if getattr(request, 'use_dbscan', False):
        # Preparamos coordenadas en radianes
        coords = np.array([[loc.coordinates[1], loc.coordinates[0]]
                           for loc in request.locations], dtype=float)
        coords_rad = np.radians(coords)
        model = DBSCAN(eps=request.dbscan_eps, min_samples=request.dbscan_min_samples, metric='haversine')
        labels = model.fit_predict(coords_rad)
        # Agrupamos índices y ruido
        db_clusters: dict[int, list[int]] = {}
        noise_idx: list[int] = []
        for idx, lbl in enumerate(labels):
            if lbl == -1:
                noise_idx.append(idx)
            else:
                db_clusters.setdefault(lbl, []).append(idx)
        # Llamadas recursivas por cada cluster y ruido
        responses: list[RouteResponse] = []
        for indices in db_clusters.values():
            sub_req = request.copy(update={
                'locations': [request.locations[i] for i in indices],
                'use_dbscan': False
            })
            responses.append(optimize_with_kmeans_capacity(sub_req))
        if noise_idx:
            sub_req = request.copy(update={
                'locations': [request.locations[i] for i in noise_idx],
                'use_dbscan': False
            })
            responses.append(optimize_with_kmeans_capacity(sub_req))
        return merge_responses(responses)

    # -----------------------------------------------------------
    # 1. Calcular la “demanda volumétrica” para cada Location
    # -----------------------------------------------------------
    for loc in request.locations:
        # 1.a. Peso físico:
        peso_real = loc.weight_kg
        # 1.b. Peso volumétrico equivalente:
        peso_volum = loc.volume_m3 * FACTOR_VOLUM_TO_KG
        # 1.c. Asignamos el demand como el mayor:
        loc.demand = max(peso_real, peso_volum)

    # -----------------------------------------------------------
    # 2. Ahora “loc.demand” ya está configurado. Seguimos con K-Means
    # -----------------------------------------------------------
    n = len(request.locations)
    k       = len(request.trucks)
    num_camiones = len(request.trucks)
    if n == 0 or num_camiones == 0:
        raise ValueError("Debe haber al menos una ubicación y un camión para optimizar.")

    # 2.a. Hacemos K-Means sobre las coordenadas (solo geografía):
    coords = np.array([loc.coordinates for loc in request.locations], dtype=float)
    demands = np.array([loc.demand      for loc in request.locations], dtype=float)
    #prio_map = {"baja": 0.0, "media": 0.5, "alta": 1.0}
    #priorities = np.array([prio_map[loc.priority] for loc in request.locations], dtype=float)

    # Normalización
    lon, lat = coords[:,0], coords[:,1]
    lon_n    = (lon - lon.min()) / (lon.max() - lon.min())
    lat_n    = (lat - lat.min()) / (lat.max() - lat.min())
    dem_n    = (demands - demands.min()) / (demands.max() - demands.min())
    '''# Normalización segura
    if priorities.max() == priorities.min():
        prio_n = np.zeros_like(priorities)
    else:
        prio_n = (priorities - priorities.min()) / (priorities.max() - priorities.min())

    # Calcular varianzas para ponderar prioridades
    var_spatial = np.var(lon_n) + np.var(lat_n)
    var_demand  = np.var(dem_n)
    var_prio    = np.var(prio_n)

    # Si no hay variación en prioridad, no la ponderamos
    if var_prio < 1e-9:
        W_prio = 0.0
    else:
        W_prio = (var_spatial + var_demand) / var_prio

    # Construimos la última dimensión sin riesgo de NaN
    prio_feat = prio_n * W_prio
    #X = np.stack([lon_n, lat_n, dem_n, prio_feat], axis=1)'''

    X = np.stack([lon_n, lat_n, dem_n], axis=1)  # (n,3)

    km     = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(X)

    # 2.b. Agrupar índices de ubicaciones por cluster_id
    prelim_clusters = {i: [] for i in range(k)}
    for idx, lbl in enumerate(labels):
        prelim_clusters[lbl].append(idx)

    print("=== CLUSTERS PRELIMINARES ===")
    for cid, indices in prelim_clusters.items():
        suma = sum(request.locations[i].demand for i in indices)
        print(f"  * cluster {cid}: índices {indices}, demanda total = {suma:.1f}  (capacidad camión {request.trucks[cid].capacity})")

    # --- 4) Dividir recursivamente si excede capacidad ---
    def split_if_over_capacity(idxs, cap):
        total = sum(request.locations[i].demand for i in idxs)
        # Base: si solo 1 o cabe, devolvemos tal cual
        if len(idxs) <= 1 or total <= cap:
            return [idxs]
        # Si no cabe y hay ≥2, subdividimos en 2:
        subcoords = np.array([request.locations[i].coordinates for i in idxs], dtype=float)
        sub_labels = KMeans(n_clusters=2, random_state=0).fit_predict(subcoords)
        A = [idxs[i] for i in range(len(idxs)) if sub_labels[i] == 0]
        B = [idxs[i] for i in range(len(idxs)) if sub_labels[i] == 1]
        return split_if_over_capacity(A, cap) + split_if_over_capacity(B, cap)

    # -----------------------------------------------------------
    # 3. Dividir recursivamente si un cluster excede capacidad
    # -----------------------------------------------------------
    def split_cluster_if_overflow(cluster_indices: list[int], truck_capacity: float) -> list[list[int]]:
        """
        Si la suma de loc.demand en cluster_indices > truck_capacity,
        dividimos en 2 subclusters con KMeans y llamamos recursivamente.
        """
        total_demand = sum(request.locations[i].demand for i in cluster_indices)
        if total_demand <= truck_capacity:
            return [cluster_indices]

        # Obtenemos coordenadas solo de ese cluster
        sub_coords = np.array([request.locations[i].coordinates for i in cluster_indices], dtype=float)
        km2 = KMeans(n_clusters=2, random_state=0)
        labels2 = km2.fit_predict(sub_coords)

        cluster_A = [cluster_indices[i] for i in range(len(cluster_indices)) if labels2[i] == 0]
        cluster_B = [cluster_indices[i] for i in range(len(cluster_indices)) if labels2[i] == 1]

        return split_cluster_if_overflow(cluster_A, truck_capacity) + split_cluster_if_overflow(cluster_B, truck_capacity)

    # 3.a. Para cada cluster preliminar, si su suma de demanda > capacidad, subdividir:
    final_clusters: list[list[int]] = []
    for cid, indices in prelim_clusters.items():
        cap_camion = request.trucks[cid].capacity
        sub_clusts = split_if_over_capacity(indices, cap_camion)
        final_clusters.extend(sub_clusts)

    # Calculamos la capacidad máxima de todos los camiones
    max_cap = max(truck.capacity for truck in request.trucks)

    # Recorremos cada cluster y validamos
    for cluster in final_clusters:
        total = sum(request.locations[i].demand for i in cluster)
        if total > max_cap:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Cluster con índices {cluster} tiene demanda total={total:.1f} kg, "
                    f"que excede la capacidad máxima de cualquier camión ({max_cap} kg)."
                )
            )

    print("=== FINAL_CLUSTERS (DESPUÉS DE SPLIT) ===")
    for i, indices in enumerate(final_clusters):
        suma_i = sum(request.locations[j].demand for j in indices)
        print(f"  * final_cluster {i}: índices {indices}, demanda total = {suma_i:.1f}")

    # -----------------------------------------------------------
    # 4. Ajustar número de camiones si hay más clusters que trucks
    # -----------------------------------------------------------
    # Si “final_clusters” es mayor que “num_camiones”, duplicamos el último camión tantas veces como haga falta.
    if len(final_clusters) > num_camiones:
        extra = len(final_clusters) - num_camiones
        ultimo_camion = request.trucks[-1]
        for _ in range(extra):
            # Duplica el último camión – en la práctica quizá quieras crear un “camión virtual” distinto
            request.trucks.append(type(ultimo_camion)(**ultimo_camion.dict()))

    # -----------------------------------------------------------
    # 5. Setup para Acceder a ORS
    # -----------------------------------------------------------
    route_url = "http://localhost:8080/ors/v2/directions/driving-hgv"
    first_truck = request.trucks[0]
    common_truck_opts = {
        "vehicle_type": "delivery",
        "profile_params": {
            "restrictions": {
                "axleload": first_truck.axleload,
                "height": first_truck.height,
                "length": first_truck.length,
                "weight": first_truck.weight,
                "width": first_truck.width,
                "hazmat": first_truck.hazmat,
            }
        }
    }
    if request.avoid_polygons:
        common_truck_opts["avoid_polygons"] = {
            "type": "MultiPolygon",
            "coordinates": request.avoid_polygons
        }
    headers = {"Content-Type": "application/json"}
    preference = "recommended" if request.optimize_by.lower() == "time" else "shortest"

    # -----------------------------------------------------------
    # 6. Para cada subcluster final, resolvemos (o creamos ruta trivial si solo 1 punto)
    # -----------------------------------------------------------
    merged_locations: list[dict] = []
    merged_routes: list[dict] = []
    merged_etas: list[dict] = []
    total_distance = 0.0
    total_duration = 0.0
    merged_truck_params: list[dict] = []

    for i, indices in enumerate(final_clusters):
        if not indices:
            continue

        # 1. Índices locales de este subcluster **más** el depósito
        orig_depot_idx = request.depot_index
        # Asegúrate de que el depósito no esté repetido
        sub_indices = [orig_depot_idx] + [j for j in indices if j != orig_depot_idx]

        sub_truck = request.trucks[i]
        sub_locs = [request.locations[idx] for idx in sub_indices]

        sub_depot_index = 0

        # Recalcular sub_depot_index (igual que antes)…
        orig_depot_idx = request.depot_index
        if orig_depot_idx in indices:
            sub_depot_index = indices.index(orig_depot_idx)
        else:
            depot_coord = request.locations[orig_depot_idx].coordinates
            dist_to_depot = [
                np.hypot(loc.coordinates[0] - depot_coord[0], loc.coordinates[1] - depot_coord[1])
                for loc in sub_locs
            ]
            sub_depot_index = int(np.argmin(dist_to_depot))

        # — CASO: Cluster de tamaño 1 →
        if len(sub_locs) == 1:
            lon_dep, lat_dep = request.locations[orig_depot_idx].coordinates
            lon_pt, lat_pt = sub_locs[0].coordinates

            if not request.round_trip:
                # SOLO IDA (depot -> punto)
                body_ida = {
                    "coordinates": [
                        [lon_dep, lat_dep],
                        [lon_pt, lat_pt]
                    ],
                    "preference": preference,
                    "options": common_truck_opts
                }
                resp_ida = requests.post(route_url, json=body_ida, headers=headers)
                if resp_ida.status_code != 200:
                    raise HTTPException(
                        status_code=resp_ida.status_code,
                        detail=f"Error ORS cluster {i} (ida solamente): {resp_ida.text}"
                    )
                data_ida = resp_ida.json()
                geom = data_ida["routes"][0]["geometry"]
                summary = data_ida["routes"][0].get("summary", {})
                dist_ida = summary.get("distance", 0)
                dur_ida = summary.get("duration", 0)

                # Un solo segmento
                segment_list = [{
                    "segment_index": 0,
                    "from_index": orig_depot_idx,
                    "to_index": indices[0],
                    "polyline": geom
                }]

                ruta_simple = {
                    "vehicle_id": sub_truck.id,
                    "route_nodes": [orig_depot_idx, indices[0]],
                    "route_location_ids": [
                        request.locations[orig_depot_idx].id,
                        sub_locs[0].id
                    ],
                    "route_geometry": geom,
                    "route_segments": segment_list,
                    "vehicle_distance": dist_ida,
                    "vehicle_duration": dur_ida
                }
                merged_routes.append(ruta_simple)

                # ETA
                eta_simple = {
                    "vehicle_id": sub_truck.id,
                    "point": indices[0],
                    "point_id": sub_locs[0].id,
                    "eta_seconds": int(round(dur_ida))
                }
                eta_dt = request.start_datetime + timedelta(seconds=eta_simple["eta_seconds"])
                eta_simple["eta_formatted"] = eta_dt.isoformat()
                merged_etas.append(eta_simple)

                total_distance += dist_ida
                total_duration = max(total_duration, dur_ida)
                merged_truck_params.append(sub_truck.dict())
                continue

            else:
                # IDA Y VUELTA (depot -> punto -> depot)
                body = {
                    "coordinates": [
                        [lon_dep, lat_dep],
                        [lon_pt, lat_pt],
                        [lon_dep, lat_dep]
                    ],
                    "preference": preference,
                    "options": common_truck_opts
                }
                resp = requests.post(route_url, json=body, headers=headers)
                if resp.status_code != 200:
                    raise HTTPException(
                        status_code=resp.status_code,
                        detail=f"Error ORS cluster {i} (ida/vuelta): {resp.text}"
                    )
                data = resp.json()
                geom = data["routes"][0]["geometry"]
                summary = data["routes"][0].get("summary", {})
                dist_total = summary.get("distance", 0)
                dur_total = summary.get("duration", 0)

                # Dos segmentos para el ida y vuelta
                segs = []
                # tramo 0: depot->punto
                body0 = {
                    "coordinates": [
                        [lon_dep, lat_dep],
                        [lon_pt, lat_pt]
                    ],
                    "preference": preference,
                    "options": common_truck_opts
                }
                r0 = requests.post(route_url, json=body0, headers=headers).json()
                poly0 = r0["routes"][0]["geometry"]
                segs.append({
                    "segment_index": 0,
                    "from_index": orig_depot_idx,
                    "to_index": indices[0],
                    "polyline": poly0
                })
                # tramo 1: punto->depot
                body1 = {
                    "coordinates": [
                        [lon_pt, lat_pt],
                        [lon_dep, lat_dep]
                    ],
                    "preference": preference,
                    "options": common_truck_opts
                }
                r1 = requests.post(route_url, json=body1, headers=headers).json()
                poly1 = r1["routes"][0]["geometry"]
                segs.append({
                    "segment_index": 1,
                    "from_index": indices[0],
                    "to_index": orig_depot_idx,
                    "polyline": poly1
                })

                ruta_idayvuelta = {
                    "vehicle_id": sub_truck.id,
                    "route_nodes": [orig_depot_idx, indices[0], orig_depot_idx],
                    "route_location_ids": [
                        request.locations[orig_depot_idx].id,
                        sub_locs[0].id,
                        request.locations[orig_depot_idx].id
                    ],
                    "route_geometry": geom,
                    "route_segments": segs,
                    "vehicle_distance": dist_total,
                    "vehicle_duration": dur_total
                }
                merged_routes.append(ruta_idayvuelta)

                eta_simple = {
                    "vehicle_id": sub_truck.id,
                    "point": indices[0],
                    "point_id": sub_locs[0].id,
                    "eta_seconds": int(round(dur_total / 2))
                }
                eta_dt = request.start_datetime + timedelta(seconds=eta_simple["eta_seconds"])
                eta_simple["eta_formatted"] = eta_dt.isoformat()
                merged_etas.append(eta_simple)

                total_distance += dist_total
                total_duration = max(total_duration, dur_total)
                merged_truck_params.append(sub_truck.dict())
                continue

        # — si el cluster tiene >1 puntos, delegamos a optimize_route_logic como antes —
        mini_req = RouteRequest(
            locations=sub_locs,
            start_datetime=request.start_datetime,
            trucks=[sub_truck],
            round_trip=request.round_trip,
            optimize_by=request.optimize_by,
            depot_index=sub_depot_index,
            avoid_polygons=request.avoid_polygons
        )
        try:
            #print(f"Cluster {i}")
            #print en json mini_req
            #print(f"  * mini_req: {mini_req}")
            #print(f"Fin Cluster {i}")
            sub_resp = optimize_route_logic(mini_req)
        except Exception as e:
            print(f"[Aviso] Cluster {i} falló al optimizar: {e}")
            #print en json mini_req
            print(f"  * mini_req: {mini_req}")
            continue

        # Combinar rutas (fold local→global)…
        for ruta in sub_resp["routes"]:
            nodes_local  = ruta["route_nodes"]              # e.g. [0,2,3,0]
            nodes_global = [ sub_indices[n] for n in nodes_local ]
            rd = ruta.copy()
            rd["route_nodes"] = nodes_global
            merged_routes.append(rd)

        for eta in sub_resp["etas"]:
            ed = eta.copy()
            # y aquí aplicas el mismo sub_indices que para las rutas:
            ed["point"] = sub_indices[ed["point"]]
            # (si usas point_id y quieres mantenerlo, no hace falta cambiarlo,
            #  porque point_id llevaba el ID original de location)
            merged_etas.append(ed)

        for loc in sub_locs:
            merged_locations.append(loc.dict())

        total_distance += sub_resp.get("total_distance", 0.0)
        total_duration = max(total_duration, sub_resp.get("total_duration", 0.0))
        for tp in sub_resp.get("truck_parameters", []):
            merged_truck_params.append(tp)

    # -----------------------------------------------------------
    # 7. Armar la respuesta global
    # -----------------------------------------------------------
    final_response = RouteResponse(
        locations=[Location(**ld) for ld in merged_locations],
        routes=[RouteInfo(**r) for r in merged_routes],
        etas=[EtaItem(**e) for e in merged_etas],
        total_distance=total_distance,
        total_duration=total_duration,
        truck_parameters=[TruckResponse(**tp) for tp in merged_truck_params]
    )
    return final_response
