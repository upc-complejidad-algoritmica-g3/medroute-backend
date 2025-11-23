import networkx as nx
from flask import Flask, jsonify, request
import math
from flask_cors import CORS
import osmnx as ox
import random
import numpy as np
import time  # Para medir la complejidad algorítmica

app = Flask(__name__)
CORS(app)

# --- CONFIGURACIÓN ---
# Distritos para tener un grafo denso y realista (San Isidro tiene muchas clínicas)
PLACE_NAME = [
    "Miraflores, Lima, Peru",
    "San Isidro, Lima, Peru",
    "Barranco, Lima, Peru",
    "Surquillo, Lima, Peru"
]
AVERAGE_SPEED_KMPH = 30
TEMPORARY_ORIGIN_ID = -1


# --- FUNCIONES DE UTILIDAD ---
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = math.sin(dLat / 2) * math.sin(dLat / 2) + \
        math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * \
        math.sin(dLon / 2) * math.sin(dLon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def calculate_penalty(patients, capacity, alpha=20, beta=10):
    """Calcula la penalización exponencial por congestión."""
    if patients >= capacity: return 1000000
    occupancy_ratio = patients / capacity
    return alpha * (occupancy_ratio ** beta)


# --- INICIALIZACIÓN DEL GRAFO ---
def initialize_real_graph(place_name):
    print(f"Descargando red vial de {place_name}...")
    try:
        G_osm = ox.graph_from_place(place_name, network_type="drive", simplify=False)
    except Exception as e:
        print(f"Error crítico al descargar grafo: {e}")
        return None

    # 1. Limpieza del Grafo
    G_osm.remove_nodes_from(list(nx.isolates(G_osm)))
    G_osm.remove_edges_from(list(nx.selfloop_edges(G_osm)))

    # Eliminar nodos corruptos (sin coordenadas)
    nodes_to_remove = [n for n, data in G_osm.nodes(data=True)
                       if 'x' not in data or 'y' not in data or np.isnan(data.get('x', 0))]
    G_osm.remove_nodes_from(nodes_to_remove)

    # 2. Asignar Pesos (Tiempo de viaje)
    G_osm = ox.add_edge_speeds(G_osm, fallback=30)
    G_osm = ox.add_edge_travel_times(G_osm)

    for u, v, k, data in G_osm.edges(keys=True, data=True):
        # Convertir segundos a minutos
        data['weight'] = data.get('travel_time', 1.0) / 60

    for node_id, data in G_osm.nodes(data=True):
        data['pos'] = [data['y'], data['x']]

    print(f"Grafo vial listo: {G_osm.number_of_nodes()} nodos.")

    # 3. Cargar Hospitales y Clínicas (Búsqueda Ampliada)
    # Buscamos varias etiquetas para encontrar más centros médicos
    tags = {"amenity": ["hospital", "clinic"], "healthcare": ["hospital", "clinic"]}

    try:
        hospitals = ox.features_from_place(place_name, tags)
    except Exception as e:
        print(f"Advertencia hospitales: {e}")
        hospitals = None

    hospital_nodes = []

    if hospitals is not None and not hospitals.empty:
        # Filtrar lugares sin nombre para limpiar el mapa
        hospitals = hospitals[hospitals['name'].notna()]
        print(f"Establecimientos de salud encontrados: {len(hospitals)}")

        for idx, row in hospitals.iterrows():
            # Obtener coordenadas (Manejo de Polígonos y Puntos)
            if row['geometry'].geom_type == 'Point':
                lat, lon = row['geometry'].y, row['geometry'].x
            else:
                lat, lon = row['geometry'].centroid.y, row['geometry'].centroid.x

            try:
                # Conectar al nodo vial más cercano
                nearest = ox.nearest_nodes(G_osm, lon, lat)

                # Evitar sobrescribir si ya es un hospital
                if G_osm.nodes[nearest].get('type') == 'hospital':
                    continue

                G_osm.nodes[nearest]['type'] = 'hospital'
                # Usar nombre real o genérico
                G_osm.nodes[nearest]['name'] = row.get('name', f"Centro Médico {nearest}")

                # Simulación Inicial de Capacidad
                G_osm.nodes[nearest]['capacity'] = random.randint(80, 150)
                G_osm.nodes[nearest]['patients'] = int(G_osm.nodes[nearest]['capacity'] * random.uniform(0.4, 0.9))

                hospital_nodes.append(nearest)
            except:
                continue

    print(f"Hospitales integrados al grafo: {len(hospital_nodes)}")
    return G_osm


GLOBAL_REAL_GRAPH = initialize_real_graph(PLACE_NAME)


# --- RUTAS API ---

@app.route('/api/graph')
def get_graph_data():
    if GLOBAL_REAL_GRAPH is None: return jsonify({"error": "Error grafo"}), 500

    nodes = []
    # Solo enviamos nodos esenciales (hospitales) para no saturar el frontend
    # Opcional: Si quieres ver las calles, el frontend dibuja las aristas, no necesitamos enviar todos los nodos de cruce
    for id, data in GLOBAL_REAL_GRAPH.nodes(data=True):
        if 'pos' in data and data.get('type') == 'hospital':
            nodes.append({
                'id': id,
                'type': 'hospital',
                'name': data.get('name'),
                'pos': data['pos'],
                'patients': data.get('patients'),
                'capacity': data.get('capacity')
            })
    # Enviamos aristas simplificadas si el frontend las necesita (o podemos omitirlas si solo usamos mapa base)
    # Para rendimiento, en React con Leaflet, es mejor NO dibujar 10,000 líneas si no es necesario.
    # Aquí enviamos una lista vacía de edges para que React no intente dibujar 20k líneas y se cuelgue.
    # El mapa base de OpenStreetMap ya muestra las calles visualmente.
    return jsonify({"nodes": nodes, "edges": []})


@app.route('/calculate', methods=['POST'])
def calculate_optimal_route():
    if GLOBAL_REAL_GRAPH is None: return jsonify({"error": "Grafo no listo"}), 500
    data = request.get_json()

    # 1. Crear copia temporal para simulación dinámica
    G_temp = GLOBAL_REAL_GRAPH.copy()

    # --- SIMULACIÓN TIEMPO REAL: Actualizar congestión aleatoriamente ---
    hospitals = [n for n, d in G_temp.nodes(data=True) if d.get('type') == 'hospital']
    for h in hospitals:
        cap = G_temp.nodes[h]['capacity']
        # Aleatorizar ocupación entre 30% y 100%
        G_temp.nodes[h]['patients'] = int(cap * random.uniform(0.3, 1.0))

    # 2. Insertar Punto de Accidente
    origin_id = TEMPORARY_ORIGIN_ID
    try:
        nearest_node = ox.nearest_nodes(GLOBAL_REAL_GRAPH, data['lon'], data['lat'])
    except Exception as e:
        return jsonify({"error": "Ubicación fuera del rango del mapa"}), 400

    G_temp.add_node(origin_id, pos=[data['lat'], data['lon']], x=data['lon'], y=data['lat'])

    # Conectar accidente a la red vial
    node_data = G_temp.nodes[nearest_node]
    dist_km = haversine(data['lat'], data['lon'], node_data['pos'][0], node_data['pos'][1])
    conn_time = (dist_km / AVERAGE_SPEED_KMPH) * 60

    G_temp.add_edge(origin_id, nearest_node, weight=conn_time)
    G_temp.add_edge(nearest_node, origin_id, weight=conn_time)

    # 3. EJECUTAR DIJKSTRA Y MEDIR TIEMPO
    start_time = time.time()

    best_hospital = None
    min_cost = float('inf')
    best_path = []
    metrics = {}

    for h_id in hospitals:
        h_data = G_temp.nodes[h_id]
        try:
            # Dijkstra: Camino más corto en tiempo
            travel_time = nx.dijkstra_path_length(G_temp, origin_id, h_id, weight='weight')

            # Penalización por congestión
            penalty = calculate_penalty(h_data['patients'], h_data['capacity'])
            total_cost = travel_time + penalty

            if total_cost < min_cost:
                min_cost = total_cost
                best_hospital = h_id
                best_path = nx.dijkstra_path(G_temp, origin_id, h_id, weight='weight')
                metrics = {
                    'travel_time': travel_time,
                    'penalty': penalty,
                    'patients': h_data['patients'],
                    'capacity': h_data['capacity']
                }
        except nx.NetworkXNoPath:
            continue

    end_time = time.time()
    execution_ms = (end_time - start_time) * 1000

    if not best_hospital: return jsonify({"error": "No se encontró ruta viable"}), 404

    # Convertir IDs de nodos a coordenadas lat/lon para el mapa
    path_coords = [G_temp.nodes[n]['pos'] for n in best_path]

    return jsonify({
        "hospital_name": G_temp.nodes[best_hospital]['name'],
        "path_coords": path_coords,
        "travel_time": round(metrics['travel_time'], 2),
        "congestion_penalty": round(metrics['penalty'], 2),
        "total_cost": round(min_cost, 2),
        "algorithm_time_ms": round(execution_ms, 2),
        "hospital_stats": {
            "patients": metrics['patients'],
            "capacity": metrics['capacity']
        },
        # Retornamos la lista actualizada de hospitales para repintar el semáforo en el frontend
        "updated_hospitals": [
            {
                'id': n,
                'patients': G_temp.nodes[n]['patients'],
                'capacity': G_temp.nodes[n]['capacity']
            } for n in hospitals
        ]
    })


if __name__ == '__main__':
    app.run(debug=True, port=5001)