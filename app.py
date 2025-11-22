import networkx as nx
from flask import Flask, jsonify, request, send_from_directory
import json
import math
from flask_cors import CORS
import osmnx as ox
import random 
import numpy as np 

# --- Configuración de la Aplicación Flask ---
app = Flask(__name__)
CORS(app) 

# --- Constantes Globales y Configuración del Modelo Real ---
PLACE_NAME = [
    "Miraflores, Lima, Peru",
    "San Isidro, Lima, Peru",
    "Barranco, Lima, Peru", 
    "Surquillo, Lima, Peru"
]
AVERAGE_SPEED_KMPH = 30 
TEMPORARY_ORIGIN_ID = -1 

# --- Funciones de Utilidad ---

def haversine(lat1, lon1, lat2, lon2):
    """Calcula la distancia en kilómetros entre dos puntos geográficos."""
    R = 6371  
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = math.sin(dLat / 2) * math.sin(dLat / 2) + \
        math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * \
        math.sin(dLon / 2) * math.sin(dLon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def calculate_penalty(patients, capacity, alpha=20, beta=10):
    """Calcula la penalización por congestión hospitalaria."""
    if patients >= capacity:
        return 1000000 
    occupancy_ratio = patients / capacity
    return alpha * (occupancy_ratio ** beta)

# --- MODELO DE DATOS REAL (Usando OSMnx) ---

def initialize_real_graph(place_name):
    print(f"Descargando red vial de {place_name}...")
    
    try:
        G_osm = ox.graph_from_place(place_name, network_type="drive", simplify=False)
    except Exception as e:
        print(f"Error al descargar el grafo de OSMnx: {e}")
        return None

    # --- LIMPIEZA ---
    G_osm.remove_nodes_from(list(nx.isolates(G_osm)))
    G_osm.remove_edges_from(list(nx.selfloop_edges(G_osm)))
    
    # Eliminar nodos sin coordenadas válidas
    nodes_to_remove = [
        n for n, data in G_osm.nodes(data=True) 
        if 'x' not in data or 'y' not in data or np.isnan(data.get('x', 0)) or np.isnan(data.get('y', 0))
    ]
    G_osm.remove_nodes_from(nodes_to_remove)
    
    # Asignar velocidades y tiempos
    G_osm = ox.add_edge_speeds(G_osm, fallback=30) 
    G_osm = ox.add_edge_travel_times(G_osm) 
    
    # Procesar pesos y posiciones
    for u, v, k, data in G_osm.edges(keys=True, data=True):
        if 'travel_time' in data:
            data['weight'] = data['travel_time'] / 60 
        else:
            data['weight'] = 1.0 
        
    for node_id, data in G_osm.nodes(data=True):
        data['pos'] = [data['y'], data['x']] 
        
    print(f"Grafo listo: {G_osm.number_of_nodes()} nodos.")
    
    # Integrar Hospitales
    tags = {"amenity": "hospital"}
    try:
        hospitals = ox.features_from_place(place_name, tags)
    except Exception as e:
        hospitals = None

    hospital_nodes = []
    
    if hospitals is not None and not hospitals.empty:
        for idx, row in hospitals.iterrows():
            if row['geometry'].geom_type == 'Point':
                lat, lon = row['geometry'].y, row['geometry'].x
            else:
                lat, lon = row['geometry'].centroid.y, row['geometry'].centroid.x

            try:
                nearest_node = ox.nearest_nodes(G_osm, lon, lat)
                
                G_osm.nodes[nearest_node]['type'] = 'hospital'
                G_osm.nodes[nearest_node]['name'] = row.get('name', f"Hospital_{nearest_node}")
                
                capacity = random.randint(80, 150)
                patients = int(capacity * random.uniform(0.60, 1.00)) 
                G_osm.nodes[nearest_node]['patients'] = patients
                G_osm.nodes[nearest_node]['capacity'] = capacity
                
                hospital_nodes.append(nearest_node)
            except Exception:
                continue # Si falla un hospital, continuar con el siguiente

    print(f"Hospitales integrados: {len(hospital_nodes)}")
    return G_osm

# Inicializar grafo
GLOBAL_REAL_GRAPH = initialize_real_graph(PLACE_NAME)

# --- Rutas de la API ---

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/api/graph')
def get_graph_data():
    if GLOBAL_REAL_GRAPH is None:
        return jsonify({"error": "Error al cargar el grafo real."}), 500

    nodes = []
    for id, data in GLOBAL_REAL_GRAPH.nodes(data=True):
        if 'pos' in data:
            nodes.append({
                'id': id,
                'type': data.get('type', 'intersection'),
                'name': data.get('name', str(id)),
                'pos': data['pos'],
                'patients': data.get('patients'),
                'capacity': data.get('capacity')
            })

    edges = [{"source": u, "target": v} for u, v, d in GLOBAL_REAL_GRAPH.edges(data=True)]
    return jsonify({"nodes": nodes, "edges": edges})


@app.route('/calculate', methods=['POST'])
def calculate_optimal_route():
    if GLOBAL_REAL_GRAPH is None:
        return jsonify({"error": "Grafo no inicializado."}), 500

    data = request.get_json()
    origin_lat = data.get('lat')
    origin_lon = data.get('lon')

    if origin_lat is None or origin_lon is None:
        return jsonify({"error": "Faltan coordenadas"}), 400

    # 1. Crear copia temporal
    G_temp = GLOBAL_REAL_GRAPH.copy()
    
    # --- NUEVO: SIMULACIÓN DINÁMICA ---
    # Re-aleatorizamos la congestión en cada clic para que la demo sea variada
    # Esto simula que el tráfico y la ocupación cambian en tiempo real
    hospitals = [n for n, d in G_temp.nodes(data=True) if d.get('type') == 'hospital']
    for h_id in hospitals:
        capacity = G_temp.nodes[h_id]['capacity']
        # Generar nueva ocupación aleatoria entre 50% y 100%
        new_patients = int(capacity * random.uniform(0.50, 1.00))
        G_temp.nodes[h_id]['patients'] = new_patients
    # ----------------------------------
    
    # 2. Encontrar el nodo más cercano
    try:
        nearest_node_id = ox.nearest_nodes(GLOBAL_REAL_GRAPH, origin_lon, origin_lat)
    except Exception as e:
        return jsonify({"error": f"Error encontrando nodo cercano: {str(e)}"}), 500
    
    # 3. Añadir nodo del accidente
    origin_node_id = TEMPORARY_ORIGIN_ID
    G_temp.add_node(origin_node_id, type="dynamic_origin", name="Accidente", 
                    pos=[origin_lat, origin_lon], x=origin_lon, y=origin_lat)

    # 4. Conectar accidente a la red
    nearest_node_data = G_temp.nodes[nearest_node_id]
    dist_km = haversine(origin_lat, origin_lon, nearest_node_data['pos'][0], nearest_node_data['pos'][1])
    connection_time = (dist_km / AVERAGE_SPEED_KMPH) * 60

    G_temp.add_edge(origin_node_id, nearest_node_id, weight=connection_time)
    G_temp.add_edge(nearest_node_id, origin_node_id, weight=connection_time)

    # 5. Dijkstra + Penalización
    if not hospitals:
         return jsonify({"error": "No hay hospitales en el grafo."}), 404

    best_hospital_id = None
    min_total_cost = float('inf')
    best_path_nodes = []
    best_travel_time = 0
    best_penalty = 0
    
    # Variables para mostrar datos del hospital elegido en el frontend
    chosen_hospital_data = {}

    for hospital_id in hospitals:
        hospital_data = G_temp.nodes[hospital_id]
        try:
            travel_time = nx.dijkstra_path_length(G_temp, source=origin_node_id, target=hospital_id, weight='weight')
            penalty = calculate_penalty(hospital_data['patients'], hospital_data['capacity'])
            total_cost = travel_time + penalty
            
            if total_cost < min_total_cost:
                min_total_cost = total_cost
                best_hospital_id = hospital_id
                best_travel_time = travel_time
                best_penalty = penalty
                best_path_nodes = nx.dijkstra_path(G_temp, source=origin_node_id, target=hospital_id, weight='weight')
                chosen_hospital_data = hospital_data # Guardar datos actuales para retornarlos

        except nx.NetworkXNoPath:
            continue

    if best_hospital_id is None:
        return jsonify({"error": "No se encontró ruta viable a ningún hospital."}), 404

    # Convertir ruta a coordenadas
    path_coords = []
    for node_id in best_path_nodes:
        path_coords.append(G_temp.nodes[node_id]['pos']) 

    return jsonify({
        "best_hospital_id": best_hospital_id,
        "hospital_name": chosen_hospital_data.get('name'),
        "path_coords": path_coords, 
        "travel_time": round(best_travel_time, 2),
        "congestion_penalty": round(best_penalty, 2),
        "total_cost": round(min_total_cost, 2)
    })

if __name__ == '__main__':
    print(f"Iniciando... por favor espere la descarga del grafo.")
    app.run(debug=True, port=5001)