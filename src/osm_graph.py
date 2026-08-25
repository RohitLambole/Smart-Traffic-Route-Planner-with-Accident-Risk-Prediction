"""
=============================================================
   MODULE 1B: OpenStreetMap (OSM) Road Network Integration
=============================================================
Real-world graph generation using OpenStreetMap data.

Features:
  - Downloads drivable road network for any city using osmnx
  - Builds graph with realistic edge metadata (speed limits, road types)
  - Handles bidirectional and one-way streets
  - Caches downloaded graphs to avoid repeated API calls
  - Provides fallback to synthetic graph if OSM unavailable

Dependencies:
  pip install osmnx networkx geopy

Example:
  graph, positions = load_osm_graph("San Francisco, California")
  # or use synthetic fallback:
  graph, positions = load_osm_graph("San Francisco", use_cache=True)
=============================================================
"""

import os
import json
import math
import networkx as nx
from pathlib import Path
from typing import Dict, Tuple, Optional

try:
    import osmnx as ox
    HAS_OSMNX = True
except ImportError:
    HAS_OSMNX = False
    print("[WARNING] osmnx not installed. Install with: pip install osmnx")


# Cache directory for downloaded graphs
CACHE_DIR = Path(__file__).parent.parent / "data" / "osm_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_cache_path(location: str) -> Path:
    """Get cache file path for a location."""
    safe_name = location.replace(" ", "_").replace(",", "").lower()
    return CACHE_DIR / f"{safe_name}_graph.json"


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate straight-line distance between two lat/lon points in km.
    Used as heuristic for A* when actual road distances aren't available.
    """
    R = 6371  # Earth radius in km
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def get_osm_edge_metadata(edge_data: dict) -> dict:
    """
    Extract metadata from OSM edge attributes.
    
    OSM provides:
      - highway: road type (motorway, primary, secondary, residential, etc.)
      - maxspeed: speed limit (in km/h, may be in various formats)
      - lanes: number of lanes
      - oneway: boolean for one-way streets
      - surface: pavement type
    """
    highway_type = edge_data.get('highway', 'residential')
    if isinstance(highway_type, list):
        highway_type = highway_type[0]  # Handle multi-value tags
    
    # Map OSM highway types to our road categories
    road_type_map = {
        'motorway': 'highway',
        'motorway_link': 'highway',
        'trunk': 'highway',
        'trunk_link': 'highway',
        'primary': 'main_road',
        'primary_link': 'main_road',
        'secondary': 'main_road',
        'secondary_link': 'main_road',
        'tertiary': 'residential',
        'tertiary_link': 'residential',
        'residential': 'residential',
        'living_street': 'residential',
        'unclassified': 'residential',
        'service': 'lane',
        'track': 'lane',
        'path': 'lane',
        'footway': 'lane',
    }
    road_type = road_type_map.get(highway_type, 'residential')
    
    # Extract speed limit (default based on road type)
    maxspeed = edge_data.get('maxspeed', '')
    speed_defaults = {
        'highway': 80,
        'main_road': 60,
        'residential': 30,
        'lane': 20,
    }
    
    speed_limit = speed_defaults[road_type]
    if maxspeed:
        try:
            if isinstance(maxspeed, list):
                maxspeed = maxspeed[0]
            speed_limit = int(str(maxspeed).split()[0])  # Handle "50 mph" format
        except (ValueError, IndexError):
            pass  # Use default
    
    # Lanes
    lanes = 2
    try:
        lanes_val = edge_data.get('lanes', '2')
        lanes = int(str(lanes_val).split(';')[0]) if lanes_val else 2  # Handle "2;2" format
    except (ValueError, TypeError):
        pass
    
    # Traffic signal (look for traffic_signals tag)
    has_signal = 1 if edge_data.get('traffic_signals') else 0
    
    return {
        'road_type': road_type,
        'speed_limit': speed_limit,
        'has_signal': has_signal,
        'num_lanes': max(1, min(lanes, 4)),  # Clamp to 1-4
        'highway_type': highway_type,
        'surface': edge_data.get('surface', 'asphalt'),
    }


def load_osm_graph(
    location: str,
    use_cache: bool = True,
    network_type: str = 'drive',
    simplify: bool = True,
    custom_filter: Optional[str] = None,
) -> Tuple[Dict, Dict]:
    """
    Load or download a road network from OpenStreetMap.
    
    Args:
        location: "City, Country" format (e.g., "San Francisco, California")
        use_cache: If True, use cached graph if available
        network_type: 'drive', 'walk', 'bike', 'all'
        simplify: Simplify network topology (merge parallel edges)
        custom_filter: Custom osmnx filter string
    
    Returns:
        (graph, positions) — adjacency dict and node positions
        graph = {node_id: [(neighbor_id, distance_km), ...]}
        positions = {node_id: (lat, lon)}
    
    Raises:
        ImportError if osmnx not installed
        Exception if location not found on OSM
    """
    if not HAS_OSMNX:
        raise ImportError(
            "osmnx not installed. Install with: pip install osmnx\n"
            "Or use the synthetic graph fallback (see graph.py)"
        )
    
    # Check cache
    cache_path = get_cache_path(location)
    if use_cache and cache_path.exists():
        print(f"[CACHE] Loading graph for '{location}' from disk...")
        return _load_cached_graph(cache_path)
    
    print(f"[OSM] Downloading road network for '{location}'...")
    print(f"     (This may take a minute on first run. Graph will be cached.)")
    
    try:
        # Download the graph
        G = ox.graph_from_place(
            location,
            network_type=network_type,
            simplify=simplify,
            custom_filter=custom_filter
        )
        
        # Convert OSM graph → our format
        graph, positions = _osm_to_routing_graph(G)
        
        # Cache for future use
        if use_cache:
            _save_cached_graph(location, graph, positions, cache_path)
        
        print(f"[OK] Loaded {len(graph)} nodes and {sum(len(v) for v in graph.values())//2} edges")
        return graph, positions
        
    except Exception as e:
        raise Exception(
            f"Failed to load '{location}' from OSM. "
            f"Check spelling and internet connection.\n"
            f"Error: {e}\n"
            f"Fallback: Use the synthetic graph in graph.py"
        )


def _osm_to_routing_graph(G: nx.MultiDiGraph) -> Tuple[Dict, Dict]:
    """
    Convert OSM NetworkX MultiDiGraph to our routing format.
    
    Returns:
        graph: {node_id: [(neighbor_id, distance_km), ...]}
        positions: {node_id: (lat, lon)}
    """
    graph = {}
    positions = {}
    
    # Extract nodes and their positions (lat/lon from OSM)
    for node, data in G.nodes(data=True):
        positions[node] = (data.get('y', 0), data.get('x', 0))  # OSM uses y=lat, x=lon
    
    # Process edges, handling multi-edges (parallel streets) by taking the shortest
    for u, v, key, edge_data in G.edges(keys=True, data=True):
        # Get edge metadata
        metadata = get_osm_edge_metadata(edge_data)
        
        # Calculate distance (OSM provides 'length' in meters)
        length_m = edge_data.get('length', 0)
        if length_m == 0:
            # Fallback: compute haversine distance
            lat1, lon1 = positions[u]
            lat2, lon2 = positions[v]
            distance = haversine_distance(lat1, lon1, lat2, lon2)
        else:
            distance = length_m / 1000  # Convert meters to km
        
        # Ensure distance is reasonable (avoid zero-length edges)
        distance = max(distance, 0.01)
        
        # Add to forward adjacency
        if u not in graph:
            graph[u] = []
        
        # Check if we already have this edge (skip if we do, or keep shorter)
        existing = [e for e in graph[u] if e[0] == v]
        if existing:
            if existing[0][1] > distance:
                graph[u].remove(existing[0])
                graph[u].append((v, distance))
        else:
            graph[u].append((v, distance))
        
        # Add reverse edge if not one-way
        oneway = edge_data.get('oneway', False)
        if not oneway:
            if v not in graph:
                graph[v] = []
            existing = [e for e in graph[v] if e[0] == u]
            if existing:
                if existing[0][1] > distance:
                    graph[v].remove(existing[0])
                    graph[v].append((u, distance))
            else:
                graph[v].append((u, distance))
    
    # Filter out isolated nodes
    graph = {k: v for k, v in graph.items() if v}
    positions = {k: v for k, v in positions.items() if k in graph}
    
    return graph, positions


def _save_cached_graph(location: str, graph: Dict, positions: Dict, cache_path: Path) -> None:
    """Save graph and positions to cache."""
    cache_data = {
        'location': location,
        'graph': graph,
        'positions': positions,
    }
    with open(cache_path, 'w') as f:
        json.dump(cache_data, f, indent=2)
    print(f"[CACHE] Saved to {cache_path}")


def _load_cached_graph(cache_path: Path) -> Tuple[Dict, Dict]:
    """Load graph and positions from cache."""
    with open(cache_path, 'r') as f:
        data = json.load(f)
    
    graph = data['graph']
    positions = data['positions']
    
    # JSON keys are strings; convert node IDs back to integers if needed
    graph = {int(k): [(int(n), d) for n, d in v] for k, v in graph.items()}
    positions = {int(k): v for k, v in positions.items()}
    
    return graph, positions


def edge_info_from_osm(graph: Dict, node_a, node_b, osm_edge_data: dict) -> dict:
    """
    Build edge metadata dict for risk model, given OSM edge data.
    This is the bridge between raw OSM and our risk prediction.
    """
    return get_osm_edge_metadata(osm_edge_data)


# ─────────────────────────────────────────────────────────────────
# DEMO: Download and inspect a real city
# ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    try:
        # Example: San Francisco
        print("=" * 60)
        print("  OpenStreetMap Graph Demo")
        print("=" * 60)
        
        location = "San Francisco, California"
        graph, positions = load_osm_graph(location, use_cache=True)
        
        print(f"\n✓ Successfully loaded {len(graph)} nodes")
        print(f"  Sample nodes: {list(graph.keys())[:5]}")
        print(f"  Sample edges from first node:")
        first_node = list(graph.keys())[0]
        for neighbor, dist in graph[first_node][:3]:
            print(f"    → {neighbor}: {dist:.3f} km")
        
    except ImportError as e:
        print(f"[ERROR] {e}")
        print("\nTo use OpenStreetMap, install osmnx:")
        print("  pip install osmnx")
    except Exception as e:
        print(f"[ERROR] {e}")
