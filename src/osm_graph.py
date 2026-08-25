"""
Build an OSM-based graph using osmnx and export it in the project's adjacency + metadata format.
This module is used as the single graph source for the application (replaces the demo graph).

Usage:
  from osm_graph import build_graph
  city_graph, node_positions, edge_info = build_graph(place="Pune, India")

Notes:
 - Nodes are stringified (str(node_id)) to keep keys JSON-serializable and consistent across the codebase.
 - Edge metadata maps common OSM tags to the model's expected fields: road_type, speed_limit, has_signal, num_lanes.
"""

from typing import Tuple, Dict, Any, Optional
import logging

try:
    import osmnx as ox
    import networkx as nx
except Exception as e:
    ox = None
    nx = None

logger = logging.getLogger(__name__)


def _map_highway_to_road_type(hw: Optional[str]) -> str:
    if hw is None:
        return 'residential'
    hw = str(hw).lower()
    if any(k in hw for k in ['motorway', 'trunk', 'motorway_link']):
        return 'highway'
    if any(k in hw for k in ['primary', 'secondary', 'tertiary', 'primary_link', 'secondary_link']):
        return 'main_road'
    if any(k in hw for k in ['residential', 'living_street', 'service']):
        return 'residential'
    # fallback to lane for tracks, paths and unclassified small ways
    if any(k in hw for k in ['track', 'path', 'footway', 'cycleway']):
        return 'lane'
    return 'residential'


def _parse_maxspeed(maxspeed) -> int:
    # maxspeed can be a list or string like '50 mph' or '50'
    if maxspeed is None:
        return 30
    if isinstance(maxspeed, (list, tuple)):
        maxspeed = maxspeed[0]
    try:
        # remove non-digit
        s = str(maxspeed)
        num = ''.join(ch for ch in s if (ch.isdigit() or ch == '.'))
        return int(float(num))
    except Exception:
        return 30


def _parse_lanes(lanes) -> int:
    if lanes is None:
        return 1
    try:
        if isinstance(lanes, (list, tuple)):
            lanes = lanes[0]
        s = str(lanes)
        num = ''.join(ch for ch in s if ch.isdigit())
        if num == '':
            return 1
        return max(1, int(num))
    except Exception:
        return 1


def _has_signal(edge_data: Dict[str, Any]) -> int:
    # Traffic signals may appear as 'traffic_signals' or 'signal'
    for key in ['traffic_signals', 'signal', 'junction']:
        if key in edge_data:
            val = edge_data.get(key)
            if val in [True, 'yes', 'traffic_signals']:
                return 1
            # if it's a string like 'signal' or 'traffic_signals'
            if isinstance(val, str) and val.lower() in ['traffic_signals', 'signal', 'yes']:
                return 1
    return 0


def build_graph(place: Optional[str] = None,
                bbox: Optional[Tuple[float, float, float, float]] = None,
                network_type: str = 'drive',
                simplify: bool = True,
                max_nodes: int = 5000):
    """
    Build and return (city_graph, node_positions, edge_info) from OSM.

    Parameters:
      - place: place name understood by Nominatim (e.g., 'Pune, India'). If provided, bbox is ignored.
      - bbox: (north, south, east, west) or (north, west, south, east) accepted by osmnx (explicit bbox takes precedence over place if provided).
      - network_type: 'drive' (default), 'walk', etc.
      - simplify: whether to simplify the graph
      - max_nodes: if the downloaded graph has more nodes than this, an exception is raised to avoid huge graphs.

    Returns:
      - city_graph: { node_str: [(neighbor_str, distance_km), ...] }
      - node_positions: { node_str: (lon, lat) }
      - edge_info: { (u_str, v_str): {road_type, speed_limit, has_signal, num_lanes} }
    """
    if ox is None or nx is None:
        raise ImportError("osmnx/networkx are required to build the OSM graph. Install osmnx and its system deps.")

    if not place and not bbox:
        raise ValueError("Either place or bbox must be provided to build the OSM graph.")

    logger.info("Building OSM graph for place=%s bbox=%s network_type=%s", place, bbox, network_type)

    if bbox:
        # osmnx expects north, south, east, west in some variants; allow user to pass (north, south, east, west)
        G = ox.graph_from_bbox(bbox[0], bbox[1], bbox[2], bbox[3], network_type=network_type)
    else:
        G = ox.graph_from_place(place, network_type=network_type)

    if simplify:
        try:
            G = ox.simplify_graph(G)
        except Exception:
            # osmnx simplify changed across versions; ignore if not available
            pass

    if len(G.nodes) > max_nodes:
        raise RuntimeError(f"Extracted graph has too many nodes ({len(G.nodes)}). Increase max_nodes or reduce area.")

    # ensure edge lengths are present
    G = ox.add_edge_lengths(G)

    city_graph = {}
    node_positions = {}
    edge_info = {}

    # nodes: use stringified node ids
    for n, attr in G.nodes(data=True):
        node_id = str(n)
        # osmnx stores x=lon, y=lat
        lon = float(attr.get('x', attr.get('lon', 0.0)))
        lat = float(attr.get('y', attr.get('lat', 0.0)))
        node_positions[node_id] = (lon, lat)
        city_graph[node_id] = []

    # edges: G is a MultiDiGraph; iterate over edges
    for u, v, key, data in G.edges(keys=True, data=True):
        u_s, v_s = str(u), str(v)
        length_m = data.get('length') or data.get('length_m') or 0.0
        distance_km = float(length_m) / 1000.0

        # build adjacency (undirected view by adding both directions)
        city_graph.setdefault(u_s, []).append((v_s, distance_km))
        city_graph.setdefault(v_s, []).append((u_s, distance_km))

        # derive metadata from OSM tags
        highway = data.get('highway')
        road_type = _map_highway_to_road_type(highway)
        speed_limit = _parse_maxspeed(data.get('maxspeed') or data.get('speed_kph'))
        num_lanes = _parse_lanes(data.get('lanes'))
        signal = _has_signal(data)

        # only store one direction (u->v)
        edge_info[(u_s, v_s)] = {
            'road_type': road_type,
            'speed_limit': int(speed_limit),
            'has_signal': int(signal),
            'num_lanes': int(num_lanes),
        }

    logger.info("Built OSM graph with %d nodes and ~%d edges", len(node_positions), sum(len(v) for v in city_graph.values()) // 2)
    return city_graph, node_positions, edge_info
