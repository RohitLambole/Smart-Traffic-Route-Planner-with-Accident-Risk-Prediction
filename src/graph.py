"""
Graph module — OSM-backed via osmnx. The module exposes:
- city_graph: adjacency { node_id: [(neighbor_id, distance_km), ...] }
- node_positions: { node_id: (lon, lat) }
- edge_info: { (u, v): {road_type, speed_limit, has_signal, num_lanes} }
- get_edge_metadata(u, v): helper to fetch metadata for an edge (u->v)

The graph is built at import time using the OSM_PLACE environment variable (default: Pune, India).
"""

import os
from typing import Tuple, Dict, Any, Optional

# Attempt to build OSM graph at import time using the configured place (defaults to Pune, India)
OSM_PLACE = os.environ.get('OSM_PLACE', 'Pune, India')

# edge_info will be populated by the OSM graph builder
city_graph: Dict[str, list] = {}
node_positions: Dict[str, Tuple[float, float]] = {}
edge_info: Dict[Tuple[str, str], Dict[str, Any]] = {}

try:
    from .osm_graph import build_graph

    city_graph, node_positions, edge_info = build_graph(place=OSM_PLACE)
except Exception as e:
    # If building the OSM graph fails at import, raise a clear error to help debugging during deploy.
    raise RuntimeError(f"Failed to build OSM graph for place '{OSM_PLACE}': {e}")


def get_edge_metadata(u: str, v: str) -> Dict[str, Any]:
    """Return edge metadata for directed edge u->v. If not found, attempt v->u, then return defaults."""
    key = (str(u), str(v))
    if key in edge_info:
        return edge_info[key]
    key_rev = (str(v), str(u))
    if key_rev in edge_info:
        return edge_info[key_rev]
    # default metadata when missing
    return {
        'road_type': 'residential',
        'speed_limit': 30,
        'has_signal': 0,
        'num_lanes': 1,
    }
