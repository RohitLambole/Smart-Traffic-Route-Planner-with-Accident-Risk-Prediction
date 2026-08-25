"""
Graph module — now backed by OSM via osmnx. By default the module will build a graph for Pune, India.

This replaces the previous fixed demo graph. For deployments you can change the place
by setting the environment variable OSM_PLACE or by importing osm_graph.build_graph directly.
"""

import os
from typing import Tuple, Dict, Any

# Attempt to build OSM graph at import time using the configured place (defaults to Pune, India)
OSM_PLACE = os.environ.get('OSM_PLACE', 'Pune, India')

try:
    from .osm_graph import build_graph

    city_graph, node_positions, edge_info = build_graph(place=OSM_PLACE)
except Exception as e:
    # If building the OSM graph fails at import, raise a clear error to help debugging during deploy.
    raise RuntimeError(f"Failed to build OSM graph for place '{OSM_PLACE}': {e}")
