"""
Build an OSM-based graph using osmnx and export it in the project's adjacency + metadata format.
This module is used as the single graph source for the application (replaces the demo graph).

Caching: built graphs can be cached to disk (JSON) to avoid repeated Overpass requests.

Usage:
  from osm_graph import build_graph
  city_graph, node_positions, edge_info = build_graph(place="Pune, India")

Notes:
 - Nodes are stringified (str(node_id)) to keep keys JSON-serializable and consistent across the codebase.
 - Edge metadata maps common OSM tags to the model's expected fields: road_type, speed_limit, has_signal, num_lanes.
"""

from typing import Tuple, Dict, Any, Optional, List
import logging
import json
import os
from pathlib import Path
from urllib.parse import quote_plus
import hashlib

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
    if maxspeed is None:
        return 30
    if isinstance(maxspeed, (list, tuple)):
        maxspeed = maxspeed[0]
    try:
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
    for key in ['traffic_signals', 'signal', 'junction']:
        if key in edge_data:
            val = edge_data.get(key)
            if val in [True, 'yes', 'traffic_signals']:
                return 1
            if isinstance(val, str) and val.lower() in ['traffic_signals', 'signal', 'yes']:
                return 1
    return 0


def _cache_key(place: Optional[str], bbox: Optional[Tuple[float, float, float, float]], network_type: str, max_nodes: int) -> str:
    if place:
        key = f"place:{place}|network:{network_type}|max:{max_nodes}"
    else:
        bbox_str = ','.join([str(x) for x in bbox])
        key = f"bbox:{bbox_str}|network:{network_type}|max:{max_nodes}"
    # hash for filesystem-safety and reasonable filename length
    h = hashlib.sha1(key.encode('utf-8')).hexdigest()
    return h


def _ensure_cache_dir(cache_dir: str) -> Path:
    p = Path(cache_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save_cache(cache_dir: str, key: str, nodes: Dict[str, Tuple[float, float]], edges: List[Dict[str, Any]]):
    p = _ensure_cache_dir(cache_dir)
    fp = p / f"{key}.json"
    payload = { 'nodes': nodes, 'edges': edges }
    with fp.open('w', encoding='utf-8') as f:
        json.dump(payload, f)
    logger.info("Saved graph cache: %s", fp)


def _load_cache(cache_dir: str, key: str):
    p = Path(cache_dir)
    fp = p / f"{key}.json"
    if not fp.exists():
        return None
    with fp.open('r', encoding='utf-8') as f:
        payload = json.load(f)
    return payload


def build_graph(place: Optional[str] = None,
                bbox: Optional[Tuple[float, float, float, float]] = None,
                network_type: str = 'drive',
                simplify: bool = True,
                max_nodes: int = 5000,
                cache_dir: Optional[str] = 'data/graphs',
                use_cache: bool = True):
    """
    Build and return (city_graph, node_positions, edge_info) from OSM, with optional caching.

    If a cached graph exists for the given place/bbox+params it will be loaded and returned.
    Otherwise the graph is downloaded via osmnx and then saved to cache.
    """
    if use_cache and cache_dir:
        key = _cache_key(place, bbox, network_type, max_nodes)
        cached = _load_cache(cache_dir, key)
        if cached is not None:
            logger.info("Loaded graph from cache for key=%s", key)
            # convert cached JSON to internal structures
            city_graph = {}
            node_positions = {}
            edge_info = {}
            for n, pos in cached.get('nodes', {}).items():
                node_positions[str(n)] = (float(pos[0]), float(pos[1]))
                city_graph[str(n)] = []
            for e in cached.get('edges', []):
                u, v = str(e['u']), str(e['v'])
                dist = float(e.get('distance', 0.0))
                city_graph.setdefault(u, []).append((v, dist))
                city_graph.setdefault(v, []).append((u, dist))
                meta = e.get('meta', {})
                edge_info[(u, v)] = {
                    'road_type': meta.get('road_type', 'residential'),
                    'speed_limit': int(meta.get('speed_limit', 30)),
                    'has_signal': int(meta.get('has_signal', 0)),
                    'num_lanes': int(meta.get('num_lanes', 1)),
                }
            return city_graph, node_positions, edge_info

    if ox is None or nx is None:
        raise ImportError("osmnx/networkx are required to build the OSM graph. Install osmnx and its system deps.")

    if not place and not bbox:
        raise ValueError("Either place or bbox must be provided to build the OSM graph.")

    logger.info("Building OSM graph for place=%s bbox=%s network_type=%s", place, bbox, network_type)

    # configure Overpass endpoint(s) with optional env list
    mirrors = os.environ.get('OSM_OVERPASS_URL', 'https://overpass-api.de/api/interpreter,https://overpass.kumi.systems/api/interpreter,https://lz4.overpass-api.de/api/interpreter').split(',')
    timeout = int(os.environ.get('OSM_OVERPASS_TIMEOUT', '180'))

    G = None
    last_exc = None
    for url in mirrors:
        try:
            ox.settings.overpass_endpoint = url
            ox.settings.timeout = timeout
            if bbox:
                # bbox expected as (north, south, east, west) by our API; osmnx.graph_from_bbox uses north, south, east, west
                G = ox.graph_from_bbox(bbox[0], bbox[1], bbox[2], bbox[3], network_type=network_type)
            else:
                G = ox.graph_from_place(place, network_type=network_type)
            break
        except Exception as e:
            logger.warning("Overpass mirror %s failed: %s", url, e)
            last_exc = e
            G = None
            continue

    if G is None:
        raise RuntimeError(f"All Overpass mirrors failed. Last error: {last_exc}")

    if simplify:
        try:
            G = ox.simplify_graph(G)
        except Exception:
            pass

    if len(G.nodes) > max_nodes:
        raise RuntimeError(f"Extracted graph has too many nodes ({len(G.nodes)}). Increase max_nodes or reduce area.")

    # ensure edge lengths are present
    G = ox.add_edge_lengths(G)

    city_graph = {}
    node_positions = {}
    edge_info = {}
    edges_for_cache = []

    for n, attr in G.nodes(data=True):
        node_id = str(n)
        lon = float(attr.get('x', attr.get('lon', 0.0)))
        lat = float(attr.get('y', attr.get('lat', 0.0)))
        node_positions[node_id] = (lon, lat)
        city_graph[node_id] = []

    for u, v, key, data in G.edges(keys=True, data=True):
        u_s, v_s = str(u), str(v)
        length_m = data.get('length') or data.get('length_m') or 0.0
        distance_km = float(length_m) / 1000.0

        city_graph.setdefault(u_s, []).append((v_s, distance_km))
        city_graph.setdefault(v_s, []).append((u_s, distance_km))

        highway = data.get('highway')
        road_type = _map_highway_to_road_type(highway)
        speed_limit = _parse_maxspeed(data.get('maxspeed') or data.get('speed_kph'))
        num_lanes = _parse_lanes(data.get('lanes'))
        signal = _has_signal(data)

        edge_info[(u_s, v_s)] = {
            'road_type': road_type,
            'speed_limit': int(speed_limit),
            'has_signal': int(signal),
            'num_lanes': int(num_lanes),
        }

        edges_for_cache.append({
            'u': u_s,
            'v': v_s,
            'distance': distance_km,
            'meta': edge_info[(u_s, v_s)],
        })

    # save to cache
    if cache_dir:
        key = _cache_key(place, bbox, network_type, max_nodes)
        try:
            _save_cache(cache_dir, key, node_positions, edges_for_cache)
        except Exception as e:
            logger.warning("Failed to save graph cache: %s", e)

    logger.info("Built OSM graph with %d nodes and ~%d edges", len(node_positions), sum(len(v) for v in city_graph.values()) // 2)
    return city_graph, node_positions, edge_info
