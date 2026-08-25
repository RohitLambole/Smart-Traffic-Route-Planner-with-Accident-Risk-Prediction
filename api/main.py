from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Tuple, Dict, Any
from fastapi.middleware.cors import CORSMiddleware
import sys, os
import logging

# Ensure we can import the project's src package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import graph
import osm_graph
from osm_graph import _cache_key, _save_cache
from risk_routing import load_model, set_conditions, predict_edge_risk
from agent import TrafficAgent

logger = logging.getLogger(__name__)

# --- Pydantic request models
class RouteRequest(BaseModel):
    source: str
    dest: str
    hour: int
    day: int  # 0..6
    weather: str  # "clear"|"rain"|"fog"
    traffic: float  # 0.1..1.0
    risk_threshold: float = 0.7

class GraphBuildRequest(BaseModel):
    place: Optional[str] = None
    # bbox as [north, south, east, west]
    bbox: Optional[List[float]] = None
    network_type: Optional[str] = 'drive'
    max_nodes: Optional[int] = 5000

class GraphUploadRequest(BaseModel):
    # nodes: { id: [lon, lat] }
    nodes: Dict[str, List[float]]
    # edges: [{ u: id, v: id, distance: float, meta: {...} }, ...]
    edges: List[Dict[str, Any]]
    place: Optional[str] = None
    network_type: Optional[str] = 'drive'
    max_nodes: Optional[int] = 5000

# --- FastAPI app
app = FastAPI(title="Smart Traffic Route Planner API")

# Allow CORS from frontends (for demo). Replace '*' with your frontend origin in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# load model once at startup
MODEL = None

@app.on_event("startup")
def startup_event():
    global MODEL
    try:
        MODEL = load_model()
        if not MODEL:
            logger.warning("Model loader returned False or failed to load model files.")
    except Exception:
        logger.exception("Exception while loading model at startup. Continuing without model.")
        MODEL = None


@app.post("/route")
def route(req: RouteRequest):
    """Compute a safe route given source/dest and conditions.

    Reuses existing model loader, risk prediction and TrafficAgent from src/.
    """
    if req.source == req.dest:
        raise HTTPException(status_code=400, detail="source and dest must differ")

    # set environment conditions (mutates module-level state in risk_routing)
    set_conditions(hour=req.hour, day=req.day, weather=req.weather, traffic=req.traffic)

    agent = TrafficAgent(graph.city_graph, graph.node_positions, risk_threshold=req.risk_threshold)
    result = agent.plan_route(req.source, req.dest)
    if result is None:
        raise HTTPException(status_code=404, detail="No path found")

    # Build edge risk list for all graph edges (directed as present in city_graph)
    edge_risks = []
    for u, neighbors in graph.city_graph.items():
        for v, _ in neighbors:
            meta = graph.get_edge_metadata(u, v)
            try:
                r = predict_edge_risk(u, v, meta)
            except Exception:
                logger.exception(f"predict_edge_risk failed for edge {u}-{v}; using fallback 0.5")
                r = 0.5
            edge_risks.append({
                "edge": (u, v),
                "risk": float(r),
                "road_type": meta.get("road_type"),
                "danger": bool(r > req.risk_threshold)
            })

    resp = {
        "path": result["path"],
        "cost": float(result["cost"]),
        "shortest_path": result.get("shortest_path"),
        "shortest_cost": float(result.get("shortest_cost", 0)),
        "action": result["action"],
        "message": result["message"],
        "avoided_edges": result.get("avoided_edges", []),
        "edge_risks": edge_risks,
    }
    return resp


@app.get("/graph")
def graph_endpoint():
    """Return node positions and edges with metadata for frontend mapping.

    The response is a lightweight JSON serializable structure derived from src/graph.py.
    """
    edges = []
    for u, neighbors in graph.city_graph.items():
        for v, dist in neighbors:
            meta = graph.get_edge_metadata(u, v)
            edges.append({
                "u": u,
                "v": v,
                "distance": float(dist),
                "meta": meta,
            })

    return {
        "nodes": {n: (float(x), float(y)) for n, (x, y) in graph.node_positions.items()},
        "edges": edges,
    }


@app.post('/graph/build')
def build_graph_endpoint(req: GraphBuildRequest):
    """Build a constrained OSM graph on demand. Accepts either a place or a bbox.

    bbox should be [north, south, east, west] (four floats). Returns the newly built graph.
    """
    if not req.place and not req.bbox:
        raise HTTPException(status_code=400, detail='place or bbox is required')

    try:
        bbox_tuple = None
        if req.bbox:
            if len(req.bbox) != 4:
                raise HTTPException(status_code=400, detail='bbox must be an array of 4 floats: [north,south,east,west]')
            bbox_tuple = tuple(req.bbox)

        new_city_graph, new_node_positions, new_edge_info = osm_graph.build_graph(
            place=req.place,
            bbox=bbox_tuple,
            network_type=req.network_type or 'drive',
            max_nodes=req.max_nodes or 5000
        )

        # update in-memory graph used by the app
        graph.city_graph = new_city_graph
        graph.node_positions = new_node_positions
        graph.edge_info = new_edge_info

        # return the graph in the same format as GET /graph
        edges = []
        for u, neighbors in graph.city_graph.items():
            for v, dist in neighbors:
                meta = graph.get_edge_metadata(u, v)
                edges.append({"u": u, "v": v, "distance": float(dist), "meta": meta})

        return {
            "nodes": {n: (float(x), float(y)) for n, (x, y) in graph.node_positions.items()},
            "edges": edges,
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to build OSM graph on demand: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post('/graph/upload')
def upload_graph_endpoint(req: GraphUploadRequest):
    """Upload a prebuilt graph JSON (nodes + edges). Saves to cache and updates in-memory graph.

    Expected JSON shape:
      {
        "nodes": {"id": [lon, lat], ...},
        "edges": [{"u":"id","v":"id","distance":0.12,"meta":{...}}, ...],
        "place": "optional place name",
        "network_type": "drive",
        "max_nodes": 2000
      }

    Returns: { cached: <filename>, nodes: N, edges: M }
    """
    if not req.nodes or not req.edges:
        raise HTTPException(status_code=400, detail='nodes and edges are required in upload')

    place = req.place
    network_type = req.network_type or 'drive'
    max_nodes = req.max_nodes or 5000

    # compute cache key and save
    try:
        key = _cache_key(place, None, network_type, max_nodes)
        # prepare edges list for cache writer: ensure meta exists
        edges_for_cache = []
        for e in req.edges:
            u = str(e.get('u'))
            v = str(e.get('v'))
            dist = float(e.get('distance', 0.0))
            meta = e.get('meta', {}) or {}
            edges_for_cache.append({'u': u, 'v': v, 'distance': dist, 'meta': meta})

        # save cache
        _save_cache('data/graphs', key, req.nodes, edges_for_cache)

        # update in-memory graph
        new_city_graph = {}
        new_node_positions = {}
        new_edge_info = {}

        for n, pos in req.nodes.items():
            nid = str(n)
            new_node_positions[nid] = (float(pos[0]), float(pos[1]))
            new_city_graph[nid] = []

        for e in edges_for_cache:
            u, v = str(e['u']), str(e['v'])
            dist = float(e.get('distance', 0.0))
            new_city_graph.setdefault(u, []).append((v, dist))
            new_city_graph.setdefault(v, []).append((u, dist))
            meta = e.get('meta', {})
            new_edge_info[(u, v)] = {
                'road_type': meta.get('road_type', 'residential'),
                'speed_limit': int(meta.get('speed_limit', 30)),
                'has_signal': int(meta.get('has_signal', 0)),
                'num_lanes': int(meta.get('num_lanes', 1)),
            }

        graph.city_graph = new_city_graph
        graph.node_positions = new_node_positions
        graph.edge_info = new_edge_info

        return { 'cached': f"{key}.json", 'nodes': len(new_node_positions), 'edges': len(edges_for_cache) }

    except Exception as exc:
        logger.exception("Failed to upload and cache graph: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
