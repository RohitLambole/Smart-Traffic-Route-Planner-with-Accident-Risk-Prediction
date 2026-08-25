from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Tuple, Dict, Any
from fastapi.middleware.cors import CORSMiddleware
import sys, os
import logging

# Ensure we can import the project's src package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from graph import city_graph, node_positions, get_edge_metadata
from risk_routing import load_model, set_conditions, predict_edge_risk
from agent import TrafficAgent

logger = logging.getLogger(__name__)

# --- Pydantic request model
class RouteRequest(BaseModel):
    source: str
    dest: str
    hour: int
    day: int  # 0..6
    weather: str  # "clear"|"rain"|"fog"
    traffic: float  # 0.1..1.0
    risk_threshold: float = 0.7

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

    agent = TrafficAgent(city_graph, node_positions, risk_threshold=req.risk_threshold)
    result = agent.plan_route(req.source, req.dest)
    if result is None:
        raise HTTPException(status_code=404, detail="No path found")

    # Build edge risk list for all graph edges (directed as present in city_graph)
    edge_risks = []
    for u, neighbors in city_graph.items():
        for v, _ in neighbors:
            meta = get_edge_metadata(u, v)
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
def graph():
    """Return node positions and edges with metadata for frontend mapping.

    The response is a lightweight JSON serializable structure derived from src/graph.py.
    """
    edges = []
    for u, neighbors in city_graph.items():
        for v, dist in neighbors:
            meta = get_edge_metadata(u, v)
            edges.append({
                "u": u,
                "v": v,
                "distance": float(dist),
                "meta": meta,
            })

    return {
        "nodes": {n: (float(x), float(y)) for n, (x, y) in node_positions.items()},
        "edges": edges,
    }
