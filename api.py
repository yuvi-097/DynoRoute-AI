"""
PathMind AI - FastAPI Endpoint
=================================
Lightweight REST API for route optimization.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse

from graph.road_network import RoadNetwork
from graph.algorithms import dijkstra, astar
from simulation.dynamic_router import DynamicRouter
from simulation.anomaly_detector import TrafficAnomalyDetector
from data.feature_engineering import load_dataset

# ------------------------------------------------------------------
# App setup
# ------------------------------------------------------------------
app = FastAPI(
    title="PathMind AI",
    description="Intelligent Route Optimization and Traffic Prediction API",
    version="1.0.0",
)

# Global state (loaded at startup)
_network = None
_router = None
_anomaly_detector = None
_df = None


@app.on_event("startup")
def startup():
    global _network, _router, _anomaly_detector, _df

    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "traffic_dataset.csv")

    print("[API] Loading dataset ...")
    _df = load_dataset(dataset_path)

    print("[API] Building road network ...")
    _network = RoadNetwork.build_from_dataset(_df)
    _router = DynamicRouter(_network)

    print("[API] Fitting anomaly detector ...")
    _anomaly_detector = TrafficAnomalyDetector()
    sample = _df.sample(n=min(5000, len(_df)), random_state=42)
    _anomaly_detector.fit(sample)

    # Set initial dynamic weights
    traffic = {}
    for u, v, _ in _network.get_all_edges():
        traffic[(u, v)] = np.random.uniform(0.1, 0.6)
    _router.update_graph_weights(traffic)

    print(f"[API] Ready! Network: {_network}")


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "name": "PathMind AI",
        "version": "1.0.0",
        "endpoints": ["/route", "/traffic", "/anomalies", "/graph/info"],
    }


@app.get("/route")
def get_route(
    source: int = Query(..., description="Source node ID"),
    destination: int = Query(..., description="Destination node ID"),
    algorithm: str = Query("dijkstra", description="'dijkstra' or 'astar'"),
):
    """Find the optimal route between two nodes."""
    if source not in _network._nodes:
        raise HTTPException(404, f"Source node {source} not found")
    if destination not in _network._nodes:
        raise HTTPException(404, f"Destination node {destination} not found")

    result = _router.find_best_route(source, destination, algorithm)

    if result["path"] is None:
        raise HTTPException(404, f"No path found from {source} to {destination}")

    return result


@app.get("/traffic")
def get_traffic(
    edge_source: int = Query(..., description="Edge source node"),
    edge_dest: int = Query(..., description="Edge destination node"),
):
    """Get current traffic info for a specific edge."""
    edge = _network.get_edge(edge_source, edge_dest)
    if edge is None:
        raise HTTPException(404, f"Edge ({edge_source}, {edge_dest}) not found")

    return {
        "edge": [edge_source, edge_dest],
        "traffic_level": edge["traffic_level"],
        "current_travel_time": edge["current_travel_time"],
        "base_travel_time": edge["base_travel_time"],
        "road_type": edge["road_type"],
    }


@app.get("/anomalies")
def get_anomalies():
    """Return current anomalous edges."""
    latest_ts = _df["timestamp"].max()
    snapshot = _df[_df["timestamp"] == latest_ts].head(50)
    anomalies = _anomaly_detector.get_anomalous_edges(snapshot)
    return {"count": len(anomalies), "anomalies": anomalies}


@app.get("/graph/info")
def graph_info():
    """Return graph statistics."""
    return {
        "nodes": _network.num_nodes,
        "edges": _network.num_edges,
        "sample_nodes": _network.get_nodes()[:10],
    }


# ------------------------------------------------------------------
# Run with: uvicorn api:app --reload --port 8000
# ------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
