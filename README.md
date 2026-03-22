# PathMind AI - Intelligent Route Optimization and Traffic Prediction System

A production-grade system combining **Data Structures & Algorithms (DSA)**, **Machine Learning**, and **Deep Learning** for real-time traffic prediction and dynamic route optimization.

---

## Architecture

```
pathmind-ai/
  data/
    feature_engineering.py    # Dataset loading, ML features, LSTM sequences
  graph/
    road_network.py           # Weighted directed graph (adjacency list)
    algorithms.py             # Dijkstra & A* from scratch
  models/
    ml_models.py              # Linear Regression + Random Forest
    lstm_model.py             # 2-layer LSTM (PyTorch)
    hybrid_engine.py          # Blended ML + DL prediction engine
  simulation/
    dynamic_router.py         # Live weight updates + routing
    anomaly_detector.py       # Isolation Forest anomaly detection
    simulation_engine.py      # Vehicle simulation with re-routing
  utils/
    evaluation.py             # MSE, MAE, R2, routing efficiency
    visualization.py          # Graph, route, heatmap, prediction plots
  main.py                     # End-to-end pipeline orchestrator
  api.py                      # FastAPI REST API
```

---

## DSA Concepts Used

| Concept | Implementation |
|---------|---------------|
| **Weighted Directed Graph** | Adjacency list (`dict[int, dict[int, dict]]`) in `road_network.py` |
| **Dijkstra's Algorithm** | Min-heap priority queue (`heapq`) for shortest path |
| **A\* Algorithm** | Euclidean distance heuristic + priority queue |
| **BFS** | Graph connectivity / reachability analysis |
| **Sliding Window** | LSTM sequence preparation (window size = 24) |

---

## ML/DL Models

### Baseline ML
- **Linear Regression** - Simple parametric baseline
- **Random Forest** (100 trees, max_depth=15) - Non-linear ensemble

### Deep Learning
- **LSTM** - 2-layer, hidden_size=64, dropout=0.2, trained from scratch in PyTorch
- Input: 24-hour traffic sequences; Output: next-hour traffic prediction

### Hybrid Engine
- `final_score = 0.4 * RF_prediction + 0.6 * LSTM_prediction`
- Converts predicted traffic to delay and travel time for graph weight updates

---

## How to Run

### Full Pipeline
```bash
cd pathmind-ai
python main.py
```

### API Server
```bash
cd pathmind-ai
uvicorn api:app --reload --port 8000
```

### API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/route?source=X&destination=Y` | GET | Find optimal route |
| `/traffic?edge_source=X&edge_dest=Y` | GET | Edge traffic info |
| `/anomalies` | GET | Current anomalous edges |
| `/graph/info` | GET | Network statistics |

---

## Design Decisions

1. **Graph algorithms from scratch** - Dijkstra and A* are hand-implemented (not using networkx) to demonstrate DSA understanding
2. **Temporal train/test split** - Chronological split preserves time-series integrity (no data leakage)
3. **LSTM over simple RNN** - Better long-term dependency capture for traffic patterns
4. **Hybrid blending** - Leverages RF's feature importance + LSTM's temporal modeling
5. **Dynamic re-routing** - Simulation detects congestion spikes and re-routes vehicles automatically
6. **Isolation Forest** - Unsupervised anomaly detection works without labeled incident data

---

## Dependencies

- Python 3.11+
- numpy, pandas, scikit-learn, matplotlib
- PyTorch (CPU)
- FastAPI, uvicorn
- networkx (visualization only)
