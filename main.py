"""
PathMind AI - Main Orchestrator
==================================
End-to-end pipeline: load data, build graph, train models,
run routing, simulate, evaluate, and visualize.
"""

import sys
import os
import numpy as np

# ---- Setup path so modules can be imported ----
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.feature_engineering import load_dataset, prepare_ml_features, prepare_lstm_sequences
from graph.road_network import RoadNetwork
from graph.algorithms import dijkstra, astar, compare_algorithms
from models.ml_models import TrafficPredictor
from models.lstm_model import train_lstm, evaluate_lstm, predict_lstm
from models.hybrid_engine import HybridPredictor
from simulation.dynamic_router import DynamicRouter
from simulation.anomaly_detector import TrafficAnomalyDetector
from simulation.simulation_engine import TrafficSimulator
from utils.evaluation import prediction_metrics, routing_efficiency, print_evaluation_report
from utils.visualization import (plot_graph, plot_route, plot_traffic_heatmap,
                                  plot_predictions, plot_training_loss)


# Path to dataset (one level up from project dir)
DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "traffic_dataset.csv")


def main():
    print("=" * 60)
    print("  PathMind AI - Intelligent Route Optimization System")
    print("=" * 60)

    # Create output directories
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    # ==================================================================
    # STEP 1: Load Data
    # ==================================================================
    print("\n[STEP 1] Loading dataset ...")
    df = load_dataset(DATASET_PATH)

    # ==================================================================
    # STEP 2: Build Graph
    # ==================================================================
    print("\n[STEP 2] Building road network graph ...")
    network = RoadNetwork.build_from_dataset(df)
    print(f"  {network}")

    # ==================================================================
    # STEP 3: Test Graph Algorithms
    # ==================================================================
    print("\n[STEP 3] Testing shortest path algorithms ...")
    nodes = network.get_nodes()

    # Find a valid pair with BFS
    test_src = nodes[0]
    reachable = network.find_reachable(test_src)
    reachable_list = sorted(reachable - {test_src})

    if len(reachable_list) >= 1:
        test_dst = reachable_list[len(reachable_list) // 2]
        result = compare_algorithms(network, test_src, test_dst)
        print(f"  Source: {result['source']} -> Destination: {result['destination']}")
        print(f"  Dijkstra: cost={result['dijkstra']['cost']}, "
              f"time={result['dijkstra']['time_ms']}ms, "
              f"path length={len(result['dijkstra']['path'] or [])}")
        print(f"  A*:       cost={result['astar']['cost']}, "
              f"time={result['astar']['time_ms']}ms, "
              f"path length={len(result['astar']['path'] or [])}")

        # Visualize route
        if result["dijkstra"]["path"]:
            plot_route(network, result["dijkstra"]["path"],
                       title=f"Dijkstra Route: {test_src} -> {test_dst}",
                       save_path="output/route_dijkstra.png")
    else:
        print("  WARNING: No reachable nodes from test source")
        test_dst = nodes[1]

    # ==================================================================
    # STEP 4: Feature Engineering
    # ==================================================================
    print("\n[STEP 4] Feature engineering ...")
    X_train, X_test, y_train, y_test, feature_names = prepare_ml_features(df)
    X_lstm_train, X_lstm_test, y_lstm_train, y_lstm_test = prepare_lstm_sequences(df)

    # ==================================================================
    # STEP 5: Train ML Models
    # ==================================================================
    print("\n[STEP 5] Training ML models ...")
    ml_pred = TrafficPredictor(model_dir="saved_models")
    train_results = ml_pred.train(X_train, y_train)
    ml_test_results = ml_pred.evaluate(X_test, y_test)

    # Visualize RF predictions
    rf_pred = ml_pred.predict(X_test, model="rf")
    plot_predictions(y_test, rf_pred,
                     title="Random Forest",
                     save_path="output/predictions_rf.png")

    # ==================================================================
    # STEP 6: Train LSTM Model
    # ==================================================================
    print("\n[STEP 6] Training LSTM model ...")
    lstm_model, train_losses, val_losses = train_lstm(
        X_lstm_train, y_lstm_train,
        X_lstm_test, y_lstm_test,
        epochs=30, batch_size=64,
        model_dir="saved_models",
    )
    lstm_test_results = evaluate_lstm(lstm_model, X_lstm_test, y_lstm_test)

    # Visualize LSTM
    lstm_pred = predict_lstm(lstm_model, X_lstm_test)
    plot_predictions(y_lstm_test, lstm_pred,
                     title="LSTM",
                     save_path="output/predictions_lstm.png")
    plot_training_loss(train_losses, val_losses,
                       save_path="output/lstm_training_loss.png")

    # ==================================================================
    # STEP 7: Hybrid Engine
    # ==================================================================
    print("\n[STEP 7] Testing hybrid prediction engine ...")
    hybrid = HybridPredictor(ml_pred, lstm_model, alpha=0.4)
    print(f"  Hybrid engine ready (alpha={hybrid.alpha})")

    # ==================================================================
    # STEP 8: Anomaly Detection
    # ==================================================================
    print("\n[STEP 8] Running anomaly detection ...")
    anomaly_det = TrafficAnomalyDetector(contamination=0.05)
    # Use a sample for speed
    sample_df = df.sample(n=min(10000, len(df)), random_state=42)
    anomaly_det.fit(sample_df)
    detected = anomaly_det.detect(sample_df)
    anomalous_edges = anomaly_det.get_anomalous_edges(sample_df)
    print(f"  Found {len(anomalous_edges)} anomalous edge-timestamps (sample)")

    # ==================================================================
    # STEP 9: Dynamic Routing
    # ==================================================================
    print("\n[STEP 9] Dynamic routing demo ...")
    router = DynamicRouter(network)

    # Simulate traffic update and reroute
    sample_traffic = {}
    for u, v, _ in network.get_all_edges():
        sample_traffic[(u, v)] = np.random.uniform(0.2, 0.8)
    router.update_graph_weights(sample_traffic)

    comparison = router.static_vs_dynamic(test_src, test_dst)
    print(f"  Static route cost:  {comparison['static']['cost']}")
    print(f"  Dynamic route cost: {comparison['dynamic']['cost']}")
    print(f"  Time saved:         {comparison['time_saved']}")

    # ==================================================================
    # STEP 10: Simulation
    # ==================================================================
    print("\n[STEP 10] Running traffic simulation ...")
    simulator = TrafficSimulator(network, router, anomaly_det)
    simulator.spawn_vehicles(count=30)
    sim_logs = simulator.run(hours=24)

    # ==================================================================
    # STEP 11: Visualization
    # ==================================================================
    print("\n[STEP 11] Generating visualizations ...")
    plot_graph(network, title="PathMind AI - Road Network",
               save_path="output/road_network.png")
    plot_traffic_heatmap(df, save_path="output/traffic_heatmap.png")

    # ==================================================================
    # STEP 12: Evaluation Report
    # ==================================================================
    routing_stats = [routing_efficiency(
        comparison["static"]["cost"],
        comparison["dynamic"]["cost"]
    )]
    print_evaluation_report(ml_test_results, lstm_test_results, routing_stats, sim_logs)

    print("\n[DONE] All outputs saved to output/ directory")
    print("  - output/road_network.png")
    print("  - output/route_dijkstra.png")
    print("  - output/predictions_rf.png")
    print("  - output/predictions_lstm.png")
    print("  - output/lstm_training_loss.png")
    print("  - output/traffic_heatmap.png")
    print("  - saved_models/linear_regression.pkl")
    print("  - saved_models/random_forest.pkl")
    print("  - saved_models/lstm_model.pt")


if __name__ == "__main__":
    main()
