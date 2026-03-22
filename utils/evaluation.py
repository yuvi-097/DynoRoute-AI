"""
PathMind AI - Evaluation Metrics
===================================
Prediction accuracy and routing efficiency metrics.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def prediction_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str = "") -> dict:
    """Compute MSE, MAE, R2 for predictions."""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {
        "label": label,
        "mse": round(mse, 6),
        "mae": round(mae, 6),
        "r2": round(r2, 4),
    }


def routing_efficiency(static_cost: float, dynamic_cost: float) -> dict:
    """Compare static vs dynamic routing costs."""
    if static_cost == 0:
        return {"static_cost": 0, "dynamic_cost": 0, "time_saved": 0, "improvement_pct": 0}

    time_saved = static_cost - dynamic_cost
    improvement = (time_saved / static_cost) * 100

    return {
        "static_cost": round(static_cost, 4),
        "dynamic_cost": round(dynamic_cost, 4),
        "time_saved": round(time_saved, 4),
        "improvement_pct": round(improvement, 2),
    }


def print_evaluation_report(
    ml_metrics: dict,
    lstm_metrics: dict,
    routing_stats: list[dict],
    sim_logs: list[dict],
) -> None:
    """Print a formatted evaluation summary."""
    print("\n" + "=" * 60)
    print("  PATHMIND AI - EVALUATION REPORT")
    print("=" * 60)

    print("\n--- Prediction Accuracy ---")
    for name, m in ml_metrics.items():
        print(f"  {name:25s}  MSE={m['mse']:.6f}  MAE={m['mae']:.6f}  R2={m['r2']:.4f}")
    print(f"  {'LSTM':25s}  MSE={lstm_metrics['mse']:.6f}  "
          f"MAE={lstm_metrics['mae']:.6f}  R2={lstm_metrics['r2']:.4f}")

    if routing_stats:
        print("\n--- Routing Efficiency ---")
        avg_saved = np.mean([r["time_saved"] for r in routing_stats])
        avg_pct = np.mean([r["improvement_pct"] for r in routing_stats])
        print(f"  Avg time saved (dynamic vs static): {avg_saved:.2f} minutes")
        print(f"  Avg improvement: {avg_pct:.1f}%")

    if sim_logs:
        print("\n--- Simulation Summary ---")
        total_reroutes = sum(s["reroutes"] for s in sim_logs)
        total_anomalies = sum(s["anomalies"] for s in sim_logs)
        avg_traffic = np.mean([s["avg_traffic"] for s in sim_logs])
        print(f"  Total re-routes triggered: {total_reroutes}")
        print(f"  Total anomaly detections:  {total_anomalies}")
        print(f"  Average traffic level:     {avg_traffic:.4f}")

    print("\n" + "=" * 60)
