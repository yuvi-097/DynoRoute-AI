"""
PathMind AI - Feature Engineering
===================================
Load the traffic dataset, encode features, and prepare
inputs for ML and LSTM models.
"""

import numpy as np
import pandas as pd
from typing import Tuple


def load_dataset(path: str) -> pd.DataFrame:
    """Load CSV and parse timestamps."""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values(["edge_id", "timestamp"]).reset_index(drop=True)
    print(f"[FeatureEng] Loaded {len(df):,} rows, {df['edge_id'].nunique()} edges")
    return df


# ------------------------------------------------------------------
# ML feature preparation
# ------------------------------------------------------------------
def prepare_ml_features(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Prepare features and target for classical ML models.

    Features used:
        hour, day_of_week, traffic_level, delay, base_travel_time,
        traffic_t-1, traffic_t-2, delay_t-1, road_length, speed_limit,
        num_lanes, is_peak_hour, is_weekend, event_flag,
        road_type (one-hot), weather (one-hot)

    Target: future_traffic

    Returns:
        X_train, X_test, y_train, y_test, feature_names
    """
    work = df.dropna(subset=["traffic_t-1", "traffic_t-2", "delay_t-1", "future_traffic"]).copy()

    # One-hot encode categoricals
    work = pd.get_dummies(work, columns=["road_type", "weather"], drop_first=True)

    # Feature columns
    exclude = {"edge_id", "source_node", "destination_node", "timestamp",
               "current_travel_time", "future_traffic"}
    feature_cols = [c for c in work.columns if c not in exclude]
    feature_names = feature_cols

    X = work[feature_cols].values.astype(np.float32)
    y = work["future_traffic"].values.astype(np.float32)

    # Chronological split (no shuffle - time-series)
    split = int(len(X) * (1 - test_ratio))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"[FeatureEng] ML features: {len(feature_cols)} cols | "
          f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    return X_train, X_test, y_train, y_test, feature_names


# ------------------------------------------------------------------
# LSTM sequence preparation
# ------------------------------------------------------------------
def prepare_lstm_sequences(
    df: pd.DataFrame,
    seq_len: int = 24,
    test_ratio: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create sliding-window sequences per edge for LSTM training.

    For each edge, take sliding windows of length `seq_len` over
    `traffic_level` and use the next value as the target.

    Returns:
        X_train, X_test, y_train, y_test
        where X shape = (N, seq_len, 1), y shape = (N,)
    """
    sequences = []
    targets = []

    for _, group in df.groupby("edge_id"):
        values = group["traffic_level"].values
        for i in range(len(values) - seq_len):
            sequences.append(values[i : i + seq_len])
            targets.append(values[i + seq_len])

    X = np.array(sequences, dtype=np.float32).reshape(-1, seq_len, 1)
    y = np.array(targets, dtype=np.float32)

    split = int(len(X) * (1 - test_ratio))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"[FeatureEng] LSTM sequences: seq_len={seq_len} | "
          f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    return X_train, X_test, y_train, y_test
