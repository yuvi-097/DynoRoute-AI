"""
PathMind AI - Anomaly Detection
==================================
Isolation Forest based anomaly detector for traffic spikes.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


class TrafficAnomalyDetector:
    """
    Detects anomalous traffic conditions using Isolation Forest
    and statistical thresholds.

    Classification:
        - Normal:    traffic <= 0.5  AND  not outlier
        - Congested: traffic >  0.7  AND  not outlier
        - Anomalous: flagged by Isolation Forest as outlier
    """

    def __init__(self, contamination: float = 0.05):
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_jobs=-1,
        )
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> None:
        """Fit on traffic features."""
        features = self._extract_features(df)
        self.model.fit(features)
        self._fitted = True
        print(f"[Anomaly] Fitted on {len(features):,} samples")

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify each row as Normal / Congested / Anomalous.
        Returns a copy of df with added 'anomaly_label' column.
        """
        result = df.copy()
        features = self._extract_features(df)

        # Isolation Forest: -1 = outlier, 1 = inlier
        preds = self.model.predict(features)

        labels = []
        for i, pred in enumerate(preds):
            traffic = df.iloc[i]["traffic_level"]
            if pred == -1:
                labels.append("Anomalous")
            elif traffic > 0.7:
                labels.append("Congested")
            else:
                labels.append("Normal")

        result["anomaly_label"] = labels

        counts = result["anomaly_label"].value_counts().to_dict()
        print(f"[Anomaly] Detection results: {counts}")
        return result

    @staticmethod
    def _extract_features(df: pd.DataFrame) -> np.ndarray:
        """Select numeric columns relevant for anomaly detection."""
        cols = ["traffic_level", "delay", "current_travel_time",
                "is_peak_hour", "event_flag"]
        available = [c for c in cols if c in df.columns]
        return df[available].fillna(0).values

    def get_anomalous_edges(self, df: pd.DataFrame) -> list[dict]:
        """Return list of anomalous edge entries."""
        detected = self.detect(df)
        anomalies = detected[detected["anomaly_label"] == "Anomalous"]
        results = []
        for _, row in anomalies.head(20).iterrows():
            results.append({
                "edge_id": int(row["edge_id"]),
                "source_node": int(row["source_node"]),
                "destination_node": int(row["destination_node"]),
                "traffic_level": round(float(row["traffic_level"]), 4),
                "delay": round(float(row["delay"]), 2),
                "label": "Anomalous",
            })
        return results
