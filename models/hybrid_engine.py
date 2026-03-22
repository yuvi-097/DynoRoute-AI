"""
PathMind AI - Hybrid Traffic Prediction Engine
=================================================
Combines Random Forest and LSTM predictions into
a single traffic score for dynamic routing.
"""

import numpy as np


class HybridPredictor:
    """
    Weighted blend of ML (Random Forest) and DL (LSTM) predictions.

    final_traffic = alpha * rf_prediction + (1 - alpha) * lstm_prediction
    """

    def __init__(self, ml_predictor, lstm_model, alpha: float = 0.4):
        """
        Args:
            ml_predictor: TrafficPredictor instance (already trained)
            lstm_model:   TrafficLSTM instance (already trained)
            alpha:        weight for RF prediction (1-alpha for LSTM)
        """
        self.ml_predictor = ml_predictor
        self.lstm_model = lstm_model
        self.alpha = alpha

    def predict(
        self,
        ml_features: np.ndarray,
        lstm_sequences: np.ndarray,
    ) -> np.ndarray:
        """
        Produce blended traffic predictions.

        Args:
            ml_features:    shape (N, num_features) for Random Forest
            lstm_sequences: shape (N, seq_len, 1) for LSTM

        Returns:
            blended predictions of shape (N,)
        """
        from models.lstm_model import predict_lstm

        rf_pred = self.ml_predictor.predict(ml_features, model="rf")
        lstm_pred = predict_lstm(self.lstm_model, lstm_sequences)

        blended = self.alpha * rf_pred + (1 - self.alpha) * lstm_pred
        return np.clip(blended, 0.0, 1.0)

    def traffic_to_delay(
        self,
        traffic_level: float,
        base_travel_time: float,
        weather: str = "clear",
        event: bool = False,
    ) -> float:
        """Convert predicted traffic level to estimated delay (minutes)."""
        delay = traffic_level * np.random.uniform(2.0, 8.0)
        if weather == "rain":
            delay *= 1.3
        if event:
            delay *= 2.0
        return round(delay, 2)

    def traffic_to_travel_time(
        self,
        traffic_level: float,
        base_travel_time: float,
        weather: str = "clear",
        event: bool = False,
    ) -> float:
        """Predict current_travel_time from traffic and base time."""
        delay = self.traffic_to_delay(traffic_level, base_travel_time, weather, event)
        return round(base_travel_time + delay, 2)
