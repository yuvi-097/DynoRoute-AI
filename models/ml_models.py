"""
PathMind AI - Classical ML Models
===================================
Linear Regression and Random Forest for traffic delay prediction.
"""

import os
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class TrafficPredictor:
    """
    Wrapper around Linear Regression and Random Forest regressors
    for predicting future traffic levels.
    """

    def __init__(self, model_dir: str = "saved_models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

        self.lr = LinearRegression()
        self.rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1,
        )
        self._trained = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> dict:
        """Train both models. Returns training metrics."""
        print("[ML] Training Linear Regression ...")
        self.lr.fit(X_train, y_train)
        lr_pred = self.lr.predict(X_train)
        lr_metrics = self._metrics(y_train, lr_pred, "LR-train")

        print("[ML] Training Random Forest ...")
        self.rf.fit(X_train, y_train)
        rf_pred = self.rf.predict(X_train)
        rf_metrics = self._metrics(y_train, rf_pred, "RF-train")

        self._trained = True

        # Save models
        joblib.dump(self.lr, os.path.join(self.model_dir, "linear_regression.pkl"))
        joblib.dump(self.rf, os.path.join(self.model_dir, "random_forest.pkl"))
        print(f"[ML] Models saved to {self.model_dir}/")

        return {"linear_regression": lr_metrics, "random_forest": rf_metrics}

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray, model: str = "rf") -> np.ndarray:
        """Predict using specified model ('lr' or 'rf')."""
        if model == "lr":
            return self.lr.predict(X)
        return self.rf.predict(X)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluate both models on test data."""
        lr_pred = self.lr.predict(X_test)
        rf_pred = self.rf.predict(X_test)

        results = {
            "linear_regression": self._metrics(y_test, lr_pred, "LR-test"),
            "random_forest": self._metrics(y_test, rf_pred, "RF-test"),
        }
        return results

    # ------------------------------------------------------------------
    # Load saved models
    # ------------------------------------------------------------------
    def load(self) -> None:
        """Load previously saved models."""
        lr_path = os.path.join(self.model_dir, "linear_regression.pkl")
        rf_path = os.path.join(self.model_dir, "random_forest.pkl")
        if os.path.exists(lr_path):
            self.lr = joblib.load(lr_path)
        if os.path.exists(rf_path):
            self.rf = joblib.load(rf_path)
        self._trained = True
        print("[ML] Models loaded from disk")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> dict:
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        print(f"  [{label}]  MSE={mse:.6f}  MAE={mae:.6f}  R2={r2:.4f}")
        return {"mse": round(mse, 6), "mae": round(mae, 6), "r2": round(r2, 4)}
