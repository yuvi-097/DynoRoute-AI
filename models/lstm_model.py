"""
PathMind AI - LSTM Deep Learning Model
=========================================
2-layer LSTM built from scratch in PyTorch for
time-series traffic prediction.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class TrafficLSTM(nn.Module):
    """
    2-layer LSTM for traffic prediction.

    Input shape  : (batch, seq_len, 1)
    Output shape : (batch, 1)
    """

    def __init__(self, input_size: int = 1, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        # Take the last time-step's output
        out = out[:, -1, :]       # (batch, hidden_size)
        out = self.fc(out)        # (batch, 1)
        return out.squeeze(-1)    # (batch,)


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------
def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 0.001,
    patience: int = 7,
    model_dir: str = "saved_models",
) -> tuple:
    """
    Train the LSTM model with early stopping.

    Returns:
        (model, train_losses, val_losses)
    """
    os.makedirs(model_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[LSTM] Training on {device}")

    # Tensors
    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.float32)
    X_te = torch.tensor(X_test, dtype=torch.float32)
    y_te = torch.tensor(y_test, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_tr, y_tr),
                              batch_size=batch_size, shuffle=True)

    model = TrafficLSTM().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []
    best_val = float("inf")
    wait = 0

    for epoch in range(1, epochs + 1):
        # -- Train --
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= len(X_tr)
        train_losses.append(epoch_loss)

        # -- Validate --
        model.eval()
        with torch.no_grad():
            val_pred = model(X_te.to(device))
            val_loss = criterion(val_pred, y_te.to(device)).item()
        val_losses.append(val_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"Train Loss: {epoch_loss:.6f} | Val Loss: {val_loss:.6f}")

        # Early stopping
        if val_loss < best_val:
            best_val = val_loss
            wait = 0
            torch.save(model.state_dict(),
                       os.path.join(model_dir, "lstm_model.pt"))
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stopping at epoch {epoch} (patience={patience})")
                break

    # Load best weights
    model.load_state_dict(torch.load(os.path.join(model_dir, "lstm_model.pt"),
                                     weights_only=True))
    model.eval()
    print(f"[LSTM] Best val loss: {best_val:.6f}")

    return model, train_losses, val_losses


# ------------------------------------------------------------------
# Prediction
# ------------------------------------------------------------------
def predict_lstm(model: TrafficLSTM, X: np.ndarray) -> np.ndarray:
    """Run inference on numpy array, return predictions."""
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        tensor = torch.tensor(X, dtype=torch.float32).to(device)
        preds = model(tensor).cpu().numpy()
    return preds


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------
def evaluate_lstm(model: TrafficLSTM, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Compute MSE, MAE, R2 for LSTM predictions."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    preds = predict_lstm(model, X_test)
    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"  [LSTM-test]  MSE={mse:.6f}  MAE={mae:.6f}  R2={r2:.4f}")
    return {"mse": round(mse, 6), "mae": round(mae, 6), "r2": round(r2, 4)}
