"""
PathMind AI - Visualization
==============================
Graph plots, route visualization, and traffic heatmaps.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Optional

from graph.road_network import RoadNetwork


def plot_graph(
    network: RoadNetwork,
    title: str = "Road Network",
    save_path: Optional[str] = None,
) -> None:
    """Draw the road network with nodes and edges colored by traffic."""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Draw edges
    for u in network.adj:
        for v, attrs in network.adj[u].items():
            if u in network.coords and v in network.coords:
                x1, y1 = network.coords[u]
                x2, y2 = network.coords[v]
                traffic = attrs.get("traffic_level", 0.3)

                # Color: green (low) -> yellow -> red (high)
                color = plt.cm.RdYlGn(1.0 - traffic)

                ax.annotate("",
                            xy=(x2, y2), xytext=(x1, y1),
                            arrowprops=dict(arrowstyle="->",
                                            color=color,
                                            lw=1.5,
                                            alpha=0.7))

    # Draw nodes
    for node, (x, y) in network.coords.items():
        ax.scatter(x, y, s=80, c="steelblue", zorder=5, edgecolors="white", linewidths=0.5)
        ax.annotate(str(node), (x, y), fontsize=6, ha="center", va="center",
                    color="white", fontweight="bold")

    # Legend
    patches = [
        mpatches.Patch(color="green", label="Low Traffic"),
        mpatches.Patch(color="yellow", label="Medium Traffic"),
        mpatches.Patch(color="red", label="High Traffic"),
    ]
    ax.legend(handles=patches, loc="upper right")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Viz] Saved graph plot to {save_path}")
    plt.close()


def plot_route(
    network: RoadNetwork,
    path: list[int],
    title: str = "Optimal Route",
    save_path: Optional[str] = None,
) -> None:
    """Highlight a specific route on the network."""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Draw all edges (gray)
    for u in network.adj:
        for v in network.adj[u]:
            if u in network.coords and v in network.coords:
                x1, y1 = network.coords[u]
                x2, y2 = network.coords[v]
                ax.plot([x1, x2], [y1, y2], c="lightgray", lw=0.8, alpha=0.5)

    # Draw route edges (bright)
    if path:
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if u in network.coords and v in network.coords:
                x1, y1 = network.coords[u]
                x2, y2 = network.coords[v]
                ax.annotate("",
                            xy=(x2, y2), xytext=(x1, y1),
                            arrowprops=dict(arrowstyle="-|>",
                                            color="#FF4444",
                                            lw=3.0))

        # Mark start and end
        sx, sy = network.coords[path[0]]
        ex, ey = network.coords[path[-1]]
        ax.scatter(sx, sy, s=200, c="green", zorder=10, marker="^", label="Start")
        ax.scatter(ex, ey, s=200, c="red", zorder=10, marker="s", label="End")

    # All nodes
    for node, (x, y) in network.coords.items():
        ax.scatter(x, y, s=50, c="steelblue", zorder=5, edgecolors="white", linewidths=0.5)

    ax.legend(fontsize=10)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Viz] Saved route plot to {save_path}")
    plt.close()


def plot_traffic_heatmap(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> None:
    """Hour-of-day vs edge_id traffic heatmap."""
    pivot = df.pivot_table(values="traffic_level",
                           index="edge_id",
                           columns="hour",
                           aggfunc="mean")

    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xlabel("Hour of Day", fontsize=12)
    ax.set_ylabel("Edge ID", fontsize=12)
    ax.set_title("Traffic Heatmap (Avg Traffic Level by Hour)", fontsize=14, fontweight="bold")
    ax.set_xticks(range(24))
    ax.set_xticklabels(range(24))

    plt.colorbar(im, ax=ax, label="Traffic Level")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Viz] Saved heatmap to {save_path}")
    plt.close()


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predictions vs Actual",
    save_path: Optional[str] = None,
) -> None:
    """Scatter plot of predicted vs actual values."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter
    sample = min(2000, len(y_true))
    idx = np.random.choice(len(y_true), sample, replace=False)
    axes[0].scatter(y_true[idx], y_pred[idx], alpha=0.3, s=10, c="steelblue")
    axes[0].plot([0, 1], [0, 1], "r--", lw=2, label="Perfect")
    axes[0].set_xlabel("Actual")
    axes[0].set_ylabel("Predicted")
    axes[0].set_title(f"{title} (scatter)")
    axes[0].legend()

    # Line (first 200 points)
    n = min(200, len(y_true))
    axes[1].plot(range(n), y_true[:n], label="Actual", alpha=0.8)
    axes[1].plot(range(n), y_pred[:n], label="Predicted", alpha=0.8)
    axes[1].set_xlabel("Sample")
    axes[1].set_ylabel("Traffic Level")
    axes[1].set_title(f"{title} (line)")
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Viz] Saved predictions plot to {save_path}")
    plt.close()


def plot_training_loss(
    train_losses: list,
    val_losses: list,
    save_path: Optional[str] = None,
) -> None:
    """Plot LSTM training and validation loss curves."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_losses, label="Train Loss", lw=2)
    ax.plot(val_losses, label="Val Loss", lw=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("LSTM Training Progress", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Viz] Saved loss plot to {save_path}")
    plt.close()
