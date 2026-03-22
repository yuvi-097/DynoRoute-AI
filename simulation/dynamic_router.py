"""
PathMind AI - Dynamic Route Optimizer
=======================================
Updates graph edge weights using predicted traffic,
then computes optimal routes via Dijkstra/A*.
"""

import numpy as np

from graph.road_network import RoadNetwork
from graph.algorithms import dijkstra, astar


class DynamicRouter:
    """
    Predicts traffic for all edges and updates graph weights
    before computing shortest paths.
    """

    def __init__(self, network: RoadNetwork, hybrid_predictor=None):
        self.network = network
        self.predictor = hybrid_predictor

    def update_graph_weights(
        self,
        predicted_traffic: dict[tuple[int, int], float],
    ) -> None:
        """
        Update current_travel_time for every edge based on
        predicted traffic levels.

        Args:
            predicted_traffic: {(u, v): predicted_traffic_level}
        """
        for (u, v), traffic in predicted_traffic.items():
            edge = self.network.get_edge(u, v)
            if edge is None:
                continue
            base_tt = edge["base_travel_time"]
            # Simple delay model: delay = traffic * base * multiplier
            delay = traffic * base_tt * np.random.uniform(0.3, 0.8)
            new_tt = round(base_tt + delay, 2)
            self.network.update_weight(u, v,
                                       current_travel_time=new_tt,
                                       traffic_level=round(traffic, 4))

    def find_best_route(
        self,
        source: int,
        destination: int,
        algorithm: str = "dijkstra",
    ) -> dict:
        """
        Find optimal route using the specified algorithm
        with current (possibly updated) edge weights.

        Returns:
            {path, cost, algorithm}
        """
        if algorithm == "astar":
            path, cost = astar(self.network, source, destination)
        else:
            path, cost = dijkstra(self.network, source, destination)

        return {
            "source": source,
            "destination": destination,
            "path": path,
            "cost": round(cost, 4),
            "algorithm": algorithm,
            "hops": len(path) - 1 if path else 0,
        }

    def static_vs_dynamic(
        self,
        source: int,
        destination: int,
    ) -> dict:
        """Compare route cost on static vs dynamic (predicted) weights."""
        # Dynamic (current weights)
        dynamic = self.find_best_route(source, destination)

        # Static (base_travel_time weights)
        path, cost = dijkstra(self.network, source, destination,
                              weight_key="base_travel_time")

        return {
            "static": {"path": path, "cost": round(cost, 4)},
            "dynamic": dynamic,
            "time_saved": round(cost - dynamic["cost"], 4) if path else 0.0,
        }
