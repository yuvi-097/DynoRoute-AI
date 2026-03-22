"""
PathMind AI - Traffic Simulation Engine
=========================================
Simulates vehicles moving through the road network
with changing traffic over time and dynamic re-routing.
"""

import random
import numpy as np
from typing import Optional

from graph.road_network import RoadNetwork
from graph.algorithms import dijkstra
from simulation.dynamic_router import DynamicRouter


class Vehicle:
    """A vehicle traveling from source to destination."""

    def __init__(self, vid: int, source: int, destination: int):
        self.vid = vid
        self.source = source
        self.destination = destination
        self.path: Optional[list[int]] = None
        self.cost: float = 0.0
        self.position_idx: int = 0    # index in path
        self.total_time: float = 0.0
        self.rerouted: int = 0
        self.finished: bool = False

    def __repr__(self) -> str:
        status = "DONE" if self.finished else f"step {self.position_idx}"
        return f"Vehicle({self.vid}: {self.source}->{self.destination} [{status}])"


class TrafficSimulator:
    """
    Runs an hourly simulation:
    1. Generate/sample traffic conditions
    2. Detect anomalies
    3. Update graph weights
    4. Route/re-route vehicles
    5. Log per-step metrics
    """

    def __init__(
        self,
        network: RoadNetwork,
        dynamic_router: DynamicRouter,
        anomaly_detector=None,
        seed: int = 42,
    ):
        self.network = network
        self.router = dynamic_router
        self.anomaly_detector = anomaly_detector
        self.rng = random.Random(seed)
        np.random.seed(seed)

        self.vehicles: list[Vehicle] = []
        self.logs: list[dict] = []

    def spawn_vehicles(self, count: int = 20) -> None:
        """Create vehicles on random connected source-destination pairs."""
        nodes = self.network.get_nodes()
        spawned = 0
        attempts = 0

        while spawned < count and attempts < count * 10:
            src = self.rng.choice(nodes)
            dst = self.rng.choice(nodes)
            attempts += 1
            if src == dst:
                continue
            # Check reachability
            path, cost = dijkstra(self.network, src, dst)
            if path is not None:
                v = Vehicle(spawned, src, dst)
                v.path = path
                v.cost = cost
                self.vehicles.append(v)
                spawned += 1

        print(f"[Sim] Spawned {spawned} vehicles")

    def _generate_traffic_snapshot(self, hour: int) -> dict[tuple[int, int], float]:
        """Generate random traffic levels for all edges at given hour."""
        traffic = {}
        for u, v, attrs in self.network.get_all_edges():
            # Simple hour-based traffic
            if 8 <= hour <= 10 or 17 <= hour <= 19:
                base = np.random.uniform(0.5, 0.9)
            elif 0 <= hour <= 5:
                base = np.random.uniform(0.05, 0.25)
            else:
                base = np.random.uniform(0.2, 0.5)

            # Random events (5% chance)
            if np.random.random() < 0.05:
                base = min(base * 2.0, 1.0)

            traffic[(u, v)] = round(base, 4)
        return traffic

    def run(self, hours: int = 24) -> list[dict]:
        """
        Run simulation for given number of hours.

        Returns list of per-hour metrics.
        """
        print(f"\n[Sim] Running {hours}-hour simulation with "
              f"{len(self.vehicles)} vehicles ...")

        for h in range(hours):
            hour = h % 24

            # 1. Generate traffic
            traffic = self._generate_traffic_snapshot(hour)

            # 2. Update graph weights
            self.router.update_graph_weights(traffic)

            # 3. Count anomalies (simple threshold)
            anomaly_count = sum(1 for t in traffic.values() if t > 0.8)

            # 4. Move vehicles & re-route if needed
            reroutes = 0
            completed = 0
            active_times = []

            for v in self.vehicles:
                if v.finished:
                    continue

                # Check if current path is still optimal
                if v.path and v.position_idx < len(v.path) - 1:
                    current_node = v.path[v.position_idx]
                    next_node = v.path[v.position_idx + 1]

                    # Current edge traffic
                    edge_traffic = traffic.get((current_node, next_node), 0.3)

                    # Re-route if high traffic on next edge
                    if edge_traffic > 0.7 and v.position_idx < len(v.path) - 2:
                        new_result = self.router.find_best_route(
                            current_node, v.destination)
                        if new_result["path"] and new_result["cost"] < v.cost * 0.9:
                            v.path = new_result["path"]
                            v.cost = new_result["cost"]
                            v.position_idx = 0
                            v.rerouted += 1
                            reroutes += 1

                    # Advance position
                    travel_time = self.network.get_weight(
                        current_node, next_node)
                    v.total_time += travel_time
                    v.position_idx += 1

                    if v.position_idx >= len(v.path) - 1:
                        v.finished = True
                        completed += 1
                        active_times.append(v.total_time)

            active = sum(1 for v in self.vehicles if not v.finished)

            step_log = {
                "hour": hour,
                "step": h,
                "active_vehicles": active,
                "completed": completed,
                "reroutes": reroutes,
                "anomalies": anomaly_count,
                "avg_traffic": round(np.mean(list(traffic.values())), 4),
                "avg_travel_time": round(np.mean(active_times), 2) if active_times else 0.0,
            }
            self.logs.append(step_log)

            if h % 6 == 0:
                print(f"  Hour {hour:2d} | Active: {active:3d} | "
                      f"Completed: {completed} | Re-routes: {reroutes} | "
                      f"Anomalies: {anomaly_count}")

        total_completed = sum(1 for v in self.vehicles if v.finished)
        total_reroutes = sum(v.rerouted for v in self.vehicles)
        print(f"\n[Sim] Complete: {total_completed}/{len(self.vehicles)} "
              f"vehicles finished | {total_reroutes} total re-routes")

        return self.logs
