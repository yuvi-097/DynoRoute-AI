"""
PathMind AI - Shortest Path Algorithms (DSA Core)
===================================================
Dijkstra's Algorithm and A* implemented from scratch
using a min-heap priority queue.
"""

import heapq
import math
from typing import Optional

from graph.road_network import RoadNetwork


# ------------------------------------------------------------------
# Dijkstra's Algorithm
# ------------------------------------------------------------------
def dijkstra(
    network: RoadNetwork,
    source: int,
    destination: int,
    weight_key: str = "current_travel_time",
) -> tuple[Optional[list[int]], float]:
    """
    Compute the shortest path from source to destination using
    Dijkstra's algorithm with a min-heap priority queue.

    Args:
        network:     RoadNetwork graph instance
        source:      start node ID
        destination: end node ID
        weight_key:  edge attribute to use as weight

    Returns:
        (path, cost)  where path is a list of node IDs, cost is total weight.
        Returns (None, inf) if no path exists.
    """
    # dist[node] = best known distance from source
    dist: dict[int, float] = {source: 0.0}
    # prev[node] = predecessor on shortest path
    prev: dict[int, Optional[int]] = {source: None}
    # visited set
    visited: set[int] = set()
    # Priority queue: (distance, node)
    heap: list[tuple[float, int]] = [(0.0, source)]

    while heap:
        curr_dist, u = heapq.heappop(heap)

        if u in visited:
            continue
        visited.add(u)

        # Early exit when destination reached
        if u == destination:
            break

        for v in network.get_neighbors(u):
            if v in visited:
                continue
            w = network.get_weight(u, v, weight_key)
            new_dist = curr_dist + w

            if new_dist < dist.get(v, float("inf")):
                dist[v] = new_dist
                prev[v] = u
                heapq.heappush(heap, (new_dist, v))

    # Reconstruct path
    if destination not in prev:
        return None, float("inf")

    path = []
    node = destination
    while node is not None:
        path.append(node)
        node = prev[node]
    path.reverse()

    return path, dist[destination]


# ------------------------------------------------------------------
# A* Algorithm
# ------------------------------------------------------------------
def _euclidean_heuristic(network: RoadNetwork, a: int, b: int) -> float:
    """Euclidean distance between node a and node b as heuristic."""
    if a not in network.coords or b not in network.coords:
        return 0.0
    ax, ay = network.coords[a]
    bx, by = network.coords[b]
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def astar(
    network: RoadNetwork,
    source: int,
    destination: int,
    weight_key: str = "current_travel_time",
) -> tuple[Optional[list[int]], float]:
    """
    Compute the shortest path using A* with Euclidean distance heuristic.

    Args:
        network:     RoadNetwork graph instance
        source:      start node ID
        destination: end node ID
        weight_key:  edge attribute to use as weight

    Returns:
        (path, cost)  same format as dijkstra().
    """
    # g_score[n] = cost of cheapest path from source to n
    g_score: dict[int, float] = {source: 0.0}
    # f_score[n] = g_score[n] + heuristic(n, destination)
    f_score: dict[int, float] = {source: _euclidean_heuristic(network, source, destination)}
    # prev for path reconstruction
    prev: dict[int, Optional[int]] = {source: None}

    visited: set[int] = set()
    # Priority queue: (f_score, node)
    heap: list[tuple[float, int]] = [(f_score[source], source)]

    nodes_expanded = 0

    while heap:
        _, u = heapq.heappop(heap)

        if u in visited:
            continue
        visited.add(u)
        nodes_expanded += 1

        if u == destination:
            break

        for v in network.get_neighbors(u):
            if v in visited:
                continue

            w = network.get_weight(u, v, weight_key)
            tentative_g = g_score[u] + w

            if tentative_g < g_score.get(v, float("inf")):
                g_score[v] = tentative_g
                f_score[v] = tentative_g + _euclidean_heuristic(network, v, destination)
                prev[v] = u
                heapq.heappush(heap, (f_score[v], v))

    # Reconstruct
    if destination not in prev:
        return None, float("inf")

    path = []
    node = destination
    while node is not None:
        path.append(node)
        node = prev[node]
    path.reverse()

    return path, g_score[destination]


# ------------------------------------------------------------------
# Comparison helper
# ------------------------------------------------------------------
def compare_algorithms(
    network: RoadNetwork,
    source: int,
    destination: int,
    weight_key: str = "current_travel_time",
) -> dict:
    """Run both Dijkstra and A*, return paths and performance comparison."""
    import time

    t0 = time.perf_counter()
    path_d, cost_d = dijkstra(network, source, destination, weight_key)
    time_d = time.perf_counter() - t0

    t0 = time.perf_counter()
    path_a, cost_a = astar(network, source, destination, weight_key)
    time_a = time.perf_counter() - t0

    return {
        "source": source,
        "destination": destination,
        "dijkstra": {"path": path_d, "cost": round(cost_d, 4), "time_ms": round(time_d * 1000, 4)},
        "astar":    {"path": path_a, "cost": round(cost_a, 4), "time_ms": round(time_a * 1000, 4)},
    }
