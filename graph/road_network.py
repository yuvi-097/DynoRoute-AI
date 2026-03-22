"""
PathMind AI - Road Network Graph (DSA Core)
=============================================
Weighted directed graph using adjacency list representation.
Nodes = intersections, Edges = roads with attributes.
"""

import math
import random
from collections import defaultdict
from typing import Optional


class RoadNetwork:
    """
    Weighted directed graph for road network modeling.

    Internal structure:
        adj[u][v] = {distance, base_travel_time, current_travel_time,
                     traffic_level, road_type, speed_limit, num_lanes, road_length}

    Each node also stores (x, y) coordinates for A* heuristic.
    """

    def __init__(self):
        # Adjacency list: node -> {neighbor -> attributes}
        self.adj: dict[int, dict[int, dict]] = defaultdict(dict)
        # Node coordinates for A* heuristic
        self.coords: dict[int, tuple[float, float]] = {}
        # Set of all nodes
        self._nodes: set[int] = set()

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------
    def add_node(self, node_id: int, x: float = 0.0, y: float = 0.0) -> None:
        """Add a node with optional (x, y) coordinates."""
        self._nodes.add(node_id)
        self.coords[node_id] = (x, y)
        if node_id not in self.adj:
            self.adj[node_id] = {}

    def remove_node(self, node_id: int) -> None:
        """Remove a node and all edges connected to it."""
        if node_id in self._nodes:
            self._nodes.discard(node_id)
            self.adj.pop(node_id, None)
            self.coords.pop(node_id, None)
            # Remove edges pointing TO this node
            for u in list(self.adj):
                self.adj[u].pop(node_id, None)

    def get_nodes(self) -> list[int]:
        return sorted(self._nodes)

    @property
    def num_nodes(self) -> int:
        return len(self._nodes)

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------
    def add_edge(self, u: int, v: int, **attrs) -> None:
        """
        Add a directed edge u -> v with arbitrary attributes.

        Expected attributes:
            road_length, base_travel_time, current_travel_time,
            traffic_level, road_type, speed_limit, num_lanes
        """
        # Ensure both nodes exist
        if u not in self._nodes:
            self.add_node(u)
        if v not in self._nodes:
            self.add_node(v)

        self.adj[u][v] = {
            "road_length": attrs.get("road_length", 1.0),
            "base_travel_time": attrs.get("base_travel_time", 1.0),
            "current_travel_time": attrs.get("current_travel_time", 1.0),
            "traffic_level": attrs.get("traffic_level", 0.0),
            "road_type": attrs.get("road_type", "local"),
            "speed_limit": attrs.get("speed_limit", 40),
            "num_lanes": attrs.get("num_lanes", 1),
        }

    def remove_edge(self, u: int, v: int) -> None:
        """Remove edge u -> v if it exists."""
        if u in self.adj:
            self.adj[u].pop(v, None)

    def get_edge(self, u: int, v: int) -> Optional[dict]:
        """Return edge attributes or None."""
        return self.adj.get(u, {}).get(v)

    def get_neighbors(self, u: int) -> list[int]:
        """Return list of neighbors of u."""
        return list(self.adj.get(u, {}).keys())

    def get_weight(self, u: int, v: int, weight_key: str = "current_travel_time") -> float:
        """Get a specific weight attribute for edge u->v."""
        edge = self.get_edge(u, v)
        if edge is None:
            return float("inf")
        return edge.get(weight_key, float("inf"))

    def update_weight(self, u: int, v: int, **updates) -> None:
        """Update attributes on an existing edge."""
        if u in self.adj and v in self.adj[u]:
            self.adj[u][v].update(updates)

    @property
    def num_edges(self) -> int:
        return sum(len(nbrs) for nbrs in self.adj.values())

    def get_all_edges(self) -> list[tuple[int, int, dict]]:
        """Return list of (u, v, attrs) for every edge."""
        edges = []
        for u in self.adj:
            for v, attrs in self.adj[u].items():
                edges.append((u, v, attrs))
        return edges

    # ------------------------------------------------------------------
    # Build from dataset
    # ------------------------------------------------------------------
    @classmethod
    def build_from_dataset(cls, df, seed: int = 42) -> "RoadNetwork":
        """
        Construct a RoadNetwork from the traffic dataset DataFrame.

        Uses the latest timestamp snapshot to set initial edge weights.
        Assigns random (x, y) positions to nodes for A* heuristic.
        """
        network = cls()
        rng = random.Random(seed)

        # Collect all unique nodes and assign coordinates
        all_nodes = set(df["source_node"].unique()) | set(df["destination_node"].unique())
        for node in all_nodes:
            network.add_node(int(node),
                             x=rng.uniform(0, 100),
                             y=rng.uniform(0, 100))

        # Use the latest snapshot for initial edge weights
        latest_ts = df["timestamp"].max()
        snapshot = df[df["timestamp"] == latest_ts]

        for _, row in snapshot.iterrows():
            network.add_edge(
                int(row["source_node"]),
                int(row["destination_node"]),
                road_length=float(row["road_length"]),
                base_travel_time=float(row["base_travel_time"]),
                current_travel_time=float(row["current_travel_time"]),
                traffic_level=float(row["traffic_level"]),
                road_type=str(row["road_type"]),
                speed_limit=int(row["speed_limit"]),
                num_lanes=int(row["num_lanes"]),
            )

        # Add some reverse edges so the graph is more connected
        # (makes routing between arbitrary nodes possible)
        existing = set()
        for u in network.adj:
            for v in network.adj[u]:
                existing.add((u, v))

        nodes_list = list(all_nodes)
        added = 0
        for u, v in existing:
            if (v, u) not in existing and rng.random() < 0.5:
                fwd = network.get_edge(u, v)
                network.add_edge(int(v), int(u),
                                 road_length=fwd["road_length"],
                                 base_travel_time=fwd["base_travel_time"],
                                 current_travel_time=fwd["current_travel_time"],
                                 traffic_level=fwd["traffic_level"],
                                 road_type=fwd["road_type"],
                                 speed_limit=fwd["speed_limit"],
                                 num_lanes=fwd["num_lanes"])
                added += 1

        # Add extra random edges to improve connectivity
        for _ in range(50):
            u = rng.choice(nodes_list)
            v = rng.choice(nodes_list)
            if u != v and (int(u), int(v)) not in existing:
                dist = math.dist(network.coords[int(u)], network.coords[int(v)])
                tt = round(dist / 0.8, 2)  # rough travel time
                network.add_edge(int(u), int(v),
                                 road_length=round(dist * 0.1, 2),
                                 base_travel_time=tt,
                                 current_travel_time=tt,
                                 traffic_level=0.3,
                                 road_type="arterial",
                                 speed_limit=60,
                                 num_lanes=2)

        return network

    # ------------------------------------------------------------------
    # Connectivity helpers
    # ------------------------------------------------------------------
    def find_reachable(self, start: int) -> set[int]:
        """BFS to find all nodes reachable from start."""
        visited = set()
        queue = [start]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            for nbr in self.get_neighbors(node):
                if nbr not in visited:
                    queue.append(nbr)
        return visited

    def __repr__(self) -> str:
        return f"RoadNetwork(nodes={self.num_nodes}, edges={self.num_edges})"
