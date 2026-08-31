"""Export complete CF graph with precomputed layout for fast WebGL rendering."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Optional

import networkx as nx

from aoob_agent.data_store import DataStore
from cf_viz.graph_builder import build_control_flow_graph, graph_stats


def _degree_sizes(graph: nx.DiGraph) -> dict[str, float]:
    # Combined degree for visual weight.
    sizes: dict[str, float] = {}
    for n in graph.nodes:
        d = graph.degree(n)
        # Keep tiny by default so 10k nodes stay light.
        sizes[n] = 1.6 + min(6.0, math.sqrt(d))
    return sizes


def compute_layout(graph: nx.DiGraph, *, seed: int = 42) -> dict[str, tuple[float, float]]:
    """
    Fast deterministic layout for large CF graphs.

    Uses a community-ish packing via BFS layers from high-degree hubs instead of
    expensive force simulation in the browser.
    """
    if graph.number_of_nodes() == 0:
        return {}

    rng = random.Random(seed)
    und = graph.to_undirected()

    # Pick hubs as anchors (top degree nodes).
    degrees = sorted(und.degree, key=lambda x: x[1], reverse=True)
    hubs = [n for n, _ in degrees[: min(24, len(degrees))]]

    # Place hubs on a ring.
    pos: dict[str, tuple[float, float]] = {}
    hub_r = 1200.0
    for i, h in enumerate(hubs):
        a = (2 * math.pi * i) / max(1, len(hubs))
        pos[h] = (hub_r * math.cos(a), hub_r * math.sin(a))

    # BFS from hubs; place remaining nodes in expanding rings around nearest hub seed.
    assigned = set(pos)
    queue: list[tuple[str, str, int]] = []  # node, root_hub, depth
    for h in hubs:
        for nb in und.neighbors(h):
            if nb not in assigned:
                queue.append((nb, h, 1))

    while queue:
        node, root, depth = queue.pop(0)
        if node in assigned:
            continue
        rx, ry = pos[root]
        # Spiral / jitter around root by depth.
        ang = rng.random() * 2 * math.pi
        rad = 40.0 * depth + rng.random() * 28.0
        pos[node] = (rx + rad * math.cos(ang), ry + rad * math.sin(ang))
        assigned.add(node)
        if depth < 8:
            for nb in und.neighbors(node):
                if nb not in assigned:
                    queue.append((nb, root, depth + 1))

    # Leftovers (disconnected bits): place in outer cloud.
    leftovers = [n for n in graph.nodes if n not in pos]
    outer = 1600.0
    for i, n in enumerate(leftovers):
        a = (2 * math.pi * i) / max(1, len(leftovers))
        jitter = rng.random() * 120.0
        pos[n] = ((outer + jitter) * math.cos(a), (outer + jitter) * math.sin(a))

    return pos


def graph_to_vis_payload(graph: nx.DiGraph) -> dict[str, Any]:
    """Lean payload: nodes=[id,x,y,s], edges=[from,to] — for sigma WebGL."""
    pos = compute_layout(graph)
    sizes = _degree_sizes(graph)

    # Stable id index for compact edges (optional). Keep string ids for highlight API simplicity.
    nodes = []
    for n in graph.nodes:
        x, y = pos[n]
        nodes.append({"id": n, "x": round(x, 2), "y": round(y, 2), "s": round(sizes[n], 2)})

    edges = []
    for u, v, data in graph.edges(data=True):
        edges.append(
            {
                "id": f"{u}->{v}",
                "s": u,
                "t": v,
                "w": int(data.get("count", 1)),
            }
        )

    return {
        "version": 2,
        "renderer": "sigma-webgl",
        "nodes": nodes,
        "edges": edges,
        "stats": graph_stats(graph),
    }


def build_and_export_full_graph(
    data_dir: Path,
    output_path: Path,
    *,
    process: Optional[str] = None,
) -> dict[str, Any]:
    store = DataStore.load(data_dir)
    graph = build_control_flow_graph(store, process=process)
    payload = graph_to_vis_payload(graph)
    payload["process_filter"] = process
    payload["data_dir"] = str(data_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Compact JSON — large file but parses fine; no pretty indent.
    output_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    return payload


def load_graph_payload(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
