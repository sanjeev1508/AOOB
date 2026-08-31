"""Control-flow graph construction for the WebGL UI."""

from __future__ import annotations

from typing import Optional

import networkx as nx

from aoob_agent.data_store import DataStore


def bare_variable_name(name: str) -> str:
    return (name or "").split("@", 1)[0].strip().strip('"')


def build_control_flow_graph(
    store: DataStore,
    *,
    process: Optional[str] = None,
) -> nx.DiGraph:
    g = nx.DiGraph()
    for caller, rows in store.control_by_caller.items():
        for rec in rows:
            if process and rec.process != process:
                continue
            if not caller or not rec.callee:
                continue
            g.add_node(caller, kind="function")
            g.add_node(rec.callee, kind="function")
            if g.has_edge(caller, rec.callee):
                data = g.edges[caller, rec.callee]
                data["count"] = int(data.get("count", 1)) + 1
                sites = data.setdefault("call_sites", [])
                if rec.call_site and rec.call_site not in sites:
                    sites.append(rec.call_site)
            else:
                g.add_edge(
                    caller,
                    rec.callee,
                    count=1,
                    call_sites=[rec.call_site] if rec.call_site else [],
                    line=rec.line,
                )
    return g


def graph_stats(graph: nx.DiGraph) -> dict:
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "weakly_connected_components": (
            nx.number_weakly_connected_components(graph)
            if graph.number_of_nodes()
            else 0
        ),
    }


def resolve_variable_from_alarm(
    store: DataStore,
    alarm_order_id: int,
) -> tuple[Optional[str], list[str]]:
    alarm = store.get_alarm(alarm_order_id)
    if alarm is None or not alarm.parsed_location:
        return None, []
    line = alarm.parsed_location["line"]
    vars_at_site: list[str] = []
    seen: set[str] = set()
    for rec in store.data_flow_by_line.get(line, []):
        bare = bare_variable_name(rec.variable)
        if bare and bare not in seen:
            seen.add(bare)
            vars_at_site.append(bare)
    return (vars_at_site[0] if vars_at_site else None), vars_at_site


def find_path_segment(
    graph: nx.DiGraph,
    source: str,
    target: str,
    max_depth: int = 5,
) -> dict:
    """Resolve direct edges, directed multi-hop bridges, or shared-caller V-bridges between two functions."""
    if source not in graph or target not in graph:
        return {
            "from": source,
            "to": target,
            "kind": "disconnected",
            "nodes": [source, target],
            "edges": [],
            "bridge_nodes": [],
        }

    if source == target:
        return {
            "from": source,
            "to": target,
            "kind": "same_node",
            "nodes": [source],
            "edges": [],
            "bridge_nodes": [],
        }

    # 1. Direct Edge
    if graph.has_edge(source, target):
        return {
            "from": source,
            "to": target,
            "kind": "direct",
            "nodes": [source, target],
            "edges": [f"{source}->{target}"],
            "bridge_nodes": [],
        }

    # 2. Directed Shortest Path
    if nx.has_path(graph, source, target):
        try:
            path = nx.shortest_path(graph, source, target)
            if len(path) - 1 <= max_depth:
                edges = [f"{u}->{v}" for u, v in zip(path, path[1:])]
                bridges = path[1:-1]
                return {
                    "from": source,
                    "to": target,
                    "kind": "directed_bridge",
                    "nodes": path,
                    "edges": edges,
                    "bridge_nodes": bridges,
                }
        except (nx.NetworkXError, nx.NodeNotFound):
            pass

    # 3. Reverse Shortest Path (caller hop)
    if nx.has_path(graph, target, source):
        try:
            rpath = nx.shortest_path(graph, target, source)
            if len(rpath) - 1 <= max_depth:
                path = list(reversed(rpath))
                edges = [f"{u}->{v}" for u, v in zip(rpath, rpath[1:])]
                bridges = rpath[1:-1]
                return {
                    "from": source,
                    "to": target,
                    "kind": "reverse_bridge",
                    "nodes": path,
                    "edges": edges,
                    "bridge_nodes": bridges,
                }
        except (nx.NetworkXError, nx.NodeNotFound):
            pass

    # 4. Shared Caller V-Bridge (both called by common parent)
    source_callers = set(graph.predecessors(source))
    target_callers = set(graph.predecessors(target))
    common_callers = source_callers & target_callers
    if common_callers:
        caller = sorted(common_callers)[0]
        return {
            "from": source,
            "to": target,
            "kind": "shared_caller_bridge",
            "nodes": [source, caller, target],
            "edges": [f"{caller}->{source}", f"{caller}->{target}"],
            "bridge_nodes": [caller],
        }

    # 5. Shared Callee A-Bridge (both call common child)
    source_callees = set(graph.successors(source))
    target_callees = set(graph.successors(target))
    common_callees = source_callees & target_callees
    if common_callees:
        callee = sorted(common_callees)[0]
        return {
            "from": source,
            "to": target,
            "kind": "shared_callee_bridge",
            "nodes": [source, callee, target],
            "edges": [f"{source}->{callee}", f"{target}->{callee}"],
            "bridge_nodes": [callee],
        }

    # 6. Undirected path fallback
    undirected = graph.to_undirected(reciprocal=False)
    if nx.has_path(undirected, source, target):
        try:
            upath = nx.shortest_path(undirected, source, target)
            if len(upath) - 1 <= max_depth:
                edges = []
                for u, v in zip(upath, upath[1:]):
                    if graph.has_edge(u, v):
                        edges.append(f"{u}->{v}")
                    elif graph.has_edge(v, u):
                        edges.append(f"{v}->{u}")
                bridges = upath[1:-1]
                return {
                    "from": source,
                    "to": target,
                    "kind": "undirected_bridge",
                    "nodes": upath,
                    "edges": edges,
                    "bridge_nodes": bridges,
                }
        except (nx.NetworkXError, nx.NodeNotFound):
            pass

    return {
        "from": source,
        "to": target,
        "kind": "disconnected",
        "nodes": [source, target],
        "edges": [],
        "bridge_nodes": [],
    }


def find_all_possible_paths(
    graph: nx.DiGraph,
    source: str,
    target: str,
    max_depth: int = 6,
    max_paths: int = 12,
) -> list[dict]:
    """Find all simple directed control-flow paths between origin and alarm function."""
    if source not in graph or target not in graph or source == target:
        return []
    paths = []
    try:
        if nx.has_path(graph, source, target):
            simple_paths = nx.all_simple_paths(graph, source, target, cutoff=max_depth)
            for i, p in enumerate(simple_paths):
                if i >= max_paths:
                    break
                edges = [f"{u}->{v}" for u, v in zip(p, p[1:])]
                paths.append({
                    "path_id": i + 1,
                    "nodes": p,
                    "edges": edges,
                    "length": len(p),
                    "is_directed": True,
                })
        if not paths:
            # Try undirected fallback for reachable graph components
            undirected = graph.to_undirected(reciprocal=False)
            if nx.has_path(undirected, source, target):
                usimple = nx.all_simple_paths(undirected, source, target, cutoff=max_depth)
                for i, p in enumerate(usimple):
                    if i >= max_paths:
                        break
                    edges = []
                    for u, v in zip(p, p[1:]):
                        if graph.has_edge(u, v):
                            edges.append(f"{u}->{v}")
                        elif graph.has_edge(v, u):
                            edges.append(f"{v}->{u}")
                    paths.append({
                        "path_id": i + 1,
                        "nodes": p,
                        "edges": edges,
                        "length": len(p),
                        "is_directed": False,
                    })
    except Exception:
        pass
    return paths

