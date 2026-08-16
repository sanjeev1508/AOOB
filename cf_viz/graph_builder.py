"""Control-flow graph construction and data-flow variable path overlay."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Optional

import networkx as nx

from aoob_agent.data_store import DataFlowRecord, DataStore


@dataclass
class VariableAccessEvent:
    order: int
    variable: str
    function: str
    access: str
    location: str
    line: Optional[int]
    process: str


@dataclass
class CfPathSegment:
    from_function: str
    to_function: str
    nodes: list[str]
    edges: list[tuple[str, str]]
    kind: str  # direct | shortest | disconnected


@dataclass
class VariablePathOverlay:
    variable: str
    events: list[VariableAccessEvent]
    function_sequence: list[str]
    segments: list[CfPathSegment]
    path_nodes: set[str] = field(default_factory=set)
    path_edges: set[tuple[str, str]] = field(default_factory=set)
    access_by_function: dict[str, set[str]] = field(default_factory=dict)


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


def collect_variable_events(
    store: DataStore,
    variable: str,
    *,
    process: Optional[str] = None,
    max_events: int = 300,
) -> list[VariableAccessEvent]:
    key = variable.strip()
    bare = bare_variable_name(key)
    records: list[DataFlowRecord] = []
    seen: set[int] = set()
    for candidate in (key, bare):
        for rec in store.data_flow_by_variable.get(candidate, []):
            rid = id(rec)
            if rid in seen:
                continue
            seen.add(rid)
            if process and rec.process != process:
                continue
            records.append(rec)

    records.sort(
        key=lambda r: (
            r.line if r.line is not None else 10**12,
            r.access,
            r.function,
            r.location,
        )
    )
    events: list[VariableAccessEvent] = []
    for i, rec in enumerate(records[:max_events], start=1):
        events.append(
            VariableAccessEvent(
                order=i,
                variable=bare_variable_name(rec.variable) or bare,
                function=rec.function,
                access=rec.access,
                location=rec.location,
                line=rec.line,
                process=rec.process,
            )
        )
    return events


def function_sequence_from_events(events: list[VariableAccessEvent]) -> list[str]:
    seq: list[str] = []
    for ev in events:
        if not ev.function:
            continue
        if not seq or seq[-1] != ev.function:
            seq.append(ev.function)
    return seq


def connect_functions_in_cf(
    graph: nx.DiGraph,
    function_sequence: list[str],
) -> list[CfPathSegment]:
    segments: list[CfPathSegment] = []
    for a, b in zip(function_sequence, function_sequence[1:]):
        if a not in graph or b not in graph:
            segments.append(
                CfPathSegment(a, b, [a, b], [], "disconnected")
            )
            continue
        if graph.has_edge(a, b):
            segments.append(CfPathSegment(a, b, [a, b], [(a, b)], "direct"))
            continue
        nodes: list[str] = []
        edges: list[tuple[str, str]] = []
        kind = "disconnected"
        try:
            nodes = nx.shortest_path(graph, a, b)
            edges = list(zip(nodes, nodes[1:]))
            kind = "shortest"
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            try:
                und = graph.to_undirected()
                nodes = nx.shortest_path(und, a, b)
                edges = []
                for u, v in zip(nodes, nodes[1:]):
                    if graph.has_edge(u, v):
                        edges.append((u, v))
                    elif graph.has_edge(v, u):
                        edges.append((v, u))
                    else:
                        edges.append((u, v))
                kind = "shortest"
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                nodes = [a, b]
                edges = []
                kind = "disconnected"
        segments.append(CfPathSegment(a, b, nodes, edges, kind))
    return segments


def build_variable_path_overlay(
    graph: nx.DiGraph,
    store: DataStore,
    variable: str,
    *,
    process: Optional[str] = None,
    max_events: int = 300,
) -> VariablePathOverlay:
    events = collect_variable_events(
        store, variable, process=process, max_events=max_events
    )
    seq = function_sequence_from_events(events)
    segments = connect_functions_in_cf(graph, seq)
    path_nodes: set[str] = set(seq)
    path_edges: set[tuple[str, str]] = set()
    for seg in segments:
        path_nodes.update(seg.nodes)
        path_edges.update(seg.edges)
    access_by_function: dict[str, set[str]] = defaultdict(set)
    for ev in events:
        if ev.function:
            access_by_function[ev.function].add(ev.access)
    return VariablePathOverlay(
        variable=bare_variable_name(variable),
        events=events,
        function_sequence=seq,
        segments=segments,
        path_nodes=path_nodes,
        path_edges=path_edges,
        access_by_function={k: set(v) for k, v in access_by_function.items()},
    )


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
