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
