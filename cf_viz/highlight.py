"""Highlight payload: compiled index origin path on the CF graph."""

from __future__ import annotations

from typing import Any, Optional

import networkx as nx

from aoob_agent.case_compiler import compile_case
from aoob_agent.data_store import DataStore
from cf_viz.graph_builder import resolve_variable_from_alarm


def highlight_for_alarm(
    graph: nx.DiGraph,
    store: DataStore,
    order_id: int,
    *,
    variable: Optional[str] = None,
) -> dict[str, Any]:
    alarm = store.get_alarm(order_id)
    if alarm is None:
        return {"error": f"Unknown alarm Order id: {order_id}"}

    try:
        case = compile_case(store, order_id)
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc), "alarm": {"order": alarm.order, "location": alarm.location}}

    _primary, site_vars = resolve_variable_from_alarm(store, order_id)
    # Default highlight is the compiled index path, not the array DF dump.
    # An explicit variable override still names the operand the UI asked for,
    # but the path stays the case-file walk.
    chosen = (variable or case.operand_symbol or case.indexed_object.name).strip()

    sequence_nodes = [s.function for s in case.path if s.function]
    # Collapse consecutive duplicates.
    collapsed: list[str] = []
    for fn in sequence_nodes:
        if not collapsed or collapsed[-1] != fn:
            collapsed.append(fn)
    sequence_nodes = collapsed

    function_access: dict[str, str] = {}
    for step in case.path:
        if not step.function:
            continue
        if step.role == "origin":
            function_access[step.function] = "write"
        elif step.role == "alarm":
            prev = function_access.get(step.function)
            function_access[step.function] = "mixed" if prev == "write" else "read"
        else:
            function_access[step.function] = "mixed"

    segments: list[dict[str, Any]] = []
    path_edge_ids: list[str] = []
    for a, b in zip(sequence_nodes, sequence_nodes[1:]):
        kind = "disconnected"
        nodes = [a, b]
        edges: list[str] = []
        if a in graph and b in graph and graph.has_edge(a, b):
            kind = "direct"
            edges = [f"{a}->{b}"]
            path_edge_ids.append(f"{a}->{b}")
        segments.append(
            {"from": a, "to": b, "kind": kind, "nodes": nodes, "edges": edges}
        )

    timeline = []
    for step in case.path:
        timeline.append(
            {
                "step": step.step,
                "function": step.function,
                "line": step.line,
                "count": 1,
                "first_order": step.step,
                "last_order": step.step,
                "location": step.location,
                "last_location": step.location,
                "alarms": (
                    [
                        {
                            "order": alarm.order,
                            "type": alarm.type,
                            "category": alarm.category,
                            "location": alarm.location,
                            "is_query_alarm": True,
                        }
                    ]
                    if step.role in {"alarm", "origin_and_alarm"}
                    else []
                ),
                "is_first": step.step == 1,
                "access_kind": "write" if step.access == "write" else "read" if step.access == "read" else "mixed",
                "access": step.access,
                "role": step.role,
                "symbol": step.symbol,
            }
        )

    first = None
    if case.path:
        s0 = case.path[0]
        first = {
            "order": 1,
            "function": s0.function,
            "access": s0.access,
            "access_kind": function_access.get(s0.function, s0.access),
            "location": s0.location,
            "line": s0.line,
            "process": "",
            "alarms": [],
            "is_first": True,
            "role": s0.role,
            "symbol": s0.symbol,
        }

    alarms_in_sequence = [
        {
            "order": alarm.order,
            "type": alarm.type,
            "category": alarm.category,
            "location": alarm.location,
            "line": case.alarm_line,
            "at_function": case.alarm_function,
            "at_access": "read",
            "is_query_alarm": True,
        }
    ]

    obj = case.indexed_object
    return {
        "alarm": {
            "order": alarm.order,
            "type": alarm.type,
            "category": alarm.category,
            "location": alarm.location,
        },
        "site_variables": site_vars,
        "variable": chosen,
        "operand_symbol": case.operand_symbol,
        "index_expression": case.index_expression,
        "indexed_object": obj.name,
        "object_size": obj.array_size,
        "object_dtype": obj.declared_type,
        "case_gaps": list(case.gaps),
        "helpers": list(case.helpers),
        "first_access": first,
        "function_sequence": sequence_nodes,
        "function_access": function_access,
        "bridge_nodes": [],
        "highlight_nodes": sequence_nodes,
        "highlight_edge_ids": path_edge_ids,
        "segments": segments,
        "events": [
            {
                "order": step.step,
                "function": step.function,
                "access": step.access,
                "access_kind": step.access,
                "location": step.location,
                "line": step.line,
                "role": step.role,
                "symbol": step.symbol,
            }
            for step in case.path
        ],
        "timeline": timeline,
        "alarms_in_sequence": alarms_in_sequence,
        "counts": {
            "events": len(case.path),
            "reads": sum(1 for s in case.path if s.access == "read"),
            "writes": sum(1 for s in case.path if s.access == "write"),
            "alarms": 1,
            "timeline_steps": len(case.path),
            "path_steps": len(case.path),
        },
    }
