"""Highlight payload: compiled index origin path on the CF graph."""

from __future__ import annotations

from typing import Any, Optional

import networkx as nx

from aoob_agent.case_compiler import compile_case
from aoob_agent.data_store import DataStore
from cf_viz.graph_builder import (
    find_all_possible_paths,
    find_path_segment,
    resolve_variable_from_alarm,
)


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
    chosen = (variable or case.operand_symbol or case.indexed_object.name).strip()

    # Step functions from primary origin path
    sequence_nodes: list[str] = []
    for step in case.path:
        if step.function and (not sequence_nodes or sequence_nodes[-1] != step.function):
            sequence_nodes.append(step.function)

    write_helpers = list(case.write_helpers)
    arg_read_helpers = list(case.arg_read_helpers)
    helpers = list(case.helpers)

    # 1. Write sequence: path steps where access == 'write' / role == origin + write_helpers + alarm function
    path_writes = [
        s.function
        for s in case.path
        if s.function and (s.access == "write" or s.role in {"origin", "origin_and_alarm"})
    ]
    write_sequence: list[str] = []
    for fn in path_writes + write_helpers + [case.alarm_function]:
        if fn and (not write_sequence or write_sequence[-1] != fn):
            write_sequence.append(fn)

    # 2. Full sequence: primary path + all helpers (write_helpers, arg_read_helpers, index helpers)
    full_sequence: list[str] = []
    for fn in sequence_nodes + write_helpers + arg_read_helpers + helpers:
        if fn and fn not in full_sequence:
            full_sequence.append(fn)

    function_roles: dict[str, str] = {}
    function_access: dict[str, str] = {}

    for step in case.path:
        if not step.function:
            continue
        prev = function_roles.get(step.function)
        if step.role == "origin_and_alarm" or (
            prev == "origin" and step.role == "alarm"
        ) or (prev == "alarm" and step.role == "origin"):
            function_roles[step.function] = "origin_and_alarm"
        elif prev in {None, "hop"}:
            function_roles[step.function] = step.role

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

    # Mark helper functions in function_roles & function_access
    for fn in write_helpers:
        if fn not in function_roles:
            function_roles[fn] = "write_helper"
        if fn not in function_access:
            function_access[fn] = "write"
    for fn in arg_read_helpers:
        if fn not in function_roles:
            function_roles[fn] = "read_helper"
        if fn not in function_access:
            function_access[fn] = "read"
    for fn in helpers:
        if fn not in function_roles:
            function_roles[fn] = "helper"
        if fn not in function_access:
            function_access[fn] = "read"

    # Compute path segments for sequence_nodes using find_path_segment (bridges!)
    segments: list[dict[str, Any]] = []
    path_edge_ids: set[str] = set()
    bridge_nodes: set[str] = set()

    for a, b in zip(sequence_nodes, sequence_nodes[1:]):
        seg = find_path_segment(graph, a, b)
        segments.append(seg)
        for e in seg.get("edges", []):
            path_edge_ids.add(e)
        for bn in seg.get("bridge_nodes", []):
            bridge_nodes.add(bn)

    # Compute segments for full_sequence to connect helpers
    full_segments: list[dict[str, Any]] = []
    for a, b in zip(full_sequence, full_sequence[1:]):
        seg = find_path_segment(graph, a, b)
        full_segments.append(seg)
        for e in seg.get("edges", []):
            path_edge_ids.add(e)
        for bn in seg.get("bridge_nodes", []):
            bridge_nodes.add(bn)

    # Find ALL possible simple paths between origin function and alarm function
    origin_fn = sequence_nodes[0] if sequence_nodes else case.alarm_function
    alarm_fn = case.alarm_function
    alternate_paths = find_all_possible_paths(graph, origin_fn, alarm_fn, max_depth=6, max_paths=10)

    for ap in alternate_paths:
        for e in ap.get("edges", []):
            path_edge_ids.add(e)

    # All nodes to highlight in WebGL graph
    highlight_nodes = list(
        dict.fromkeys(
            sequence_nodes
            + full_sequence
            + write_sequence
            + list(bridge_nodes)
            + [n for ap in alternate_paths for n in ap.get("nodes", [])]
        )
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
                "access_kind": (
                    "write"
                    if step.access == "write"
                    else "read"
                    if step.access == "read"
                    else "mixed"
                ),
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
            "alarm_function": case.alarm_function,
        },
        "site_variables": site_vars,
        "variable": chosen,
        "operand_symbol": case.operand_symbol,
        "index_expression": case.index_expression,
        "indexed_object": obj.name,
        "object_size": obj.array_size,
        "object_dtype": obj.declared_type,
        "case_gaps": list(case.gaps),
        "helpers": helpers,
        "write_helpers": write_helpers,
        "arg_read_helpers": arg_read_helpers,
        "first_access": first,
        "function_sequence": sequence_nodes,
        "full_sequence": full_sequence,
        "write_sequence": write_sequence,
        "function_access": function_access,
        "function_roles": function_roles,
        "bridge_nodes": list(bridge_nodes),
        "highlight_nodes": highlight_nodes,
        "highlight_edge_ids": list(path_edge_ids),
        "segments": segments,
        "full_segments": full_segments,
        "alternate_paths": alternate_paths,
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
            "write_helpers": len(write_helpers),
            "helpers": len(helpers),
            "alternate_paths": len(alternate_paths),
            "alarms": 1,
            "timeline_steps": len(case.path),
            "path_steps": len(sequence_nodes),
        },
    }

