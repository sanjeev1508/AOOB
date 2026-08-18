"""Highlight payload for an alarm Order id on the full CF graph."""

from __future__ import annotations

from typing import Any, Optional

import networkx as nx

from aoob_agent.data_store import DataStore
from aoob_agent.sequence_utils import build_cross_function_df_payload
from cf_viz.graph_builder import (
    bare_variable_name,
    build_variable_path_overlay,
    resolve_variable_from_alarm,
)


def highlight_for_alarm(
    graph: nx.DiGraph,
    store: DataStore,
    order_id: int,
    *,
    variable: Optional[str] = None,
    process: Optional[str] = None,
    max_events: int = 300,
) -> dict[str, Any]:
    alarm = store.get_alarm(order_id)
    if alarm is None:
        return {"error": f"Unknown alarm Order id: {order_id}"}

    shared = build_cross_function_df_payload(
        store,
        order_id,
        variable=variable,
        process=process,
        max_events=max_events,
    )

    primary, site_vars = resolve_variable_from_alarm(store, order_id)
    chosen = (variable or primary or "").strip()
    if not chosen:
        return {
            "error": (
                f"No data-flow variable found at alarm {order_id} "
                f"({alarm.location})."
            ),
            "alarm": {
                "order": alarm.order,
                "category": alarm.category,
                "location": alarm.location,
            },
            "site_variables": site_vars,
        }

    if shared.get("error"):
        return shared

    overlay = build_variable_path_overlay(
        graph,
        store,
        chosen,
        process=process,
        max_events=max_events,
    )

    sequence_nodes = list(overlay.function_sequence)
    bridge_nodes = sorted(overlay.path_nodes - set(overlay.function_sequence))
    path_edge_ids = [f"{u}->{v}" for u, v in sorted(overlay.path_edges)]

    # Per-function access mix for graph coloring.
    access_by_function: dict[str, set[str]] = {}
    for ev in overlay.events:
        if ev.function:
            kinds = access_by_function.setdefault(ev.function, set())
            ak = (ev.access or "").strip().lower()
            kinds.add("read" if ak == "read" else "write" if ak == "write" else "other")

    function_access = {
        fn: (
            "write"
            if "write" in kinds and "read" not in kinds
            else "read"
            if "read" in kinds and "write" not in kinds
            else "mixed"
            if "read" in kinds and "write" in kinds
            else "other"
        )
        for fn, kinds in access_by_function.items()
    }

    enriched_events = shared.get("events") or []
    alarms_in_sequence = shared.get("alarms_in_sequence") or []
    first = shared.get("first_access")
    timeline = shared.get("timeline") or []
    shared_counts = shared.get("counts") or {}

    return {
        "alarm": {
            "order": alarm.order,
            "type": alarm.type,
            "category": alarm.category,
            "location": alarm.location,
        },
        "site_variables": shared.get("site_variables") or site_vars,
        "variable": shared.get("variable") or overlay.variable or bare_variable_name(chosen),
        "first_access": first,
        "function_sequence": sequence_nodes,
        "function_access": function_access,
        "bridge_nodes": bridge_nodes,
        "highlight_nodes": sorted(overlay.path_nodes),
        "highlight_edge_ids": path_edge_ids,
        "segments": [
            {
                "from": s.from_function,
                "to": s.to_function,
                "kind": s.kind,
                "nodes": s.nodes,
                "edges": [f"{u}->{v}" for u, v in s.edges],
            }
            for s in overlay.segments
        ],
        "events": enriched_events,
        "timeline": timeline,
        "alarms_in_sequence": alarms_in_sequence,
        "counts": {
            "events": int(shared_counts.get("events", len(enriched_events))),
            "reads": int(shared_counts.get("reads", 0)),
            "writes": int(shared_counts.get("writes", 0)),
            "alarms": len(alarms_in_sequence),
            "timeline_steps": len(timeline),
        },
    }
