"""Highlight payload for an alarm Order id on the full CF graph."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional

import networkx as nx

from aoob_agent.data_store import DataStore
from cf_viz.graph_builder import (
    bare_variable_name,
    build_variable_path_overlay,
    resolve_variable_from_alarm,
)


def _alarm_index_by_line(store: DataStore) -> dict[int, list[dict[str, Any]]]:
    by_line: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for alarm in store.alarms.values():
        parsed = alarm.parsed_location
        if not parsed:
            continue
        by_line[parsed["line"]].append(
            {
                "order": alarm.order,
                "type": alarm.type,
                "category": alarm.category,
                "location": alarm.location,
                "line": parsed["line"],
                "col_start": parsed["col_start"],
                "col_end": parsed["col_end"],
            }
        )
    for line in by_line:
        by_line[line].sort(key=lambda a: a["order"])
    return by_line


def _access_kind(access: str) -> str:
    a = (access or "").strip().lower()
    if a == "read":
        return "read"
    if a == "write":
        return "write"
    return "other"


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

    overlay = build_variable_path_overlay(
        graph,
        store,
        chosen,
        process=process,
        max_events=max_events,
    )

    alarms_by_line = _alarm_index_by_line(store)
    sequence_nodes = list(overlay.function_sequence)
    bridge_nodes = sorted(overlay.path_nodes - set(overlay.function_sequence))
    path_edge_ids = [f"{u}->{v}" for u, v in sorted(overlay.path_edges)]

    # Per-function access mix for graph coloring.
    access_by_function: dict[str, set[str]] = defaultdict(set)
    for ev in overlay.events:
        if ev.function:
            access_by_function[ev.function].add(_access_kind(ev.access))

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

    enriched_events: list[dict[str, Any]] = []
    alarms_in_sequence: list[dict[str, Any]] = []
    seen_alarm_orders: set[int] = set()

    for e in overlay.events:
        kind = _access_kind(e.access)
        at_line = list(alarms_by_line.get(e.line or -1, []))
        for a in at_line:
            if a["order"] not in seen_alarm_orders:
                seen_alarm_orders.add(a["order"])
                alarms_in_sequence.append(
                    {
                        **a,
                        "at_function": e.function,
                        "at_access": kind,
                        "df_event_order": e.order,
                        "is_query_alarm": a["order"] == order_id,
                    }
                )
        enriched_events.append(
            {
                "order": e.order,
                "function": e.function,
                "access": e.access,
                "access_kind": kind,
                "location": e.location,
                "line": e.line,
                "process": e.process,
                "alarms": at_line,
                "is_first": e.order == 1,
            }
        )

    # Also include the query alarm if its line wasn't already covered via DF events
    # (e.g. alarm columns differ slightly from DF location span).
    query_parsed = alarm.parsed_location
    if query_parsed and order_id not in seen_alarm_orders:
        alarms_in_sequence.append(
            {
                "order": alarm.order,
                "type": alarm.type,
                "category": alarm.category,
                "location": alarm.location,
                "line": query_parsed["line"],
                "col_start": query_parsed["col_start"],
                "col_end": query_parsed["col_end"],
                "at_function": None,
                "at_access": None,
                "df_event_order": None,
                "is_query_alarm": True,
            }
        )
        seen_alarm_orders.add(order_id)

    alarms_in_sequence.sort(
        key=lambda a: (a.get("line") is None, a.get("line") or 0, a["order"])
    )

    first = enriched_events[0] if enriched_events else None

    # Compact step timeline: collapse consecutive same-function+access rows.
    timeline: list[dict[str, Any]] = []
    for ev in enriched_events:
        if (
            timeline
            and timeline[-1]["function"] == ev["function"]
            and timeline[-1]["access_kind"] == ev["access_kind"]
        ):
            timeline[-1]["count"] += 1
            timeline[-1]["last_order"] = ev["order"]
            timeline[-1]["last_location"] = ev["location"]
            # merge alarms
            existing = {a["order"] for a in timeline[-1]["alarms"]}
            for a in ev["alarms"]:
                if a["order"] not in existing:
                    timeline[-1]["alarms"].append(a)
                    existing.add(a["order"])
        else:
            timeline.append(
                {
                    "step": len(timeline) + 1,
                    "function": ev["function"],
                    "access_kind": ev["access_kind"],
                    "access": ev["access"],
                    "count": 1,
                    "first_order": ev["order"],
                    "last_order": ev["order"],
                    "location": ev["location"],
                    "last_location": ev["location"],
                    "line": ev["line"],
                    "alarms": list(ev["alarms"]),
                    "is_first": ev["is_first"],
                }
            )

    return {
        "alarm": {
            "order": alarm.order,
            "type": alarm.type,
            "category": alarm.category,
            "location": alarm.location,
        },
        "site_variables": site_vars,
        "variable": overlay.variable or bare_variable_name(chosen),
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
            "events": len(enriched_events),
            "reads": sum(1 for e in enriched_events if e["access_kind"] == "read"),
            "writes": sum(1 for e in enriched_events if e["access_kind"] == "write"),
            "alarms": len(alarms_in_sequence),
            "timeline_steps": len(timeline),
        },
    }
