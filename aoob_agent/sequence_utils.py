"""Shared data-flow sequence builders used by agent tools and cf_viz UI."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional

from aoob_agent.data_store import DataFlowRecord, DataStore
from cf_viz.graph_builder import bare_variable_name, resolve_variable_from_alarm


def _access_kind(access: str) -> str:
    a = (access or "").strip().lower()
    if a == "read":
        return "read"
    if a == "write":
        return "write"
    return "other"


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


def _collect_records(
    store: DataStore,
    variable: str,
    *,
    process: Optional[str] = None,
    function_scope: Optional[str] = None,
) -> list[DataFlowRecord]:
    key = (variable or "").strip()
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
            if function_scope and rec.function != function_scope:
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
    return records


def _events_from_records(
    records: list[DataFlowRecord],
    *,
    alarms_by_line: dict[int, list[dict[str, Any]]],
    order_id: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int], Optional[dict[str, Any]]]:
    enriched_events: list[dict[str, Any]] = []
    alarms_in_sequence: list[dict[str, Any]] = []
    seen_alarm_orders: set[int] = set()
    reads = 0
    writes = 0

    for i, rec in enumerate(records, start=1):
        kind = _access_kind(rec.access)
        if kind == "read":
            reads += 1
        elif kind == "write":
            writes += 1
        at_line = list(alarms_by_line.get(rec.line or -1, []))
        for alarm in at_line:
            if alarm["order"] in seen_alarm_orders:
                continue
            seen_alarm_orders.add(alarm["order"])
            alarms_in_sequence.append(
                {
                    **alarm,
                    "at_function": rec.function,
                    "at_access": kind,
                    "df_event_order": i,
                    "is_query_alarm": alarm["order"] == order_id,
                }
            )
        enriched_events.append(
            {
                "order": i,
                "function": rec.function,
                "access": rec.access,
                "access_kind": kind,
                "location": rec.location,
                "line": rec.line,
                "process": rec.process,
                "alarms": at_line,
                "is_first": i == 1,
            }
        )

    counts = {
        "events": len(enriched_events),
        "reads": reads,
        "writes": writes,
        "alarms": len(alarms_in_sequence),
    }
    first = enriched_events[0] if enriched_events else None
    return enriched_events, alarms_in_sequence, counts, first


def _collapse_timeline(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    timeline: list[dict[str, Any]] = []
    for ev in events:
        if (
            timeline
            and timeline[-1]["function"] == ev["function"]
            and timeline[-1]["access_kind"] == ev["access_kind"]
        ):
            timeline[-1]["count"] += 1
            timeline[-1]["last_order"] = ev["order"]
            timeline[-1]["last_location"] = ev["location"]
            existing = {a["order"] for a in timeline[-1]["alarms"]}
            for alarm in ev["alarms"]:
                if alarm["order"] not in existing:
                    timeline[-1]["alarms"].append(alarm)
                    existing.add(alarm["order"])
            continue

        timeline.append(
            {
                "step": len(timeline) + 1,
                "function": ev["function"],
                "line": ev["line"],
                "count": 1,
                "first_order": ev["order"],
                "last_order": ev["order"],
                "location": ev["location"],
                "last_location": ev["location"],
                "alarms": list(ev["alarms"]),
                "is_first": ev["is_first"],
                "access_kind": ev["access_kind"],
                "access": ev["access_kind"],
            }
        )

    return timeline


def build_cross_function_df_payload(
    store: DataStore,
    alarm_order_id: int,
    *,
    variable: Optional[str] = None,
    process: Optional[str] = None,
    function_scope: Optional[str] = None,
    max_events: int = 300,
) -> dict[str, Any]:
    alarm = store.get_alarm(alarm_order_id)
    if alarm is None:
        return {"error": f"Unknown alarm Order id: {alarm_order_id}"}

    primary, site_vars = resolve_variable_from_alarm(store, alarm_order_id)
    chosen = (variable or primary or "").strip()
    if not chosen:
        return {
            "error": (
                f"No data-flow variable found at alarm {alarm_order_id} "
                f"({alarm.location})."
            ),
            "alarm": {
                "order": alarm.order,
                "category": alarm.category,
                "location": alarm.location,
            },
            "site_variables": site_vars,
            "variable": "",
            "events": [],
            "timeline": [],
            "counts": {"events": 0, "reads": 0, "writes": 0, "alarms": 0, "timeline_steps": 0},
            "first_access": None,
            "alarms_in_sequence": [],
            "truncated": False,
        }

    all_records = _collect_records(
        store,
        chosen,
        process=process,
        function_scope=function_scope,
    )
    max_n = max(50, min(int(max_events), 4000))
    records = all_records[:max_n]
    truncated = len(all_records) > len(records)

    alarms_by_line = _alarm_index_by_line(store)
    events, alarms_in_sequence, counts, first = _events_from_records(
        records,
        alarms_by_line=alarms_by_line,
        order_id=alarm_order_id,
    )

    if alarm.parsed_location and alarm.order not in {a["order"] for a in alarms_in_sequence}:
        parsed = alarm.parsed_location
        alarms_in_sequence.append(
            {
                "order": alarm.order,
                "type": alarm.type,
                "category": alarm.category,
                "location": alarm.location,
                "line": parsed["line"],
                "col_start": parsed["col_start"],
                "col_end": parsed["col_end"],
                "at_function": None,
                "at_access": None,
                "df_event_order": None,
                "is_query_alarm": True,
            }
        )

    alarms_in_sequence.sort(
        key=lambda a: (a.get("line") is None, a.get("line") or 0, a["order"])
    )

    timeline = _collapse_timeline(events)
    counts["timeline_steps"] = len(timeline)

    return {
        "alarm": {
            "order": alarm.order,
            "type": alarm.type,
            "category": alarm.category,
            "location": alarm.location,
        },
        "function_scope": function_scope,
        "site_variables": site_vars,
        "variable": bare_variable_name(chosen),
        "events": events,
        "timeline": timeline,
        "counts": counts,
        "first_access": first,
        "alarms_in_sequence": alarms_in_sequence,
        "truncated": truncated,
    }


def build_trimmed_writes_alarm_sequence(
    store: DataStore,
    alarm_order_id: int,
    *,
    variable: Optional[str] = None,
    process: Optional[str] = None,
    function_scope: Optional[str] = None,
    max_events: int = 500,
) -> dict[str, Any]:
    payload = build_cross_function_df_payload(
        store,
        alarm_order_id,
        variable=variable,
        process=process,
        function_scope=function_scope,
        max_events=max_events,
    )
    if payload.get("error"):
        return {
            "alarm_order_id": alarm_order_id,
            "variable": (variable or "").strip(),
            "sequence": [],
            "total_steps": 0,
            "writes": 0,
            "reads": 0,
            "alarms_on_path": [],
            "origin_resolved": False,
            "truncated": bool(payload.get("truncated")),
            "note": payload["error"],
        }

    timeline = payload.get("timeline") or []
    query = payload.get("alarm") or {}
    query_order = query.get("order")

    sequence: list[dict[str, Any]] = []
    for row in timeline:
        alarms = row.get("alarms") or []
        alarm_orders = [int(a["order"]) for a in alarms if isinstance(a, dict) and "order" in a]
        is_alarm_site = query_order in alarm_orders
        access = row.get("access") or "other"

        if access not in {"write", "mixed"} and not is_alarm_site:
            continue
        if access == "read" and not is_alarm_site:
            continue

        sequence.append(
            {
                "step": len(sequence) + 1,
                "function": row.get("function"),
                "access": access,
                "location": row.get("location") or row.get("last_location") or "",
                "is_alarm_site": bool(is_alarm_site),
                "alarms_on_step": alarm_orders,
            }
        )

    origin_resolved = bool(
        sequence
        and sequence[0].get("is_alarm_site")
        and sequence[0].get("access") in {"write", "mixed"}
    )

    alarms_on_path = sorted(
        {
            int(order)
            for step in sequence
            for order in (step.get("alarms_on_step") or [])
            if isinstance(order, int)
        }
    )

    note = (
        "Trim rule applied from shared DF timeline: keep write/mixed steps plus "
        "the queried alarm step; drop unrelated pre-alarm reads."
    )
    if function_scope:
        note = (
            f"{note} Scope mode=function; records restricted to function "
            f"{function_scope!r}."
        )
    if not sequence:
        note = (
            "No matching DF records for variable in index. "
            "Sequence is empty and origin remains unresolved."
        )
        if function_scope:
            note = (
                f"{note} Scope mode=function; records restricted to function "
                f"{function_scope!r}."
            )

    counts = payload.get("counts") or {}
    return {
        "alarm_order_id": alarm_order_id,
        "variable": payload.get("variable") or (variable or "").strip(),
        "scope_mode": "function" if function_scope else "global",
        "trace_function": function_scope,
        "sequence": sequence,
        "total_steps": len(sequence),
        "writes": int(counts.get("writes") or 0),
        "reads": int(counts.get("reads") or 0),
        "alarms_on_path": alarms_on_path,
        "origin_resolved": origin_resolved,
        "truncated": bool(payload.get("truncated")),
        "note": note,
    }
