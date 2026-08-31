"""Cheap first-pass agentic triage — routes alarms, does not hardcode dismissal rules."""

from __future__ import annotations

import json
from typing import Literal, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from aoob_agent.agent import build_llm
from aoob_agent.case_compiler import CaseFile, compile_case
from aoob_agent.data_store import DataStore
from aoob_agent.session import begin_session, clear_session, get_session
from aoob_agent.tools import bind_store, get_micro_window, run_simulation

PREFILTER_PROMPT = """You are a lightweight AOOB alarm triage assistant.

Given a micro-window slice and raw bounds facts, output ONLY a JSON object:
{"route": "likely_false" | "needs_full_investigation" | "uncertain", "rationale": "<one sentence>"}

Rules:
- likely_false: strong visible guard/loop bound AND mechanical derived_max < declared_array_capacity
- needs_full_investigation: gaps, write_helpers, derived_max >= capacity, or ambiguous guards
- uncertain: insufficient context in the micro-window

Do NOT output a final false/true verdict — only the route."""


Route = Literal["likely_false", "needs_full_investigation", "uncertain"]


def _extract_route(text: str) -> tuple[Route, str]:
    text = (text or "").strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            route = str(data.get("route") or "uncertain").strip().lower()
            rationale = str(data.get("rationale") or "")
            if route in {"likely_false", "needs_full_investigation", "uncertain"}:
                return route, rationale  # type: ignore[return-value]
    except json.JSONDecodeError:
        pass
    lower = text.lower()
    for candidate in ("likely_false", "needs_full_investigation", "uncertain"):
        if candidate in lower:
            return candidate, text[:200]  # type: ignore[return-value]
    return "uncertain", text[:200]


def prefilter_alarm(
    order_id: int,
    store: DataStore,
    *,
    model: Optional[str] = None,
    case: Optional[CaseFile] = None,
    manage_session: bool = True,
) -> dict:
    """One cheap LLM call + two tools max to route an alarm."""
    from aoob_agent.token_metrics import naive_full_file_tokens

    owned_session = False
    if manage_session:
        clear_session()
        owned_session = True

    active_case = case or compile_case(store, order_id)
    if manage_session:
        begin_session(active_case, naive_full_file_tokens=naive_full_file_tokens(store.source_lines))
    else:
        try:
            get_session()
        except RuntimeError:
            begin_session(
                active_case, naive_full_file_tokens=naive_full_file_tokens(store.source_lines)
            )

    bind_store(store)

    micro = json.loads(get_micro_window.invoke({"window_size": 15, "confidence_so_far": "low"}))
    sim = json.loads(run_simulation.invoke({"confidence_so_far": "low"}))

    llm = build_llm(model=model)
    payload = {
        "order_id": order_id,
        "indexed_object": active_case.indexed_object.name,
        "declared_size": active_case.array_size,
        "operand": active_case.operand_symbol,
        "gaps": active_case.gaps,
        "micro_window": micro,
        "bounds_facts": sim,
    }
    messages = [
        SystemMessage(content=PREFILTER_PROMPT),
        HumanMessage(content=json.dumps(payload, indent=2)),
    ]
    response = llm.invoke(messages)
    content = response.content if isinstance(response.content, str) else str(response.content)
    route, rationale = _extract_route(content)

    if owned_session:
        clear_session()

    return {
        "order_id": order_id,
        "route": route,
        "rationale": rationale,
        "bounds_facts": sim,
        "gaps": list(active_case.gaps),
    }
