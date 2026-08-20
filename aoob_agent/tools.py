"""Window-walk tools over a compiled AOOB case file."""

from __future__ import annotations

import json
from typing import Any, Optional

from langchain_core.tools import tool

from aoob_agent.data_store import DataStore
from aoob_agent.session import get_session

_STORE: Optional[DataStore] = None


def bind_store(store: DataStore) -> None:
    global _STORE
    _STORE = store


def _store() -> DataStore:
    if _STORE is None:
        raise RuntimeError("DataStore is not bound. Call bind_store() first.")
    return _STORE


def _dump(payload: dict) -> str:
    return json.dumps(payload, indent=2)


def _find_function_span(
    function_name: str, anchor_line: Optional[int] = None
) -> tuple[Optional[int], Optional[int]]:
    name = (function_name or "").strip()
    store = _store()
    if anchor_line is not None:
        enc_name, start, end = store.find_enclosing_function(anchor_line)
        if start is not None and end is not None and start <= int(anchor_line) <= end:
            return start, end
    if not name:
        return None, None
    matches: list[tuple[int, int]] = []
    for ln in range(1, len(store.source_lines)):
        if store._looks_like_function_header(ln) == name:
            end = store._function_end_from(ln)
            matches.append((ln, end))
    if anchor_line is not None:
        for start, end in matches:
            if start <= int(anchor_line) <= end:
                return start, end
    if matches:
        return matches[0]
    return None, None


def _snippet_for_function(
    function_name: str,
    *,
    anchor_line: Optional[int],
) -> dict[str, Any]:
    """Return the complete source span for one function (no context-window cut).

    `get_window` / `move_window` / `inspect` are the LLM's explicit "open this
    function" tools, so they always hand back the whole function body
    (start_line..end_line) rather than a clipped anchor-centered slice.
    `_find_function_span` (via the data store's brace matcher) already caps a
    single function at 1200 lines, so this stays bounded without an extra cut
    here. Bulk, budget-limited previews are `pack_path_windows`'s job, not
    this function's.
    """
    store = _store()
    start, end = _find_function_span(function_name, anchor_line=anchor_line)
    if start is None or end is None:
        return {"error": f"Could not resolve function span for {function_name!r}."}
    snippet_start, snippet_end = start, end
    lines = [
        {"line": ln, "text": txt}
        for ln, txt in store.get_source_slice(snippet_start, snippet_end)
    ]
    return {
        "function_name": function_name,
        "span": {
            "start_line": start,
            "end_line": end,
            "snippet_start": snippet_start,
            "snippet_end": snippet_end,
        },
        "snippet": lines,
    }


def _window_payload() -> dict[str, Any]:
    session = get_session()
    step = session.current_step()
    if step is None:
        return {"error": "Case path is empty."}
    session.mark_opened()
    body = _snippet_for_function(step.function, anchor_line=step.line)
    if body.get("error"):
        return body
    span = body.get("span") or {}
    fn_start, fn_end = span.get("start_line"), span.get("end_line")
    snip_start, snip_end = span.get("snippet_start"), span.get("snippet_end")
    span_truncated = (
        fn_start is not None
        and fn_end is not None
        and (snip_start > fn_start or snip_end < fn_end)
    )
    session.mark_truncated(span_truncated)
    return {
        "path_step": step.step,
        "path_length": session.n_steps,
        "role": step.role,
        "symbol": step.symbol,
        "scope": step.scope,
        "access": step.access,
        "location": step.location,
        "argument_expression": step.argument_expression,
        "call_site": step.call_site,
        "note": step.note,
        "guards": list(step.guards or session.case.guards),
        "case_gaps": list(session.case.gaps),
        "declared_size": session.case.array_size,
        "alarm_window_opened": session.alarm_window_opened(),
        "window_truncated": bool(span_truncated),
        "remaining_until_alarm": max(0, session.n_steps - 1 - session.current_index),
        **body,
    }


def pack_path_windows(*, max_lines_per_window: int = 80) -> str:
    """Open every compiled path step and return compact snippets (no LLM)."""
    session = get_session()
    saved = session.current_index
    parts: list[str] = []
    try:
        for i in range(session.n_steps):
            session.current_index = i
            payload = _window_payload()
            if payload.get("error"):
                parts.append(f"### Window {i + 1}: {payload['error']}")
                continue
            rows = payload.get("snippet") or []
            shown_rows = rows[:max_lines_per_window]
            body = "\n".join(
                f"{row.get('line')}|{row.get('text', '')}" for row in shown_rows
            )
            hidden = len(rows) - len(shown_rows)
            if hidden > 0:
                session.mark_truncated(True)
                body += (
                    f"\n[WINDOW TRUNCATED: {hidden} more line(s) of this function "
                    "were not shown. Do not assume the omitted lines don't modify "
                    "the operand — treat this as an evidence gap, not a green light.]"
                )
            span = payload.get("span") or {}
            fn_start, fn_end = span.get("start_line"), span.get("end_line")
            if fn_start is not None and fn_end is not None:
                snip_start, snip_end = span.get("snippet_start"), span.get("snippet_end")
                if snip_start is not None and (snip_start > fn_start or snip_end < fn_end):
                    body += (
                        f"\n[FUNCTION SPAN {fn_start}-{fn_end}; only "
                        f"{snip_start}-{snip_end} shown. Code outside this range "
                        "was not shown and may still affect the operand.]"
                    )
            guards = payload.get("guards") or []
            gtxt = ("\n  guards: " + " | ".join(guards[:6])) if guards else ""
            parts.append(
                f"### Window {payload.get('path_step')}/{payload.get('path_length')} "
                f"[{payload.get('role')}] {payload.get('function_name')}\n"
                f"symbol={payload.get('symbol')}  loc={payload.get('location')}"
                f"{gtxt}\n{body}"
            )
    finally:
        session.current_index = saved
        session.mark_opened()
    return "\n\n".join(parts) if parts else "(empty path)"


@tool
def get_window() -> str:
    """Return the source snippet for the current origin-path step.

    Starts at the index origin (first path step). Does not take a function
    name — the compiled case file owns the path.
    """
    get_session().mark_explicit_review()
    return _dump(_window_payload())


@tool
def move_window(direction: str = "next", step: int = 0) -> str:
    """Move to another step on the compiled origin → alarm path.

    direction: next (toward alarm) or prev (toward origin).
    step: 1-based path index; when set, overrides direction.
    Returns the new window snippet in the same call.
    """
    session = get_session()
    try:
        session.move(direction=direction, step=step)
    except ValueError as exc:
        return _dump({"error": str(exc)})
    session.mark_explicit_review()
    return _dump(_window_payload())


@tool
def inspect(function_name: str) -> str:
    """Open a helper named in the index expression (for example Mo_Inst_GetCoreIdx).

    Only helpers listed on the case file are allowed — this is not a free
    walk of the call graph.
    """
    session = get_session()
    name = (function_name or "").strip()
    if not session.can_inspect(name):
        allowed = session.case.helpers
        return _dump(
            {
                "error": (
                    f"{name!r} is not a listed index-expression helper. "
                    f"Allowed: {allowed}"
                ),
                "allowed_helpers": allowed,
            }
        )
    session.inspected.add(name)
    body = _snippet_for_function(name, anchor_line=None)
    if body.get("error"):
        return _dump(body)
    return _dump({"helper": name, "inspected": sorted(session.inspected), **body})


@tool
def submit_verdict(
    classification: str,
    comment: str,
    confidence: str = "medium",
    summary: str = "",
) -> str:
    """Submit the investigation verdict. Call only after the alarm window is open.

    classification: true | false | review
    Python does not invent the verdict; it only rejects premature submits.
    """
    session = get_session()
    cls = (classification or "").strip().lower()
    if cls in {"true", "t", "bug", "true_positive"}:
        cls = "true"
    elif cls in {"false", "f", "fp", "false_positive"}:
        cls = "false"
    elif cls in {"review", "uncertain", "unknown"}:
        cls = "review"
    else:
        return _dump(
            {
                "accepted": False,
                "error": "classification must be true, false, or review.",
            }
        )
    conf = (confidence or "medium").strip().lower()
    if conf not in {"low", "medium", "high"}:
        conf = "medium"

    if not session.alarm_window_opened():
        return _dump(
            {
                "accepted": False,
                "error": (
                    "Alarm window is not open. Call get_window and move_window "
                    "until the alarm function snippet is shown."
                ),
                "current_step": session.current_index + 1,
                "path_length": session.n_steps,
            }
        )
    obj = session.case.indexed_object
    gaps = set(session.case.gaps)
    size_unknown = obj.array_size is None
    incomplete = size_unknown or "object_undeclared" in gaps or "index_unparsed" in gaps
    listed = list(session.case.guards) + [
        g for s in session.case.path for g in (s.guards or [])
    ]
    has_if_guards = any(
        (g or "").split(" ", 1)[0].lower() in {"if", "for", "while"} for g in listed
    )
    has_origin_init = any((g or "").split(" ", 1)[0].lower() == "init" for g in listed)
    has_guards = has_if_guards or has_origin_init

    if cls in {"true", "false"} and incomplete:
        return _dump(
            {
                "accepted": False,
                "error": (
                    "Capacity/object is unresolved (gaps: "
                    f"{', '.join(session.case.gaps) or 'size_unknown'}). "
                    "Submit classification=review, not true/false."
                ),
                "case_gaps": list(session.case.gaps),
                "declared_size": obj.array_size,
            }
        )
    if cls == "true" and conf == "high" and has_if_guards:
        return _dump(
            {
                "accepted": False,
                "error": (
                    "Python extracted bound-style guards on this path. "
                    "Cannot submit true/high; use false if a guard keeps the index "
                    "inside capacity, or review."
                ),
                "guards": list(session.case.guards)[:8],
            }
        )
    if cls == "false" and conf == "high" and not has_if_guards and not has_origin_init:
        # Constant-folding / helper-range arguments are valid without an if().
        # Do not bounce the LLM into a parroted review; keep false at medium.
        conf = "medium"
    if (
        cls in {"true", "false"}
        and conf == "high"
        and not session.alarm_window_fully_reviewed()
    ):
        return _dump(
            {
                "accepted": False,
                "error": (
                    "The alarm step's window was truncated (see [WINDOW "
                    "TRUNCATED] / [FUNCTION SPAN] markers) and was never "
                    "re-opened with get_window/move_window. The omitted code "
                    "may still modify the operand. Call get_window or "
                    "move_window on that step to check it, or submit at "
                    "confidence=medium/low, or submit review."
                ),
                "alarm_step_index": session.alarm_step_index(),
            }
        )
    if cls == "true" and conf == "high" and "missing_df" in gaps and not has_guards:
        return _dump(
            {
                "accepted": False,
                "error": (
                    "Index origin was not traced. Submit review, not true/high."
                ),
                "case_gaps": list(session.case.gaps),
            }
        )

    verdict = {
        "classification": cls,
        "comment": comment or "",
        "confidence": conf,
        "summary": summary or comment or "",
        "alarm_order_id": session.case.order_id,
    }
    session.verdict = verdict
    return _dump({"accepted": True, "verdict": verdict})


TOOLS = [
    get_window,
    move_window,
    inspect,
    submit_verdict,
]
