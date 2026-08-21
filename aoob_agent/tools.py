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


def _dedent_rows(rows: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Shift snippet lines left by their shared leading whitespace.

    Indentation is cosmetic in C, so cutting the shared prefix saves tokens
    on deeply-nested legacy code without changing what the LLM can read.
    Structural anchor lines (a bare brace, or a preprocessor line marker
    like ``# 144 "file.c"``) sit at column 0 by convention and would force
    the shared amount to 0 for every function if counted, so they're
    ignored when computing how much to strip — but every line still only
    ever loses whitespace it actually has, never other characters.
    """
    def _indent(text: str) -> Optional[int]:
        stripped = text.lstrip(" \t")
        if not stripped or stripped.startswith("#") or stripped in ("{", "}"):
            return None
        return len(text) - len(stripped)

    indents = [i for i in (_indent(t) for _, t in rows) if i is not None]
    shared = min(indents) if indents else 0
    if shared <= 0:
        return rows
    out: list[tuple[int, str]] = []
    for ln, text in rows:
        lead = len(text) - len(text.lstrip(" \t"))
        cut = min(shared, lead)
        out.append((ln, text[cut:]))
    return out


_MAX_FUNCTION_LINES = 2000  # safety valve for pathological generated functions


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
    span = store.function_span(name, near_line=anchor_line)
    if span:
        return span
    return None, None


def _snippet_for_function(
    function_name: str,
    *,
    anchor_line: Optional[int],
) -> dict[str, Any]:
    """Return the complete source span for one function.

    No windowing / context_lines crop. A 2000-line ceiling is only a safety
    valve for pathological generated functions, not a design cap.
    """
    store = _store()
    start, end = _find_function_span(function_name, anchor_line=anchor_line)
    if start is None or end is None:
        return {"error": f"Could not resolve function span for {function_name!r}."}
    capped = False
    snippet_end = end
    if end - start + 1 > _MAX_FUNCTION_LINES:
        snippet_end = start + _MAX_FUNCTION_LINES - 1
        capped = True
    raw_rows = store.get_source_slice(start, snippet_end)
    lines = [
        {"line": ln, "text": txt}
        for ln, txt in _dedent_rows(raw_rows)
    ]
    payload: dict[str, Any] = {
        "function_name": function_name,
        "span": {
            "start_line": start,
            "end_line": end,
            "snippet_start": start,
            "snippet_end": snippet_end,
        },
        "line_count": len(lines),
        "snippet": lines,
    }
    if capped:
        payload["note"] = (
            f"Function spans {end - start + 1} lines; showing the first "
            f"{_MAX_FUNCTION_LINES} as a safety valve."
        )
    return payload


def _window_payload() -> dict[str, Any]:
    session = get_session()
    step = session.current_step()
    if step is None:
        return {"error": "Case path is empty."}
    session.mark_opened()
    body = _snippet_for_function(step.function, anchor_line=step.line)
    if body.get("error"):
        return body
    return {
        "path_step": step.step,
        "path_length": session.n_steps,
        "role": step.role,
        "symbol": step.symbol,
        "scope": step.scope,
        "access": step.access,
        "declared_size": step.declared_size if step.declared_size is not None else session.case.array_size,
        "datatype": step.datatype,
        "first_access": step.first_access,
        "location": step.location,
        "argument_expression": step.argument_expression,
        "call_site": step.call_site,
        "note": step.note,
        "guards": list(step.guards or session.case.guards),
        "case_gaps": list(session.case.gaps),
        **body,
    }


@tool
def get_window() -> str:
    """Return the complete current function (origin-path step), no line cap.

    Starts at the index origin (first path step). Does not take a function
    name — the compiled case file owns the path.
    """
    return _dump(_window_payload())


@tool
def move_window(direction: str = "next", step: int = 0) -> str:
    """Move to another step on the compiled origin → alarm path.

    direction: next (toward alarm) or prev (toward origin).
    step: 1-based path index; when set, overrides direction.
    State transition only — does not return source. Call get_window after.
    """
    session = get_session()
    try:
        moved = session.move(direction=direction, step=step)
    except ValueError as exc:
        return _dump({"error": str(exc)})
    return _dump(
        {
            "moved_to_step": moved.step,
            "function": moved.function,
            "role": moved.role,
        }
    )


@tool
def inspect(function_name: str) -> str:
    """Open a helper named in the index expression (for example Mo_Inst_GetCoreIdx).

    Only helpers listed on the case file are allowed — this is not a free
    walk of the call graph. Returns the complete helper function body.
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
    """Submit the investigation verdict. This ends the investigation.

    classification: true | false | review
    If the indexed object's declared size is unknown, classification must be review.
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

    obj = session.case.indexed_object
    size_unknown = obj.array_size is None
    if cls in {"true", "false"} and size_unknown:
        session.size_unknown_retries += 1
        return _dump(
            {
                "accepted": False,
                "error": (
                    "Indexed object declared_size is unknown, so the access "
                    "cannot be bound-checked. Submit classification=review."
                ),
                "declared_size": None,
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








