"""Window-walk tools over a compiled AOOB case file."""

from __future__ import annotations

import json
import re
from typing import Any, Literal, Optional

from langchain_core.tools import tool

from aoob_agent.data_store import DataStore
from aoob_agent.session import get_session

_STORE: Optional[DataStore] = None

DEFAULT_MICRO_WINDOW = 15


def bind_store(store: DataStore) -> None:
    global _STORE
    _STORE = store


def _store() -> DataStore:
    if _STORE is None:
        raise RuntimeError("DataStore is not bound. Call bind_store() first.")
    return _STORE


def _dump(payload: dict) -> str:
    return json.dumps(payload, indent=2)


def _record_confidence(confidence_so_far: str = "") -> None:
    conf = (confidence_so_far or "").strip().lower()
    if conf in {"low", "medium", "high"}:
        get_session().confidence_so_far = conf


def _dedent_rows(rows: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Shift snippet lines left by their shared leading whitespace."""
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


_MAX_FUNCTION_LINES = 2000


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
    """Return the complete source span for one function."""
    store = _store()
    cached = store.cached_function_snippet(function_name, anchor_line=anchor_line)
    if cached is not None:
        return dict(cached)
    start, end = _find_function_span(function_name, anchor_line=anchor_line)
    if start is None or end is None:
        return {"error": f"Could not resolve function span for {function_name!r}."}
    capped = False
    snippet_end = end
    if end - start + 1 > _MAX_FUNCTION_LINES:
        snippet_end = start + _MAX_FUNCTION_LINES - 1
        capped = True
    raw_rows = store.get_source_slice(start, snippet_end)
    lines = [{"line": ln, "text": txt} for ln, txt in _dedent_rows(raw_rows)]
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
        "from_cache": False,
    }
    if capped:
        payload["note"] = (
            f"Function spans {end - start + 1} lines; showing the first "
            f"{_MAX_FUNCTION_LINES} as a safety valve."
        )
    store.store_function_snippet(function_name, anchor_line=anchor_line, payload=payload)
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
    if body.get("from_cache") is not False:
        body = dict(body)
        body["from_cache"] = True
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


def _micro_window_payload(window_size: int = DEFAULT_MICRO_WINDOW) -> dict[str, Any]:
    session = get_session()
    step = session.current_step()
    if step is None:
        return {"error": "Case path is empty."}
    session.mark_micro_opened()
    store = _store()
    anchor_line = step.line or 1
    half = max(3, window_size // 2)
    start_line = max(1, anchor_line - half)
    end_line = anchor_line + half
    fn_start, fn_end = _find_function_span(step.function, anchor_line=anchor_line)
    window_truncated = False
    if fn_start is not None and fn_end is not None:
        if start_line > fn_start or end_line < fn_end:
            window_truncated = True
    cache_key_fn = step.function or ""
    cached = store.cached_micro_window(cache_key_fn, start_line, end_line)
    if cached is not None:
        payload = dict(cached)
        payload["from_cache"] = True
        payload["path_step"] = step.step
        payload["path_length"] = session.n_steps
        return payload
    raw_rows = store.get_source_slice(start_line, end_line)
    lines = [{"line": ln, "text": txt} for ln, txt in _dedent_rows(raw_rows)]
    payload: dict[str, Any] = {
        "path_step": step.step,
        "path_length": session.n_steps,
        "alarm_line": anchor_line,
        "function_name": step.function,
        "function_span": {"start_line": fn_start, "end_line": fn_end},
        "indexed_object": session.case.indexed_object.name,
        "declared_size": session.case.array_size,
        "operand_symbol": session.case.operand_symbol,
        "index_expression": session.case.index_expression,
        "micro_window_span": {"start_line": start_line, "end_line": end_line},
        "window_size_requested": window_size,
        "window_truncated": window_truncated,
        "guards_found": list(step.guards or session.case.guards),
        "snippet": lines,
        "from_cache": False,
    }
    store.store_micro_window(cache_key_fn, start_line, end_line, payload)
    return payload


def _dtype_bit_width(datatype: str, var_name: str) -> tuple[Optional[int], Optional[int], Optional[int]]:
    """Mechanical bit-width from declared type name (not a classification decision)."""
    dtype_lower = (datatype or "").lower()
    var_lower = (var_name or "").lower()
    combined = f"{dtype_lower} {var_lower}"
    if "uint8" in combined or re.search(r"\bu8\b", combined):
        return 8, 0, 255
    if "uint16" in combined or re.search(r"\bu16\b", combined):
        return 16, 0, 65535
    if "uint32" in combined or re.search(r"\bu32\b", combined):
        return 32, 0, 4294967295
    if "int8" in combined:
        return 8, -128, 127
    if "int16" in combined:
        return 16, -32768, 32767
    if "int32" in combined:
        return 32, -2147483648, 2147483647
    return None, None, None


def _detect_mask_literal(index_expr: str) -> tuple[Optional[str], Optional[int]]:
    """Extract mask literal and its arithmetic max from index expression text."""
    expr = index_expr or ""
    expr_lower = expr.lower()
    patterns = [
        (r"&\s*0x([0-9a-f]+)", 16),
        (r"&\s*(\d+)\s*u?", 10),
    ]
    for pattern, base in patterns:
        match = re.search(pattern, expr_lower)
        if not match:
            continue
        try:
            mask_val = int(match.group(1), base)
        except ValueError:
            continue
        if mask_val >= 0:
            return match.group(0).strip(), mask_val
    return None, None


def _derive_index_bounds(
    *,
    var_name: str,
    index_expr: str,
    datatype: str,
    guards: list[str],
) -> dict[str, Any]:
    """Return mechanical bounds only — no classification."""
    bit_width, derived_min, derived_max = _dtype_bit_width(datatype, var_name)
    mask_literal, mask_max = _detect_mask_literal(index_expr)
    bound_derivation = "none"
    derivations: list[str] = []

    if bit_width is not None and derived_max is not None:
        bound_derivation = "dtype"
        derivations.append("dtype")

    if mask_literal is not None and mask_max is not None:
        if derived_max is None:
            derived_min = 0
            derived_max = mask_max
            bound_derivation = "mask"
        else:
            derived_max = min(derived_max, mask_max)
            bound_derivation = "dtype+mask" if "dtype" in derivations else "mask"
        derivations.append("mask")

    guard_texts = [g.strip() for g in guards if (g or "").strip()]
    numeric_in_guards: list[int] = []
    for g in guard_texts:
        for tok in re.split(r"[\s()<>=!,;]+", g):
            clean = tok.rstrip("uUlL")
            if clean.isdigit():
                val = int(clean)
                if 0 <= val <= 65535:
                    numeric_in_guards.append(val)

    return {
        "declared_type": datatype or "",
        "bit_width": bit_width,
        "mask_literal": mask_literal,
        "guard_conditions": guard_texts,
        "guard_numeric_literals": sorted(set(numeric_in_guards)),
        "derived_min": derived_min,
        "derived_max": derived_max,
        "bound_derivation": bound_derivation if derived_max is not None else "none",
        "derivation_notes": (
            "Mechanical bounds from type width and/or bitmask in index expression. "
            "Guards are returned as raw text — you must judge whether they constrain "
            "the index at the alarm site."
        ),
    }


@tool
def get_micro_window(
    window_size: int = DEFAULT_MICRO_WINDOW,
    confidence_so_far: str = "low",
) -> str:
    """Return a token-trimmed slice around the alarm line.

    window_size: lines above/below anchor (default 15). Increase or call get_window()
    if window_truncated is true and guard/context is clipped.
    confidence_so_far: your current investigation confidence (low/medium/high).
    """
    _record_confidence(confidence_so_far)
    return _dump(_micro_window_payload(window_size))


@tool
def query_call_graph(function_name: str = "", confidence_so_far: str = "low") -> str:
    """Query inter-procedural call graph callers, callees, and function nodes."""
    _record_confidence(confidence_so_far)
    session = get_session()
    name = (function_name or "").strip() or session.case.alarm_function
    session.call_graph_queries.add(name)
    if session.case.gaps:
        session.gaps_addressed = bool(
            session.call_graph_queries or session.inspected or session.data_flow_queries
        )
    store = _store()
    callers = sorted({e.caller for e in store.control_by_callee.get(name, []) if e.caller})
    callees = sorted({e.callee for e in store.control_by_caller.get(name, []) if e.callee})
    return _dump({
        "function_name": name,
        "callers": callers[:24],
        "callees": callees[:24],
        "helpers": list(session.case.helpers),
        "path_functions": [s.function for s in session.case.path],
        "indexed_object": session.case.indexed_object.name,
        "case_gaps": list(session.case.gaps),
    })


@tool
def query_data_flow(variable_name: str = "", confidence_so_far: str = "low") -> str:
    """Query data-flow read and write sites for an index variable or operand."""
    _record_confidence(confidence_so_far)
    session = get_session()
    var = (variable_name or "").strip() or session.case.operand_symbol
    session.data_flow_queries.add(var)
    if session.case.gaps:
        session.gaps_addressed = bool(
            session.call_graph_queries or session.inspected or session.data_flow_queries
        )
    store = _store()
    root = var.split(".", 1)[0]
    rows = store.data_flow_by_variable.get(var) or store.data_flow_by_variable.get(root) or []
    sites = [
        {
            "function": r.function,
            "access": r.access,
            "location": r.location,
            "line": r.line,
        }
        for r in rows[:20]
    ]
    return _dump({
        "variable": var,
        "operand_symbol": session.case.operand_symbol,
        "indexed_object": session.case.indexed_object.name,
        "sites": sites,
        "index_origin_summary": " | ".join([s.note for s in session.case.path if s.note][:4]),
        "case_gaps": list(session.case.gaps),
    })


@tool
def get_window(confidence_so_far: str = "low") -> str:
    """Return the complete current function (origin-path step), no line cap."""
    _record_confidence(confidence_so_far)
    return _dump(_window_payload())


@tool
def move_window(
    direction: str = "next",
    step: int = 0,
    confidence_so_far: str = "low",
) -> str:
    """Move to another step on the compiled origin → alarm path."""
    _record_confidence(confidence_so_far)
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
def inspect(function_name: str, confidence_so_far: str = "low") -> str:
    """Open a helper named in the index expression."""
    _record_confidence(confidence_so_far)
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
    if session.case.gaps:
        session.gaps_addressed = bool(
            session.call_graph_queries or session.inspected or session.data_flow_queries
        )
    body = _snippet_for_function(name, anchor_line=None)
    if body.get("error"):
        return _dump(body)
    return _dump({"helper": name, "inspected": sorted(session.inspected), **body})


@tool
def inspect_declaration(symbol_name: str, confidence_so_far: str = "low") -> str:
    """Open the declaration/initializer for the indexed object or operand."""
    _record_confidence(confidence_so_far)
    session = get_session()
    name = (symbol_name or "").strip()
    if not session.can_inspect_declaration(name):
        return _dump(
            {
                "error": (
                    f"{name!r} is not a declaration this case can resolve. "
                    f"Try the indexed object ({session.case.indexed_object.name!r}) "
                    f"or the operand ({session.case.operand_symbol!r})."
                ),
            }
        )
    store = _store()
    decl = store.get_declaration(name)
    if decl is None:
        return _dump({"error": f"No declaration found for {name!r}."})
    return _dump(
        {
            "symbol": name,
            "declared_type": decl.declared_type,
            "array_size": decl.array_size,
            "is_pointer": decl.is_pointer,
            "location": decl.location,
            "raw_declaration_text": decl.raw_declaration_text,
        }
    )


@tool
def run_simulation(symbol_name: str = "", confidence_so_far: str = "low") -> str:
    """Return raw mechanical facts about index bounds vs array capacity.

    Does NOT recommend a classification — you must reason about risk yourself
    and cite declared capacity vs derived range in submit_verdict.
    """
    _record_confidence(confidence_so_far)
    session = get_session()
    case = session.case
    session.simulation_run = True

    var_name = (symbol_name or "").strip() or case.operand_symbol
    obj = case.indexed_object
    declared_size = obj.array_size
    index_expr = case.index_expression or var_name

    curr_step = session.current_step()
    datatype = (curr_step.datatype if curr_step else "") or obj.declared_type or ""
    guards = list(curr_step.guards if curr_step else case.guards)

    bounds = _derive_index_bounds(
        var_name=var_name,
        index_expr=index_expr,
        datatype=datatype,
        guards=guards,
    )

    sim_result = {
        "symbol_evaluated": var_name,
        "indexed_object": obj.name,
        "index_expression": index_expr,
        "declared_array_capacity": declared_size,
        "declared_type": bounds["declared_type"],
        "bit_width": bounds["bit_width"],
        "mask_literal": bounds["mask_literal"],
        "guard_conditions": bounds["guard_conditions"],
        "guard_numeric_literals": bounds["guard_numeric_literals"],
        "derived_min": bounds["derived_min"],
        "derived_max": bounds["derived_max"],
        "bound_derivation": bounds["bound_derivation"],
        "derivation_notes": bounds["derivation_notes"],
        "write_helpers": list(case.write_helpers),
        "case_gaps": list(case.gaps),
    }

    session.simulation = sim_result
    return _dump(sim_result)


@tool
def submit_verdict(
    classification: str,
    comment: str,
    confidence: str = "medium",
    summary: str = "",
    human_tag_pattern: str = "",
    confidence_so_far: str = "",
    reason_for_review: str = "",
    counterargument: str = "",
) -> str:
    """Submit the investigation verdict. This ends the investigation.

    classification: false | true (low) | true | undecided | review
    confidence_so_far: required when finishing — must be high unless classification=review
    reason_for_review: mandatory when classification=review
    counterargument: when confidence is medium, state the strongest case for the opposite verdict
    """
    session = get_session()
    conf_progress = (confidence_so_far or session.confidence_so_far or "").strip().lower()
    if conf_progress in {"low", "medium", "high"}:
        session.confidence_so_far = conf_progress

    cls = (classification or "").strip().lower()
    if cls in {"false", "f", "fp", "false_positive"}:
        cls = "false"
    elif cls in {"true (low)", "true_low", "low", "potential"}:
        cls = "true (low)"
    elif cls in {"true", "t", "bug", "true_positive"}:
        cls = "true"
    elif cls in {"undecided", "u", "inter_procedural"}:
        cls = "undecided"
    elif cls in {"review", "uncertain", "unknown"}:
        cls = "review"
    else:
        return _dump(
            {
                "accepted": False,
                "error": "classification must be false, true (low), true, undecided, or review.",
            }
        )

    if session.force_review:
        cls = "review"
        if not (reason_for_review or "").strip():
            reason_for_review = (
                session.force_review_reason
                or "Safety ceiling reached without high-confidence evidence."
            )

    if cls == "review" and not (reason_for_review or "").strip():
        return _dump(
            {
                "accepted": False,
                "error": "classification=review requires reason_for_review.",
            }
        )

    if not session.simulation_run and cls != "review":
        return _dump(
            {
                "accepted": False,
                "error": "Call run_simulation() at least once before a non-review verdict.",
            }
        )

    if session.case.gaps and not session.gaps_addressed and cls not in {"review", "undecided"}:
        return _dump(
            {
                "accepted": False,
                "error": (
                    "Case file has unresolved gaps — call query_call_graph() and/or "
                    "inspect() on relevant helpers before a decisive verdict."
                ),
                "case_gaps": list(session.case.gaps),
            }
        )

    blocked, block_reason = session.gaps_block_false()
    if cls == "false" and blocked:
        return _dump(
            {
                "accepted": False,
                "error": block_reason,
                "case_gaps": list(session.case.gaps),
                "simulation": session.simulation,
            }
        )

    conf = (confidence or "medium").strip().lower()
    if conf not in {"low", "medium", "high"}:
        conf = "medium"

    if (
        conf == "medium"
        and cls not in {"review", "undecided"}
        and not session.counterargument_done
        and not (counterargument or "").strip()
    ):
        session.counterargument_pending = True
        return _dump(
            {
                "accepted": False,
                "error": (
                    "confidence=medium requires one self-verification round: resubmit with "
                    "counterargument=<strongest case for the opposite classification>."
                ),
            }
        )
    if (counterargument or "").strip():
        session.counterargument_done = True

    if (
        cls not in {"review", "undecided"}
        and session.confidence_so_far != "high"
        and not session.force_review
    ):
        return _dump(
            {
                "accepted": False,
                "error": (
                    "Decisive verdict requires confidence_so_far=high. "
                    "Gather more evidence or submit classification=review."
                ),
                "confidence_so_far": session.confidence_so_far,
            }
        )

    tag = (human_tag_pattern or "").strip()
    session.human_tag_match = tag

    if cls in {"true", "false"} and session.case.indexed_object.array_size is None:
        return _dump(
            {
                "accepted": False,
                "error": (
                    "Indexed object declared_size is unknown, so the access "
                    "cannot be bound-checked. Submit classification=review or undecided."
                ),
                "declared_size": None,
                "case_gaps": list(session.case.gaps),
            }
        )

    from aoob_agent.token_metrics import compute_token_savings_percent

    savings = compute_token_savings_percent(
        session.tokens_used, session.naive_full_file_tokens
    )

    verdict = {
        "classification": cls,
        "comment": comment or "",
        "confidence": conf,
        "summary": summary or comment or "",
        "human_tag_pattern": tag,
        "reason_for_review": (reason_for_review or "").strip(),
        "counterargument": (counterargument or "").strip(),
        "confidence_so_far": session.confidence_so_far,
        "alarm_order_id": session.case.order_id,
        "token_savings_percent": savings,
        "tokens_used": session.tokens_used,
        "naive_full_file_tokens": session.naive_full_file_tokens,
        "micro_window_used": bool(session.micro_windows_opened),
        "simulation": session.simulation,
        "safety_ceiling_hit": session.force_review,
    }
    session.verdict = verdict
    return _dump({"accepted": True, "verdict": verdict})


TOOLS = [
    get_micro_window,
    run_simulation,
    query_call_graph,
    query_data_flow,
    get_window,
    move_window,
    inspect,
    inspect_declaration,
    submit_verdict,
]
