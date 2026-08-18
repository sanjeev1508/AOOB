"""Retrieval-only tools for AOOB investigation."""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from langchain_core.tools import tool

from aoob_agent.data_store import DataStore
from aoob_agent.sequence_utils import build_trimmed_writes_alarm_sequence
from cf_viz.graph_builder import resolve_variable_from_alarm

# NOTE:
# The previous tool set had dedicated guard/constant/index-expression tools.
# This 5-tool set intentionally removes those APIs per spec. Guard/sentinel/
# transform evidence must now be read directly from get_function_snippet output.

_STORE: Optional[DataStore] = None
_TRACE_CONTEXT: dict[str, Any] = {
    "order_id": None,
    "current_function": None,
    "anchor_line": None,
    "variable": None,
    "scope": None,
}


def bind_store(store: DataStore) -> None:
    global _STORE
    _STORE = store


def _store() -> DataStore:
    if _STORE is None:
        raise RuntimeError("DataStore is not bound. Call bind_store() first.")
    return _STORE


def _dump(payload: dict) -> str:
    return json.dumps(payload, indent=2)


def _parse_line(location: str) -> Optional[int]:
    return _store().parse_location_line(location)


def _alarm_meta(order_id: int) -> tuple[Optional[Any], Optional[dict]]:
    alarm = _store().get_alarm(order_id)
    if alarm is None:
        return None, None
    return alarm, alarm.parsed_location


def _sanitize_symbol(name: str) -> str:
    return (name or "").strip().split("@", 1)[0].strip('"')


def _base_scope_symbol(name: str) -> str:
    """Return base variable used for scope lookup while preserving full path elsewhere."""
    raw = _sanitize_symbol(name)
    if not raw:
        return ""
    text = raw.strip()
    while text.startswith("*"):
        text = text[1:].strip()
    if "->" in text:
        return text.split("->", 1)[0].strip()
    if "." in text:
        return text.split(".", 1)[0].strip()
    return text


def _is_pointer_style_symbol(name: str) -> bool:
    raw = (name or "").strip()
    return raw.startswith("*") or ("->" in raw)


def _symbol_matches(observed: str, expected: str) -> bool:
    obs = _sanitize_symbol(observed)
    exp = _sanitize_symbol(expected)
    if not obs or not exp:
        return False
    if obs == exp:
        return True
    obs_root = obs.split(".", 1)[0]
    exp_root = exp.split(".", 1)[0]
    return bool(obs_root and exp_root and obs_root == exp_root)


def _normalize_member_access(expr: str) -> str:
    text = (expr or "").strip()
    text = re.sub(r"\s*->\s*", ".", text)
    text = re.sub(r"\s*\.\s*", ".", text)
    return text.strip()


_OPERAND_MEMBER_RE = re.compile(
    r"\b([A-Za-z_]\w*(?:\s*(?:\.|->)\s*[A-Za-z_]\w*)+)\b"
)
_OPERAND_IDENT_RE = re.compile(r"\b([A-Za-z_]\w*)\b")


def _operand_symbol_candidates(expr: str) -> list[str]:
    """Best-effort symbol candidates from an index operand expression."""
    out: list[str] = []
    seen: set[str] = set()

    for m in _OPERAND_MEMBER_RE.finditer(expr or ""):
        name = _normalize_member_access(m.group(1))
        if name and name not in seen:
            seen.add(name)
            out.append(name)

    for m in _OPERAND_IDENT_RE.finditer(expr or ""):
        token = (m.group(1) or "").strip()
        if not token:
            continue
        if token in {"if", "for", "while", "return", "sizeof"}:
            continue
        if token not in seen:
            seen.add(token)
            out.append(token)
    return out


def _set_context(
    *,
    order_id: Optional[int] = None,
    current_function: Optional[str] = None,
    anchor_line: Optional[int] = None,
    variable: Optional[str] = None,
    scope: Optional[str] = None,
) -> None:
    if order_id is not None:
        _TRACE_CONTEXT["order_id"] = order_id
    if current_function is not None:
        _TRACE_CONTEXT["current_function"] = current_function
    if anchor_line is not None:
        _TRACE_CONTEXT["anchor_line"] = anchor_line
    if variable is not None:
        _TRACE_CONTEXT["variable"] = variable
    if scope is not None:
        _TRACE_CONTEXT["scope"] = scope


def _current_order(order_id: int = 0) -> Optional[int]:
    if order_id:
        return int(order_id)
    return _TRACE_CONTEXT.get("order_id")


def _resolve_function_from_alarm(order_id: int) -> tuple[Optional[str], Optional[int], Optional[int], Optional[int]]:
    alarm, parsed = _alarm_meta(order_id)
    if alarm is None or not parsed:
        return None, None, None, None
    line = parsed["line"]
    fn, start, end = _store().find_enclosing_function(line)
    return fn, start, end, line


def _infer_scope_for_symbol(
    store: DataStore,
    *,
    symbol: str,
    order_id: Optional[int],
    trace_function: Optional[str],
    trace_line: Optional[int],
) -> str:
    sym = _base_scope_symbol(symbol)
    pointer_style = _is_pointer_style_symbol(symbol)
    if not sym:
        return "unknown"

    fn = (trace_function or "").strip()
    near = trace_line
    if not fn and order_id:
        fn, _s, _e, near = _resolve_function_from_alarm(order_id)

    if fn:
        params = store.parse_function_parameters(fn, near_line=near)
        if any((p.get("name") or "") == sym for p in params):
            return "dynamic_heap" if pointer_style else "parameter"

    info = store.resolve_declaration_lookup(sym)
    if not info.get("found"):
        return "unknown"

    decl = info.get("declaration")
    if pointer_style and bool(getattr(decl, "is_pointer", False)):
        return "dynamic_heap"
    loc = getattr(decl, "location", "") if decl else ""
    dline = _parse_line(loc)
    if dline is None:
        return "unknown"
    dfn, dstart, dend = store.find_enclosing_function(dline)
    if dfn and dstart is not None and dend is not None and dstart <= dline <= dend:
        return "local"
    return "global"


def _context_scope_for_symbol(symbol: str) -> Optional[str]:
    ctx_var = _sanitize_symbol(str(_TRACE_CONTEXT.get("variable") or ""))
    if not ctx_var:
        return None
    if _symbol_matches(ctx_var, symbol):
        scope = str(_TRACE_CONTEXT.get("scope") or "").strip().lower()
        return scope or None
    return None


def _find_function_span_by_name(function_name: str) -> tuple[Optional[int], Optional[int]]:
    name = (function_name or "").strip()
    if not name:
        return None, None
    src = _store().source_lines
    for ln in range(1, len(src)):
        if _store()._looks_like_function_header(ln) == name:
            return ln, _store()._function_end_from(ln)
    return None, None


def _kind_from_lookup(symbol: str, info: dict) -> str:
    if not info.get("found"):
        return "unknown"
    kind = info.get("kind")
    decl = info.get("declaration")
    if kind == "array":
        return "array"
    if kind == "object":
        return "struct"
    if "." in symbol:
        return "field"
    if decl is not None and getattr(decl, "is_pointer", False):
        return "pointer"
    if decl is not None:
        return "variable"
    return "other"


def _extract_index_operands(line_text: str) -> list[tuple[str, str]]:
    """Extract indexed[operand] pairs from one source line.

    Returns tuples of (indexed_object_text, operand_text). For nested accesses
    (for example arr[a][b]), each bracket pair is returned with its immediate
    left expression (arr, then arr[a]).
    """
    text = line_text or ""
    out: list[tuple[str, str]] = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] != "[":
            i += 1
            continue

        depth = 1
        j = i + 1
        while j < n and depth > 0:
            if text[j] == "[":
                depth += 1
            elif text[j] == "]":
                depth -= 1
            j += 1
        if depth != 0:
            break

        operand = text[i + 1 : j - 1].strip()

        k = i - 1
        while k >= 0 and text[k].isspace():
            k -= 1
        left_end = k
        allowed = set("_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.>[]()")
        while k >= 0 and (text[k] in allowed or text[k].isspace()):
            if text[k] in ";,{}":
                break
            k -= 1
        indexed = text[k + 1 : left_end + 1].strip()
        if indexed and operand:
            out.append((indexed, operand))

        i = j

    # Pointer arithmetic / dereference pattern (e.g. *(ptr + offset)).
    for m in re.finditer(r"\*\s*\(\s*([^\)]+?)\s*\)", text):
        operand = (m.group(1) or "").strip()
        if not operand:
            continue
        out.append(("*()", operand))

    # De-duplicate while preserving order.
    deduped: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in out:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


@tool
def get_variable_scope(symbol: str, alarm_order_id: int = 0) -> str:
    """Return whether a symbol is local/global/parameter/dynamic_heap/unknown."""
    store = _store()
    full_sym = _sanitize_symbol(symbol)
    sym = _base_scope_symbol(symbol)
    pointer_style = _is_pointer_style_symbol(symbol)
    oid = _current_order(alarm_order_id)
    if not sym:
        return _dump(
            {
                "symbol": full_sym,
                "base_symbol": sym,
                "scope": "unknown",
                "declared_in_function": None,
                "location": "",
                "parse_note": "Empty symbol.",
            }
        )

    declared_in_fn = None
    parse_note = None

    if oid:
        fn, _start, _end, line = _resolve_function_from_alarm(oid)
        if fn:
            params = store.parse_function_parameters(fn, near_line=line)
            if any((p.get("name") or "") == sym for p in params):
                scope = "dynamic_heap" if pointer_style else "parameter"
                _set_context(
                    order_id=oid,
                    current_function=fn,
                    anchor_line=line,
                    variable=sym,
                    scope=scope,
                )
                return _dump(
                    {
                        "symbol": full_sym,
                        "base_symbol": sym,
                        "scope": scope,
                        "declared_in_function": fn,
                        "location": "",
                        "parse_note": (
                            "Pointer/member parameter access may require allocation context."
                            if scope == "dynamic_heap"
                            else None
                        ),
                    }
                )

    info = store.resolve_declaration_lookup(sym)
    if not info.get("found"):
        if oid:
            fn, _s, _e, line = _resolve_function_from_alarm(oid)
            if fn and line is not None:
                near: list[Any] = []
                for ln in range(max(1, line - 5), line + 6):
                    near.extend(store.data_flow_by_line.get(ln, []))
                if any((r.function == fn) and _symbol_matches(r.variable, sym) for r in near):
                    _set_context(
                        order_id=oid,
                        current_function=fn,
                        anchor_line=line,
                        variable=sym,
                        scope="local",
                    )
                    return _dump(
                        {
                            "symbol": full_sym,
                            "base_symbol": sym,
                            "scope": "local",
                            "declared_in_function": fn,
                            "location": "",
                            "parse_note": (
                                "Declaration index miss; scope inferred as local from "
                                "same-function data-flow usage near alarm site."
                            ),
                        }
                    )
        suggestions = store.declaration_name_suggestions(sym)
        parse_note = (
            "No declaration indexed for this exact symbol name. "
            "Possible macro-generated name, aliasing, or shadowing."
        )
        return _dump(
            {
                "symbol": full_sym,
                "base_symbol": sym,
                "scope": "unknown",
                "declared_in_function": None,
                "location": "",
                "parse_note": parse_note,
                "name_suggestions": suggestions[:8],
            }
        )

    decl = info.get("declaration")
    loc = getattr(decl, "location", "") or ""
    dline = _parse_line(loc)
    if dline is None:
        if oid:
            fn, _s, _e, line = _resolve_function_from_alarm(oid)
            if fn:
                _set_context(
                    order_id=oid,
                    current_function=fn,
                    anchor_line=line,
                    variable=sym,
                    scope="local",
                )
                return _dump(
                    {
                        "symbol": full_sym,
                        "base_symbol": sym,
                        "scope": "local",
                        "declared_in_function": fn,
                        "location": loc,
                        "parse_note": (
                            "Declaration location missing; conservatively treated as local for "
                            "in-function tracing."
                        ),
                    }
                )
        return _dump(
            {
                "symbol": full_sym,
                "base_symbol": sym,
                "scope": "unknown",
                "declared_in_function": None,
                "location": loc,
                "parse_note": "Declaration location could not be parsed.",
            }
        )

    fn, fstart, fend = store.find_enclosing_function(dline)
    if fn and fstart is not None and fend is not None and fstart <= dline <= fend:
        declared_in_fn = fn
        scope = "local"
    else:
        scope = "global"

    if pointer_style and bool(getattr(decl, "is_pointer", False)):
        scope = "dynamic_heap"
        parse_note = (
            "Pointer/member style access traced via base symbol; dynamic memory "
            "bounds may require allocation context."
        )

    if oid:
        alarm_fn, _, _, alarm_line = _resolve_function_from_alarm(oid)
        if alarm_fn and scope == "local" and declared_in_fn and declared_in_fn != alarm_fn:
            parse_note = (
                "Symbol declaration appears local to a different function; "
                "possible shadowing/aliasing."
            )
        _set_context(
            order_id=oid,
            current_function=alarm_fn or declared_in_fn,
            anchor_line=alarm_line,
            variable=sym,
            scope=scope,
        )

    return _dump(
        {
            "symbol": full_sym,
            "base_symbol": sym,
            "scope": scope,
            "declared_in_function": declared_in_fn,
            "location": loc,
            "parse_note": parse_note,
        }
    )


@tool
def get_declaration_info(symbol: str) -> str:
    """Return declaration facts for a symbol."""
    store = _store()
    sym = _sanitize_symbol(symbol)
    info = store.resolve_declaration_lookup(sym)
    if not info.get("found"):
        return _dump(
            {
                "symbol": sym,
                "found": False,
                "kind": "unknown",
                "declared_type": "",
                "array_size": None,
                "is_pointer": False,
                "location": "",
                "parse_note": "No declaration found for this symbol.",
                "name_suggestions": store.declaration_name_suggestions(sym)[:8],
            }
        )

    decl = info.get("declaration")
    nested = list(info.get("nested_array_fields") or [])
    if info.get("kind") == "object" and not nested:
        # Fallback: surface member-level candidates if typedef parsing missed them.
        root = sym
        seen: set[str] = set()
        for name, drec in store.declarations.items():
            if not name.startswith(root + "."):
                continue
            member = name.split(".", 1)[1]
            if not member or member in seen:
                continue
            seen.add(member)
            nested.append(drec)
        for df_name in store.data_flow_by_variable:
            if not df_name.startswith(root + "."):
                continue
            member = df_name.split(".", 1)[1].split("@", 1)[0].strip('"')
            if not member or member in seen:
                continue
            seen.add(member)
            nested.append(
                type(decl)(
                    name=f"{root}.{member}",
                    raw_declaration_text="",
                    array_size=None,
                    declared_type="unknown",
                    is_pointer=False,
                    location="",
                    parse_note=(
                        "Member inferred from data-flow keys; declaration line not resolved."
                    ),
                )
            )
    return _dump(
        {
            "symbol": sym,
            "found": True,
            "kind": _kind_from_lookup(sym, info),
            "declared_type": getattr(decl, "declared_type", "") if decl else "",
            "array_size": getattr(decl, "array_size", None) if decl else None,
            "is_pointer": bool(getattr(decl, "is_pointer", False)) if decl else False,
            "location": getattr(decl, "location", "") if decl else "",
            "parse_note": getattr(decl, "parse_note", None) if decl else None,
            "nested_array_fields": [
                {
                    "name": f.name,
                    "declared_type": f.declared_type,
                    "array_size": f.array_size,
                    "location": f.location,
                    "parse_note": f.parse_note,
                }
                for f in nested
            ],
        }
    )


@tool
def get_trimmed_sequence(alarm_order_id: int = 0, variable: str = "") -> str:
    """Return cross-function DF sequence trimmed to writes/mixed + alarm step."""
    store = _store()
    oid = _current_order(alarm_order_id)
    if not oid:
        return _dump({"error": "No active alarm context. Provide alarm_order_id."})

    alarm, parsed = _alarm_meta(oid)
    if alarm is None or not parsed:
        return _dump({"error": f"Unknown alarm order id: {oid}"})

    primary, _site_vars = resolve_variable_from_alarm(store, oid)
    chosen = _sanitize_symbol(variable) or (primary or "")

    trace_fn = (_TRACE_CONTEXT.get("current_function") or "").strip()
    trace_line = _TRACE_CONTEXT.get("anchor_line")
    scope = _context_scope_for_symbol(chosen) or _infer_scope_for_symbol(
        store,
        symbol=chosen,
        order_id=oid,
        trace_function=trace_fn,
        trace_line=trace_line,
    )
    function_scope = trace_fn if scope in {"local", "parameter"} and trace_fn else None

    payload = build_trimmed_writes_alarm_sequence(
        store,
        oid,
        variable=chosen,
        function_scope=function_scope,
        max_events=500,
    )
    payload["scope"] = scope

    fn, _s, _e, line = _resolve_function_from_alarm(oid)
    _set_context(
        order_id=oid,
        current_function=trace_fn or fn,
        anchor_line=trace_line or line,
        variable=payload.get("variable") or chosen,
        scope=scope,
    )
    return _dump(payload)


@tool
def get_function_snippet(alarm_order_id: int = 0, context_lines: int = 120) -> str:
    """Return snippet for current function context with index operand hints."""
    store = _store()
    oid = _current_order(alarm_order_id)
    if not oid:
        return _dump({"error": "No active alarm context. Call get_trimmed_sequence first."})

    alarm, parsed = _alarm_meta(oid)
    if alarm is None or not parsed:
        return _dump({"error": f"Unknown alarm order id: {oid}"})

    fn = (_TRACE_CONTEXT.get("current_function") or "").strip()
    anchor = _TRACE_CONTEXT.get("anchor_line")
    if not fn:
        fn, _s, _e, anchor = _resolve_function_from_alarm(oid)
    if not fn:
        return _dump({"error": "Could not resolve current function."})

    start, end = _find_function_span_by_name(fn)
    if start is None or end is None:
        return _dump({"error": f"Could not resolve function span for {fn!r}."})

    max_lines = 220
    if end - start + 1 <= max_lines:
        snippet_start, snippet_end = start, end
    else:
        ctx = max(20, min(int(context_lines), 180))
        anchor_line = int(anchor or parsed["line"])
        snippet_start = max(start, anchor_line - ctx)
        snippet_end = min(end, snippet_start + max_lines - 1)

    lines = [
        {"line": ln, "text": txt}
        for ln, txt in store.get_source_slice(snippet_start, snippet_end)
    ]
    callers = sorted({r.caller for r in store.control_by_callee.get(fn, []) if r.caller})[:40]
    callees = sorted({r.callee for r in store.control_by_caller.get(fn, []) if r.callee})[:40]

    alarm_line_text = ""
    alarm_line = parsed["line"]
    if 0 < alarm_line < len(store.source_lines):
        alarm_line_text = store.source_lines[alarm_line]

    index_operands_at_alarm = []
    for indexed, operand in _extract_index_operands(alarm_line_text):
        candidates = _operand_symbol_candidates(operand)
        index_operands_at_alarm.append(
            {
                "indexed_object": indexed,
                "operand_text": operand,
                "operand_symbol_candidates": candidates,
                "resolved_operand_symbol": candidates[0] if candidates else "",
                "location": alarm.location,
            }
        )

    _set_context(order_id=oid, current_function=fn, anchor_line=int(anchor or parsed["line"]))

    return _dump(
        {
            "alarm_order_id": oid,
            "function_name": fn,
            "span": {
                "start_line": start,
                "end_line": end,
                "snippet_start": snippet_start,
                "snippet_end": snippet_end,
            },
            "snippet": lines,
            "callers": callers,
            "callees": callees,
            "index_operands_at_alarm": index_operands_at_alarm,
            "note": (
                "Index operands are a lightweight source-text parse of the alarm line. "
                "Nested/computed subscripts are returned as raw operand text and are "
                "not decomposed semantically."
            ),
        }
    )


@tool
def get_caller_context(parameter: str, alarm_order_id: int = 0, current_function: str = "") -> str:
    """Return one-hop caller context for parameter-origin tracing."""
    store = _store()
    oid = _current_order(alarm_order_id)
    if not oid:
        return _dump({"error": "No active alarm context. Provide alarm_order_id."})

    alarm, parsed = _alarm_meta(oid)
    if alarm is None or not parsed:
        return _dump({"error": f"Unknown alarm order id: {oid}"})

    param = _sanitize_symbol(parameter)
    if not param:
        return _dump(
            {
                "alarm_order_id": oid,
                "current_function": "",
                "callers": [],
                "unresolved": True,
                "note": "parameter is required for caller-context tracing.",
            }
        )

    fn = (current_function or "").strip() or (_TRACE_CONTEXT.get("current_function") or "").strip()
    if not fn:
        fn, _s, _e, _l = _resolve_function_from_alarm(oid)
    if not fn:
        return _dump({"error": "Could not resolve current function."})

    params = store.parse_function_parameters(fn, near_line=parsed["line"])
    param_index = None
    for p in params:
        if (p.get("name") or "") == param:
            param_index = int(p.get("index", 0))
            break

    if param_index is None:
        _set_context(
            order_id=oid,
            current_function=fn,
            anchor_line=parsed["line"],
            variable=param,
            scope="parameter",
        )
        return _dump(
            {
                "alarm_order_id": oid,
                "current_function": fn,
                "callers": [],
                "unresolved": True,
                "note": (
                    "Parameter not found in current function signature. "
                    "Caller argument extraction is not applicable here."
                ),
            }
        )

    callers_rows = list(store.control_by_callee.get(fn, []))
    if not callers_rows:
        _set_context(
            order_id=oid,
            current_function=fn,
            anchor_line=parsed["line"],
            variable=param,
            scope="parameter",
        )
        return _dump(
            {
                "alarm_order_id": oid,
                "current_function": fn,
                "callers": [],
                "unresolved": True,
                "note": "No caller found in control-flow graph (entry point or unresolved call edge).",
            }
        )

    seen: set[tuple[str, int]] = set()
    out_callers: list[dict[str, Any]] = []
    for edge in callers_rows:
        if edge.line is None:
            continue
        key = (edge.caller, edge.line)
        if key in seen:
            continue
        seen.add(key)
        entry = {
            "function": edge.caller,
            "call_site_location": edge.call_site or f"ALL_bc_with_context.c:{edge.line}.1-1",
            "line": edge.line,
            "argument_expression_text": None,
        }
        extracted = store.extract_call_argument_expression(fn, edge.line, param_index)
        if extracted:
            entry["argument_expression_text"] = extracted.get("argument_expression")
            entry["argument_identifiers"] = extracted.get("argument_identifiers") or []
        out_callers.append(entry)

    out_callers.sort(key=lambda x: (x.get("line") or 10**12, x.get("function") or ""))
    next_fn = out_callers[0]["function"] if out_callers else fn
    next_line = out_callers[0].get("line") if out_callers else parsed["line"]
    _set_context(
        order_id=oid,
        current_function=next_fn,
        anchor_line=next_line,
        variable=param,
        scope="parameter",
    )

    return _dump(
        {
            "alarm_order_id": oid,
            "current_function": fn,
            "callers": out_callers[:20],
            "unresolved": False if out_callers else True,
            "note": (
                "Parameter-only caller retrieval. Current function context is advanced "
                f"to {next_fn!r} for the next snippet/sequence step."
            ),
        }
    )


TOOLS = [
    get_variable_scope,
    get_declaration_info,
    get_trimmed_sequence,
    get_function_snippet,
    get_caller_context,
]
