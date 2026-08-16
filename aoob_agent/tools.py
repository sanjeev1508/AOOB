"""Retrieval-only tools. No classification / heuristic analysis — the agent decides."""

from __future__ import annotations

import json
import re
from typing import Optional

from langchain_core.tools import tool

from aoob_agent.data_store import DataStore

_STORE: Optional[DataStore] = None

# Identifiers / GetXxx(arg) inside subscript expressions.
_IDENT_RE = re.compile(r"\b([A-Za-z_]\w*)\b")
_GET_ARG_RE = re.compile(
    r"\b(?:Mo_Inst_)?Get(?:Core|Tsk)Idx\s*\(\s*([A-Za-z_]\w*)\s*\)"
)
_INDEXED_OBJ_RE = re.compile(
    r"([A-Za-z_]\w*(?:\s*\.\s*[A-Za-z_]\w*)*)\s*\["
)


def bind_store(store: DataStore) -> None:
    global _STORE
    _STORE = store


def _store() -> DataStore:
    if _STORE is None:
        raise RuntimeError("DataStore is not bound. Call bind_store() first.")
    return _STORE


def extract_index_operands(source_texts: list[str]) -> dict:
    """Structurally list indexed objects vs index-operand candidates from C text.

    Retrieval aid only — does not classify safety. Nested brackets are handled
    with a simple scan; complex macros may yield partial candidates.
    """
    indexed_objects: list[str] = []
    candidates: list[str] = []
    seen_idx: set[str] = set()
    seen_obj: set[str] = set()

    def _add_cand(name: str) -> None:
        name = (name or "").strip()
        if not name or name in seen_idx:
            return
        # Skip common type/cast tokens and keywords.
        if name.lower() in {
            "if",
            "for",
            "while",
            "return",
            "sizeof",
            "void",
            "uint8",
            "uint16",
            "uint32",
            "int",
            "const",
            "static",
            "null",
        }:
            return
        # Index helper functions are not the index value itself.
        if re.search(r"Get\w*Idx$", name) or name.endswith("Idx"):
            if name.startswith("Get") or name.startswith("Mo_Inst_Get"):
                return
        seen_idx.add(name)
        candidates.append(name)

    def _add_obj(name: str) -> None:
        name = re.sub(r"\s+", "", name or "")
        if not name or name in seen_obj:
            return
        seen_obj.add(name)
        indexed_objects.append(name)
        # Also record bare field / tail for matching DF bare names.
        bare = name.split(".")[-1]
        if bare != name and bare not in seen_obj:
            seen_obj.add(bare)

    for text in source_texts:
        if not text:
            continue
        for m in _INDEXED_OBJ_RE.finditer(text):
            _add_obj(m.group(1))
        for m in _GET_ARG_RE.finditer(text):
            _add_cand(m.group(1))

        # Collect bracket contents (non-greedy, multi-pass for nesting).
        i = 0
        while i < len(text):
            if text[i] != "[":
                i += 1
                continue
            depth = 0
            j = i
            while j < len(text):
                if text[j] == "[":
                    depth += 1
                elif text[j] == "]":
                    depth -= 1
                    if depth == 0:
                        inner = text[i + 1 : j].strip()
                        if inner:
                            for gm in _GET_ARG_RE.finditer(inner):
                                _add_cand(gm.group(1))
                            # Field access: Struct.Field → Field (+ full path)
                            field = re.search(
                                r"([A-Za-z_]\w*)\s*\.\s*([A-Za-z_]\w*)\s*$",
                                inner,
                            )
                            if field:
                                _add_cand(field.group(2))
                                _add_cand(f"{field.group(1)}.{field.group(2)}")
                            # Bare identifier index
                            if re.fullmatch(r"[A-Za-z_]\w*", inner):
                                _add_cand(inner)
                            else:
                                # Other idents inside, excluding known indexed objs
                                for im in _IDENT_RE.finditer(inner):
                                    tok = im.group(1)
                                    if tok.startswith("Get") and tok.endswith("Idx"):
                                        continue
                                    if any(
                                        tok == o or o.endswith("." + tok) or o == tok
                                        for o in indexed_objects
                                    ):
                                        continue
                                    _add_cand(tok)
                        i = j + 1
                        break
                j += 1
            else:
                break

    # Candidates must not be the indexed object itself.
    obj_tails = {o.split(".")[-1] for o in indexed_objects}
    filtered = [
        c
        for c in candidates
        if c not in indexed_objects
        and c.split(".")[-1] not in obj_tails
        and c not in obj_tails
    ]
    # Prefer keeping both full path and bare field if present.
    return {
        "indexed_objects": indexed_objects,
        "index_operand_candidates": filtered,
        "note": (
            "Structural parse of source subscripts only. Prefer slicing "
            "index_operand_candidates (not indexed_objects / sentinel pointers)."
        ),
    }


def _alarm_payload(order_id: int) -> dict:
    """Alarm fields exposed to the agent.

    Includes Astrée's own ``message`` (diagnostic intervals etc.).
    Deliberately omits human Classification / Comment — those are ground-truth
    / meta and must not short-circuit investigation.
    """
    store = _store()
    alarm = store.get_alarm(order_id)
    if alarm is None:
        return {"error": f"Unknown alarm order id: {order_id}"}
    parsed = alarm.parsed_location
    return {
        "order": alarm.order,
        "type": alarm.type,
        "category": alarm.category,
        "location": alarm.location,
        "message": alarm.message,
        "parsed_location": parsed,
    }


def _df_to_dict(rec) -> dict:
    return {
        "variable": rec.variable,
        "bare_variable": rec.variable.split("@", 1)[0].strip('"'),
        "function": rec.function,
        "access": rec.access,
        "process": rec.process,
        "byte_offset": rec.byte_offset,
        "location": rec.location,
        "line": rec.line,
    }


def _cf_to_dict(rec) -> dict:
    return {
        "caller": rec.caller,
        "callee": rec.callee,
        "call_site": rec.call_site,
        "process": rec.process,
        "line": rec.line,
    }


_MAX_TOOL_CHARS = 4000

# Keep navigation-critical fields intact when shrinking large tool payloads.
_TRUNCATE_PROTECT = frozenset(
    {
        "alarm",
        "indexed_objects",
        "index_operand_candidates",
        "parameter_note",
        "navigation_hint",
        "call_site_arguments",
        "next_slice_candidates",
        "nested_array_fields",
        "parameter_type_width",
        "classification",
    }
)

# Prefer trimming noisy/long lists before evidence lists.
_TRUNCATE_PRIORITY = (
    "functions_visited",
    "unresolved_paths",
    "control_flow_at_site",
    "data_flow_records",
    "source_lines",
    "writes_found",
    "guards_found",
    "call_sites",
)


def _truncate_payload(payload: dict, max_chars: int = _MAX_TOOL_CHARS) -> dict:
    """Shrink large list fields until dumps fit, keeping valid JSON."""
    data = json.loads(json.dumps(payload))  # deep copy via JSON
    text = json.dumps(data, indent=2)
    if len(text) <= max_chars:
        return data

    list_keys = [
        k
        for k, v in data.items()
        if isinstance(v, list) and k not in _TRUNCATE_PROTECT
    ]

    def _rank(k: str) -> tuple[int, int]:
        try:
            pri = _TRUNCATE_PRIORITY.index(k)
        except ValueError:
            pri = len(_TRUNCATE_PRIORITY)
        return (pri, -len(json.dumps(data[k])))

    list_keys.sort(key=_rank)
    for key in list_keys:
        while isinstance(data.get(key), list) and len(data[key]) > 0:
            data[key] = data[key][: max(0, len(data[key]) // 2)]
            data["_truncated"] = True
            data["_truncated_field"] = key
            text = json.dumps(data, indent=2)
            if len(text) <= max_chars:
                return data
        if isinstance(data.get(key), list):
            data[key] = []
            data["_truncated"] = True
            data["_truncated_field"] = key
            text = json.dumps(data, indent=2)
            if len(text) <= max_chars:
                return data

    # Nested call_sites inside call_site_arguments (dict) — trim only that list.
    csa = data.get("call_site_arguments")
    if isinstance(csa, dict) and isinstance(csa.get("call_sites"), list):
        while len(csa["call_sites"]) > 1:
            csa["call_sites"] = csa["call_sites"][: max(1, len(csa["call_sites"]) // 2)]
            data["_truncated"] = True
            data["_truncated_field"] = "call_site_arguments.call_sites"
            text = json.dumps(data, indent=2)
            if len(text) <= max_chars:
                return data

    compact = json.dumps(data, separators=(",", ":"))
    if len(compact) <= max_chars:
        data["_truncated"] = True
        return data
    data = {
        "error": "tool payload exceeded size budget after truncation",
        "_truncated": True,
        "keys": sorted(payload.keys()),
    }
    return data


def _dump(payload: dict) -> str:
    data = _truncate_payload(payload)
    text = json.dumps(data, indent=2)
    if len(text) <= _MAX_TOOL_CHARS:
        return text
    # Compact form still valid JSON.
    return json.dumps(data, separators=(",", ":"))


@tool
def get_affected_symbols(alarm_order_id: int, line_window: int = 2) -> str:
    """Return variables / arrays / pointers / structs referenced at an alarm site.

    Looks up the alarm by order id, then returns matching data-flow records near the
    alarm location. Also returns the source line text(s) so the agent can classify
    symbol kinds (var/array/pointer/struct). The alarm ``message`` field is Astrée's
    own ABSTRACT interval diagnostic when present (e.g. ``[0, 7] not included in
    array index range [0, 1]``) — not a human true/false label and not a concrete
    runtime index value.

    Args:
        alarm_order_id: Astrée alarm Order id from Full_alarms.csv.
        line_window: Extra lines above/below the alarm line to include (default 2).
    """
    store = _store()
    meta = _alarm_payload(alarm_order_id)
    if "error" in meta:
        return _dump(meta)

    parsed = meta["parsed_location"]
    if not parsed:
        return _dump({**meta, "error": "Could not parse alarm location"})

    line = parsed["line"]
    window = max(0, min(int(line_window), 20))
    lo, hi = line - window, line + window
    records = []
    for ln in range(lo, hi + 1):
        records.extend(store.data_flow_by_line.get(ln, []))

    seen = set()
    unique = []
    for rec in records:
        key = (rec.variable, rec.function, rec.access, rec.location, rec.process)
        if key in seen:
            continue
        seen.add(key)
        unique.append(_df_to_dict(rec))

    source = store.get_source_slice(lo, hi)
    control_at_site = [_cf_to_dict(r) for r in store.control_by_line.get(line, [])]
    index_ops = extract_index_operands([t for _, t in source])

    return _dump(
        {
            "alarm": meta,
            "data_flow_records": unique[:40],
            "control_flow_at_site": control_at_site[:20],
            "source_lines": [{"line": ln, "text": text} for ln, text in source],
            "indexed_objects": index_ops.get("indexed_objects") or [],
            "index_operand_candidates": index_ops.get("index_operand_candidates") or [],
            "note": (
                "Raw retrieval only. alarm.message is Astrée's ABSTRACT interval "
                "diagnostic when exported (e.g. [lo,hi] vs declared range) — not a "
                "concrete runtime index and not a human true/false label. For OOB: "
                "(1) get_declaration_bounds on indexed_objects (e.g. Obj.raw), not "
                "only the DF aggregate name; (2) get_backward_slice on "
                "index_operand_candidates — if parameter_note / call_site_arguments "
                "appear, next-slice those argument identifiers; (3) never treat "
                "Astrée hi vs array_size alone as proof of classification true."
            ),
        }
    )


@tool
def get_function_snippet(
    alarm_order_id: int,
    context_lines: int = 12,
    max_snippet_lines: int = 40,
) -> str:
    """Return the function source snippet containing the alarm location.

    Resolves the alarm line in input.c, finds the enclosing function, and returns
    a bounded source snippet plus compact control-flow edges for that function.

    Args:
        alarm_order_id: Astrée alarm Order id.
        context_lines: Preferred half-window around the alarm line.
        max_snippet_lines: Cap on returned snippet size.
    """
    store = _store()
    meta = _alarm_payload(alarm_order_id)
    if "error" in meta:
        return _dump(meta)

    parsed = meta["parsed_location"]
    if not parsed:
        return _dump({**meta, "error": "Could not parse alarm location"})

    line = parsed["line"]
    name, func_start, func_end = store.find_enclosing_function(line)
    max_lines = max(10, min(int(max_snippet_lines), 120))
    half = max(1, max_lines // 2)
    start = max(func_start, line - half)
    end = min(func_end, start + max_lines - 1)
    if end - start + 1 < max_lines:
        start = max(func_start, end - max_lines + 1)

    snippet = store.get_source_slice(start, end)
    ctx = max(0, min(int(context_lines), 40))
    highlight = store.get_source_slice(max(1, line - ctx), line + ctx)

    # Compact unique callee / caller names instead of full edge dumps.
    outgoing = sorted(
        {r.callee for r in store.control_by_caller.get(name or "", []) if r.callee}
    )[:40]
    incoming = sorted(
        {r.caller for r in store.control_by_callee.get(name or "", []) if r.caller}
    )[:40]
    site_edges = [_cf_to_dict(r) for r in store.control_by_line.get(line, [])]

    return _dump(
        {
            "alarm": meta,
            "function_name": name,
            "function_span": {
                "start_line": func_start,
                "end_line": func_end,
                "snippet_start": start,
                "snippet_end": end,
            },
            "alarm_highlight": [{"line": ln, "text": text} for ln, text in highlight],
            "function_snippet": [
                {"line": ln, "text": text} for ln, text in snippet
            ],
            "callees_of_function": outgoing,
            "callers_of_function": incoming,
            "control_flow_at_alarm_line": site_edges,
            "note": (
                "Raw retrieval only. Decide which portion of the snippet is "
                "relevant to the alarm."
            ),
        }
    )


@tool
def get_variable_manipulation_sequence(
    alarm_order_id: int,
    variable_name: str = "",
    max_events: int = 40,
) -> str:
    """Return the sequence of functions that read/write variables related to an alarm.

    If variable_name is empty, uses every variable appearing in data-flow records
    near the alarm line. Returns ordered access events and a compact function list.
    Prefer passing a specific variable_name after inspecting get_affected_symbols.

    Args:
        alarm_order_id: Astrée alarm Order id.
        variable_name: Optional specific variable / array / pointer / struct field
            name (bare name or full Astrée variable string). Empty = all at site.
        max_events: Maximum access events to return per variable.
    """
    store = _store()
    meta = _alarm_payload(alarm_order_id)
    if "error" in meta:
        return _dump(meta)

    parsed = meta["parsed_location"]
    if not parsed:
        return _dump({**meta, "error": "Could not parse alarm location"})

    line = parsed["line"]
    site_records = []
    for ln in range(line - 2, line + 3):
        site_records.extend(store.data_flow_by_line.get(ln, []))

    if variable_name.strip():
        key = variable_name.strip()
        bare = key.split("@", 1)[0].strip('"')
        variables = [bare]
    else:
        variables = sorted(
            {
                r.variable.split("@", 1)[0].strip('"')
                for r in site_records
                if r.variable
            }
        )

    max_ev = max(1, min(int(max_events), 80))
    per_var = []
    involved_functions: set[str] = set()

    for var in variables[:8]:
        events = store.data_flow_by_variable.get(var, [])
        # Prefer events in the same function as the alarm site, then nearby lines.
        site_fns = {r.function for r in site_records if r.function}
        same_fn = [r for r in events if r.function in site_fns]
        others = [r for r in events if r.function not in site_fns]

        def _sort_key(r):
            return (
                abs((r.line or 10**12) - line),
                r.line if r.line is not None else 10**12,
                r.access,
                r.function,
            )

        ordered = sorted(same_fn, key=_sort_key) + sorted(others, key=_sort_key)
        # Dedup (function, access, line)
        seen = set()
        trimmed = []
        for r in ordered:
            key = (r.function, r.access, r.line, r.process)
            if key in seen:
                continue
            seen.add(key)
            trimmed.append(r)
            if len(trimmed) >= max_ev:
                break

        for r in trimmed:
            if r.function:
                involved_functions.add(r.function)

        # Unique function access summary for the agent.
        fn_summary = []
        seen_fn = set()
        for r in trimmed:
            sk = (r.function, r.access)
            if sk in seen_fn:
                continue
            seen_fn.add(sk)
            fn_summary.append(
                {
                    "function": r.function,
                    "access": r.access,
                    "example_location": r.location,
                    "line": r.line,
                }
            )

        per_var.append(
            {
                "variable": var,
                "event_count_sampled": len(trimmed),
                "event_count_total_indexed": len(events),
                "function_access_summary": fn_summary[:30],
                "sample_events": [_df_to_dict(r) for r in trimmed[:15]],
            }
        )

    related_cf = []
    seen_cf = set()
    for fn in sorted(involved_functions):
        for rec in store.control_by_caller.get(fn, [])[:30]:
            if rec.callee in involved_functions:
                key = (rec.caller, rec.callee)
                if key in seen_cf:
                    continue
                seen_cf.add(key)
                related_cf.append(_cf_to_dict(rec))
        if len(related_cf) >= 40:
            break

    return _dump(
        {
            "alarm": meta,
            "requested_variable": variable_name,
            "variables": per_var,
            "related_control_flow_among_involved_functions": related_cf,
            "involved_functions": sorted(involved_functions),
            "note": (
                "Raw retrieval only. Decide the meaningful manipulation sequence "
                "and narrative from these events and edges."
            ),
        }
    )


@tool
def get_declaration_bounds(symbol_name: str) -> str:
    """Look up the declaration record for a symbol name in input.c.

    Retrieval only: returns declared type text, literal array size when the
    declarator uses a numeric constant, pointer-ness, source location, and a
    parse_note when the size/type could not be resolved unambiguously.
    Does not classify safety or compare sizes to indices.

    For union/struct objects (e.g. TcpIp_IpV6EthBufData), returns the object
    type plus nested_array_fields (e.g. .raw[2]) — the object itself may not
    be an array. Prefer looking up indexed_objects like ``Obj.raw`` when the
    access is ``Obj.raw[i]``.

    This tool indexes ``input.c`` declarations — it does NOT read data-flow
    CSV rows. DF access records prove a symbol is used, not how it is declared.

    Args:
        symbol_name: Bare symbol / array / pointer / ``Obj.field`` name.
    """
    store = _store()
    key = (symbol_name or "").strip().split("@", 1)[0].strip('"')
    info = store.resolve_declaration_lookup(key)
    nested = [
        {
            "name": f.name,
            "array_size": f.array_size,
            "declared_type": f.declared_type,
            "location": f.location,
            "raw_declaration_text": f.raw_declaration_text,
            "parse_note": f.parse_note,
        }
        for f in (info.get("nested_array_fields") or [])
    ]

    if not info.get("found"):
        suggestions = store.declaration_name_suggestions(key)
        # Suggest nested fields / object.field if object type known.
        typ = store.object_types.get(key)
        if typ:
            for f in store.type_array_fields.get(typ, []):
                suggestions.append(f"{key}.{f.name.split('.')[-1]}")
        return _dump(
            {
                "symbol": key,
                "found": False,
                "error": f"No declaration indexed for symbol {key!r}",
                "parse_note": (
                    "No declaration row in the input.c index for this exact bare "
                    "name. This is a data/index miss (name spelling, macro-built "
                    "declarator, or skipped form) — not a parsed-but-ambiguous size. "
                    "Data-flow CSV rows for this name (if any) only record accesses, "
                    "not the C declarator. Try indexed_objects / Obj.field from "
                    "get_affected_symbols when the access is Obj.field[i]."
                ),
                "name_suggestions": suggestions[:12],
                "object_type": typ,
                "nested_array_fields": nested,
                "alternate_hits": 0,
                "note": (
                    "Raw lookup only. found=false with parse_note means the index "
                    "has no entry; it does not mean the object is unbounded. "
                    "DF presence ≠ declaration."
                ),
            }
        )

    rec: DeclarationRecord = info["declaration"]
    size = None if rec.array_size == 0 else rec.array_size
    return _dump(
        {
            "symbol": key,
            "found": True,
            "kind": info.get("kind"),
            "object_type": info.get("object_type"),
            "declared_type": rec.declared_type,
            "array_size": size,
            "is_pointer": rec.is_pointer,
            "location": rec.location,
            "raw_declaration_text": rec.raw_declaration_text,
            "nested_array_fields": nested,
            "parse_note": (
                rec.parse_note
                if rec.array_size != 0
                else (
                    (rec.parse_note + " " if rec.parse_note else "")
                    + "array_size literal 0 suppressed (likely call-site / "
                    "non-declarator false match); treat size as unresolved."
                ).strip()
            ),
            "note": (
                "Raw retrieval only from input.c (not data-flow.csv). "
                "array_size is set only for positive literal sizes. For objects "
                "with nested_array_fields, the bound that matters is usually a "
                "member (e.g. Obj.raw), not the object itself. Do not treat "
                "missing size as a safety verdict."
            ),
        }
    )


@tool
def get_backward_slice(
    variable_name: str,
    from_location: str,
    max_depth: int = 8,
) -> str:
    """Walk backward over DF writes and CF callers from a location.

    Retrieval only: returns write events that appear earlier in source order
    within visited functions, plus unresolved_paths when the CF graph has no
    caller or max_depth/write limits stop the walk. truncated=true means a
    structural limit was hit — not a judgment about whether the alarm is real.

    For out-of-bounds alarms, pass the INDEX or pointer operand being traced
    (e.g. checkword, idxMRPChnl_u16), not the array/table name being indexed.
    Empty writes_found on a const lookup table does not explain the index.

    Args:
        variable_name: Bare variable / index symbol whose value origin to trace.
        from_location: Astrée location string (file:line.col-...).
        max_depth: Max CF caller hops to climb (default 8).
    """
    store = _store()
    depth = max(0, min(int(max_depth), 32))
    payload = store.backward_slice(
        variable_name,
        from_location,
        max_depth=depth,
    )
    payload["note"] = (
        "Raw retrieval only. writes_found lists earlier DF writes encountered "
        "while climbing callers (each frame filtered by that call-site location). "
        "lookup explains bare/qualified DF key matches and writes excluded only "
        "because they appear at/after the filter location. unresolved_paths / "
        "truncated describe graph boundaries or depth limits — not whether the "
        "access is out of bounds. If parameter_note is set, the symbol is likely "
        "a function parameter with no local writes — do NOT invent an unrelated "
        "global for-loop as its origin. When call_site_arguments is present, "
        "slice/guard those next_slice_candidates next."
    )
    return _dump(payload)


@tool
def get_call_site_arguments(
    callee_function: str,
    parameter_name: str,
    from_location: str = "",
) -> str:
    """Retrieve caller argument expressions for a callee parameter.

    Use when get_backward_slice returns parameter_note / empty local writes for
    an index that is a function parameter. Returns CF call sites, the argument
    expression passed for that parameter, identifiers to slice next, and an
    optional parameter type-width hint (Astrée [0, 255] often equals uint8
    domain max — not proof of a runtime index of 255).

    Args:
        callee_function: Function that declares the parameter (enclosing fn).
        parameter_name: Parameter / index symbol name (e.g. Index).
        from_location: Optional alarm location to disambiguate the definition.
    """
    store = _store()
    payload = store.find_call_site_arguments(
        callee_function,
        parameter_name,
        from_location=from_location or None,
        max_sites=12,
    )
    return _dump(payload)


def parse_index_expression_structure(expression_text: str) -> dict:
    """Structurally parse a C index / subscript expression (no range math)."""
    raw = (expression_text or "").strip()
    if not raw:
        return {
            "raw": "",
            "found": False,
            "parse_note": "Empty expression_text.",
        }

    # Prefer first [...] contents if a full statement was pasted.
    expr = raw
    i = 0
    while i < len(raw):
        if raw[i] != "[":
            i += 1
            continue
        depth = 0
        j = i
        while j < len(raw):
            if raw[j] == "[":
                depth += 1
            elif raw[j] == "]":
                depth -= 1
                if depth == 0:
                    expr = raw[i + 1 : j].strip()
                    i = len(raw)
                    break
            j += 1
        else:
            break

    operations: list[dict] = []
    work = expr

    def _strip_outer_parens(text: str) -> str:
        t = text.strip()
        while t.startswith("(") and t.endswith(")"):
            depth = 0
            wraps = True
            for k, ch in enumerate(t):
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0 and k != len(t) - 1:
                        wraps = False
                        break
            if wraps and depth == 0:
                t = t[1:-1].strip()
            else:
                break
        return t

    work = _strip_outer_parens(work)

    cast_re = re.compile(r"^\(\s*([A-Za-z_][\w\s]*?)\s*\)\s*")
    while True:
        m = cast_re.match(work)
        if not m:
            break
        operations.append(
            {"op": "cast", "type": re.sub(r"\s+", " ", m.group(1)).strip()}
        )
        work = _strip_outer_parens(work[m.end() :])

    # Peel binary ops from the outside (low precedence first for index idioms).
    bin_ops = [
        (r"\|", "bit_or"),
        (r"\^", "bit_xor"),
        (r"&", "bit_and"),
        (r"<<", "left_shift"),
        (r">>", "right_shift"),
        (r"\+", "add"),
        (r"-", "subtract"),
        (r"\*", "multiply"),
        (r"/", "divide"),
        (r"%", "modulo"),
    ]

    def _find_top_level(op_pat: str, text: str) -> Optional[re.Match]:
        depth = 0
        hit = None
        for m in re.finditer(op_pat, text):
            prefix = text[: m.start()]
            if prefix.count("(") - prefix.count(")") != 0:
                continue
            # Avoid matching the second char of << >> or compound assigns.
            if m.start() > 0 and text[m.start() - 1] in "<>+-*/%&|^=":
                continue
            if m.end() < len(text) and text[m.end()] in "<>=":
                continue
            hit = m
        return hit

    for _ in range(8):
        found_op = False
        for pat, op_name in bin_ops:
            hit = _find_top_level(pat, work)
            if hit is None:
                continue
            left = work[: hit.start()].strip()
            right = work[hit.end() :].strip()
            if not left or not right:
                continue
            # Skip unary +/-
            if op_name in {"add", "subtract"} and (
                not left or left[-1] in "(,=<>!&|^+-*/%"
            ):
                continue
            rhs = _strip_outer_parens(right)
            amount: object = rhs
            lit = re.fullmatch(
                r"([+\-]?(?:0[xX][0-9A-Fa-f]+|\d+))(?:[uUlL]{0,3})",
                rhs,
            )
            if lit:
                try:
                    amount = int(lit.group(1), 0)
                except ValueError:
                    amount = rhs
            operations.append({"op": op_name, "amount": amount, "rhs_text": right})
            work = _strip_outer_parens(left)
            found_op = True
            break
        if not found_op:
            break

    operand = re.sub(r"\s+", "", work.strip())
    fields = operand.split(".") if "." in operand else []

    return {
        "raw": raw,
        "expression": expr,
        "found": True,
        "operand": operand,
        "fields": fields,
        "operations": operations,
        "note": (
            "Structure only. Agent must reason about how operations transform "
            "the operand's value range — this tool does not compute ranges or "
            "compare to Astrée intervals."
        ),
    }


@tool
def get_condition_guards(
    variable_name: str,
    from_location: str,
    max_depth: int = 8,
) -> str:
    """Find conditional guards referencing a variable before a location.

    Retrieval only: walks DF/CF-related enclosing functions (same climb idea as
    get_backward_slice) and returns raw if / else-if / switch / assert / ternary
    text that mentions variable_name. Does NOT evaluate whether a guard is
    sufficient or correctly ordered relative to the access.

    Use when a slice shows a sentinel / invalid / default initializer — look for
    checks between that write and the risky access (or in callers).

    Args:
        variable_name: Index / status symbol to search for in conditions.
        from_location: Astrée location of the access (or nearby).
        max_depth: Max CF caller hops (default 8).
    """
    store = _store()
    payload = store.find_condition_guards(
        variable_name,
        from_location,
        max_depth=max(0, min(int(max_depth), 32)),
    )
    payload["note"] = (
        "Raw retrieval only. Presence of a guard does not confirm it prevents "
        "this specific access — reason about control-flow position and compared "
        "values (use resolve_symbolic_constant on named constants in the guard)."
    )
    return _dump(payload)


@tool
def resolve_symbolic_constant(name: str) -> str:
    """Resolve an enum member or #define to a literal integer when possible.

    Retrieval only: returns value / value_text / kind (define|enum_member) or a
    parse_note when the constant is computed/conditional rather than a plain
    literal. Does not infer meaning from the identifier spelling alone.

    Args:
        name: Bare enumerator or macro name (e.g. NvM_Prv_idJob_Invalid_e___…).
    """
    store = _store()
    payload = store.resolve_symbolic_constant(name)
    payload["note"] = (
        "Raw lookup only. A resolved numeric value is a fact from input.c — not "
        "a safety verdict. Do not treat the identifier spelling (Invalid, MAX, …) "
        "as proof without a resolved value and related guards."
    )
    return _dump(payload)


@tool
def get_index_expression_structure(expression_text: str) -> str:
    """Parse a C index expression into operand + operations (no range eval).

    Retrieval only: exposes shifts, masks, arithmetic, field access, and casts
    structurally. Prefer pasting the subscript / index expression from the
    alarm snippet (e.g. ``(idDFC.id) >> 4u`` or a full ``arr[…]`` line).

    Args:
        expression_text: Index / subscript expression or a statement containing one.
    """
    payload = parse_index_expression_structure(expression_text)
    return _dump(payload)


@tool
def get_all_writes_to_symbol(symbol_name: str) -> str:
    """List every DF write to a symbol across the indexed program (flat).

    Retrieval only and exhaustive within the DF CSV index — not filtered by
    call-graph reachability or depth. Use as a cross-check when
    get_backward_slice is truncated or misses an initializer. Reachability to
    the alarm site still requires slice / CF reasoning.

    Args:
        symbol_name: Bare variable / index symbol name.
    """
    store = _store()
    payload = store.get_all_writes_to_symbol(symbol_name)
    payload["note"] = (
        "Flat DF write listing only. A write appearing here is not automatically "
        "reachable from the alarm site — combine with get_backward_slice / "
        "get_condition_guards for path-local evidence."
    )
    return _dump(payload)


TOOLS = [
    get_affected_symbols,
    get_function_snippet,
    get_variable_manipulation_sequence,
    get_declaration_bounds,
    get_backward_slice,
    get_call_site_arguments,
    get_condition_guards,
    resolve_symbolic_constant,
    get_index_expression_structure,
    get_all_writes_to_symbol,
]
