"""Deterministic per-alarm case file: operand, object bounds, origin path."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from aoob_agent.data_store import (
    DataStore,
    callees_from_text,
    collapse_ws,
    condition_compares_token,
    identifiers_from_text,
    is_constant_init_text,
    member_paths_from_text,
    subscripts_from_text,
    token_in_text,
)

_SKIP_IDENT = frozenset(
    {
        "if",
        "for",
        "while",
        "return",
        "sizeof",
        "void",
        "uint8",
        "uint16",
        "uint32",
        "uint8_t",
        "uint16_t",
        "uint32_t",
        "int",
        "const",
        "static",
        "null",
        "true",
        "false",
        "else",
        "switch",
        "case",
        "break",
        "continue",
        "unsigned",
        "signed",
        "struct",
        "enum",
        "typedef",
        "volatile",
        "NULL",
        "TRUE",
        "FALSE",
        "__aoob_snip",
    }
)

_LEADING_KEYWORDS = (
    "return",
    "if",
    "else",
    "while",
    "for",
    "switch",
    "case",
    "sizeof",
    "void",
)


def sanitize_symbol(name: str) -> str:
    return (name or "").strip().split("@", 1)[0].strip().strip('"')


def normalize_member_access(expr: str) -> str:
    text = (expr or "").strip().replace("->", ".")
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == ".":
            out.append(".")
            i += 1
            while i < n and text[i].isspace():
                i += 1
            continue
        if ch.isspace() and out and out[-1] == ".":
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out).strip()


def _strip_leading_keyword(text: str) -> str:
    s = (text or "").strip()
    lower = s.lower()
    for kw in _LEADING_KEYWORDS:
        if lower.startswith(kw):
            rest = s[len(kw) :]
            if not rest or not (rest[0].isalnum() or rest[0] == "_"):
                return rest.strip()
    return s


def _rightmost_member_path(text: str) -> str:
    """Keep the rightmost Ident(.Ident)* suffix without regex."""
    s = (text or "").strip()
    if not s:
        return s
    i = len(s) - 1
    while i >= 0:
        ch = s[i]
        if ch.isalnum() or ch in "._":
            i -= 1
            continue
        break
    return s[i + 1 :].strip(" .")


def clean_indexed_object(raw: str) -> str:
    """Strip statement/cast junk so only the array lvalue remains."""
    text = _strip_leading_keyword((raw or "").strip())
    while text.startswith("("):
        text = text[1:].lstrip()
    while text.endswith("("):
        text = text[:-1].rstrip()
    text = text.strip().lstrip(">*+&")
    text = normalize_member_access(text)
    paths = member_paths_from_text(text)
    if paths:
        return paths[-1]
    ident = _rightmost_member_path(text)
    return ident or text.strip()


def extract_index_accesses(line_text: str) -> list[dict[str, Any]]:
    """Indexed[operand] accesses with 0-based column of '['."""
    out: list[dict[str, Any]] = []
    for rec in subscripts_from_text(line_text or ""):
        indexed = str(rec.get("indexed") or "")
        if indexed.startswith("(") or _strip_leading_keyword(indexed) != indexed:
            indexed = clean_indexed_object(indexed)
        else:
            indexed = normalize_member_access(indexed)
        operand = str(rec.get("operand") or "").strip()
        if indexed and operand:
            out.append(
                {
                    "indexed": indexed,
                    "operand": operand,
                    "bracket_col": rec.get("bracket_col", 0),
                }
            )
    return out


def extract_index_operands(line_text: str) -> list[tuple[str, str]]:
    """Extract indexed[operand] pairs from one source line."""
    return [(a["indexed"], a["operand"]) for a in extract_index_accesses(line_text)]


def pick_alarm_index_access(
    line_text: str, col_start: Optional[int] = None
) -> tuple[str, str]:
    """Choose the access Astrée pointed at (column), else the outermost []."""
    accesses = extract_index_accesses(line_text)
    if not accesses:
        return "", ""
    if col_start is not None:
        # Astrée columns are 1-based.
        target = max(0, int(col_start) - 1)
        accesses = sorted(accesses, key=lambda a: abs(a["bracket_col"] - target))
        return accesses[0]["indexed"], accesses[0]["operand"]
    nested = [a for a in accesses if "[" in a["operand"]]
    chosen = nested[0] if nested else accesses[0]
    return chosen["indexed"], chosen["operand"]


def operand_symbol_candidates(expr: str) -> list[str]:
    """Identifiers in an index operand, members first, then bare names."""
    out: list[str] = []
    seen: set[str] = set()
    for name in member_paths_from_text(expr or ""):
        name = normalize_member_access(name)
        if name and name not in seen:
            seen.add(name)
            out.append(name)
    for token in identifiers_from_text(expr or ""):
        if not token or token in _SKIP_IDENT or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def callees_in_expression(expr: str) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for name in callees_from_text(expr or ""):
        if not name or name in _SKIP_IDENT or name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def rank_operand_symbol(
    *,
    operand_text: str,
    indexed_object: str,
    param_names: set[str],
    df_names: set[str],
    declared_names: set[str],
) -> str:
    """Pick the index operand, never the array and never a wrapper callee."""
    candidates = operand_symbol_candidates(operand_text)
    callees = set(callees_in_expression(operand_text))
    indexed_root = sanitize_symbol(indexed_object).split(".", 1)[0]
    compact = collapse_ws(operand_text or "").replace(" ", "")
    scored: list[tuple[int, str]] = []
    for cand in candidates:
        root = cand.split(".", 1)[0]
        if cand == indexed_object or root == indexed_root:
            continue
        if cand in _SKIP_IDENT:
            continue
        score = 0
        if cand in param_names or root in param_names:
            score += 100
        if cand in df_names or root in df_names:
            score += 50
        if cand in declared_names or root in declared_names:
            score += 20
        if cand in callees or root in callees:
            score -= 80
        # Nested index: Foo[Bar[i]] — Foo is the inner array, not the outer index.
        if f"{root}[" in compact or f"{cand}[" in compact:
            score -= 120
        scored.append((score, cand))
    scored.sort(key=lambda x: (-x[0], len(x[1])))
    if scored and scored[0][0] > -80:
        return scored[0][1]
    # Last identifier in the expression is usually the argument of Foo(x).
    idents = [
        c
        for c in candidates
        if c not in callees
        and c.split(".", 1)[0] != indexed_root
        and f"{c.split('.', 1)[0]}[" not in compact
    ]
    if idents:
        return idents[-1]
    return candidates[0] if candidates else sanitize_symbol(operand_text)


@dataclass
class IndexedObject:
    name: str
    kind: str
    declared_type: str
    array_size: Optional[int]
    is_pointer: bool
    location: str
    parse_note: Optional[str] = None
    raw_declaration_text: str = ""


@dataclass
class PathStep:
    step: int
    function: str
    role: str
    symbol: str
    scope: str
    line: Optional[int]
    location: str
    access: str
    sequence: int = 0
    declared_size: Optional[int] = None
    datatype: str = ""
    first_access: bool = False
    argument_expression: Optional[str] = None
    call_site: Optional[str] = None
    note: str = ""
    guards: list[str] = field(default_factory=list)


@dataclass
class CaseFile:
    order_id: int
    alarm_type: str
    alarm_category: str
    location: str
    astree_message: str
    alarm_function: str
    alarm_line: Optional[int]
    alarm_line_text: str
    index_expression: str
    operand_symbol: str
    operand_scope_at_alarm: str
    indexed_object: IndexedObject
    helpers: list[str] = field(default_factory=list)
    write_helpers: list[str] = field(default_factory=list)
    path: list[PathStep] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    extra_callers: list[str] = field(default_factory=list)
    guards: list[str] = field(default_factory=list)

    @property
    def array_size(self) -> Optional[int]:
        return self.indexed_object.array_size

    def path_records(self) -> list[dict[str, Any]]:
        """Structured path steps for the case brief and the UI."""
        rows: list[dict[str, Any]] = []
        for step in self.path:
            rows.append(
                {
                    "sequence": step.sequence or step.step,
                    "step": step.step,
                    "function": step.function,
                    "role": step.role,
                    "symbol": step.symbol,
                    "scope": step.scope,
                    "access": step.access,
                    "declared_size": step.declared_size,
                    "datatype": step.datatype,
                    "line": step.line,
                    "location": step.location,
                    "first_access": step.first_access,
                    "argument_expression": step.argument_expression,
                    "call_site": step.call_site,
                    "note": step.note,
                    "guards": list(step.guards or []),
                }
            )
        return rows

    def path_summary(self) -> str:
        lines = [
            f"Origin path ({len(self.path)} steps, walk origin -> alarm):",
        ]
        if not self.path:
            return "Origin path: empty"
        for step in self.path:
            size = step.declared_size if step.declared_size is not None else "unknown"
            first = " first" if step.first_access else ""
            extra = f"  arg={step.argument_expression}" if step.argument_expression else ""
            lines.append(
                f"  {step.sequence or step.step}. [{step.role}] {step.function}  "
                f"{step.symbol} ({step.scope})  access={step.access}  "
                f"size={size}  dtype={step.datatype or '—'}  "
                f"line={step.line}{first}  @{step.location}{extra}"
            )
        return "\n".join(lines)

    def brief(self) -> str:
        obj = self.indexed_object
        size = obj.array_size if obj.array_size is not None else "unknown"
        lines = [
            "--- Case file (compiled, not a verdict) ---",
            f"Order: {self.order_id}",
            f"Alarm: {self.location}",
            f"Astrée: {self.astree_message}",
            f"Alarm function: {self.alarm_function}",
            f"Indexed object: {obj.name}  kind={obj.kind}  dtype={obj.declared_type}  size={size}",
            f"Index expression: {self.index_expression or '(unparsed)'}",
            f"Operand: {self.operand_symbol}  scope_at_alarm={self.operand_scope_at_alarm}",
        ]
        if self.helpers:
            lines.append("Helpers available to inspect(): " + ", ".join(self.helpers))
        if self.write_helpers:
            lines.append(
                "Of those, these mutate the operand before the alarm access and "
                "MUST be inspected before a true/false verdict: "
                + ", ".join(self.write_helpers)
            )
        if self.indexed_object.raw_declaration_text:
            lines.append(
                "Indexed object has a declaration/initializer available via "
                "inspect_declaration() — use it if a guard's truth depends on "
                "a specific slot's contents (e.g. a sentinel entry)."
            )
        lines.append(self.path_summary())
        if self.gaps:
            lines.append("Gaps: " + ", ".join(self.gaps))
        if self.guards:
            lines.append(
                "Python-extracted guards / origin inits (use these; do not invent):"
            )
            for g in self.guards[:12]:
                lines.append(f"  - {g}")
        lines.append(
            "Call get_window for a step's full function, inspect listed helpers if needed, "
            "then submit_verdict. Gaps size_unknown / object_undeclared ⇒ review."
        )
        lines.append("--- End case file ---")
        return "\n".join(lines)


def _df_names_at_line(store: DataStore, line: Optional[int]) -> set[str]:
    if line is None:
        return set()
    names: set[str] = set()
    for rec in store.data_flow_by_line.get(line, []):
        bare = sanitize_symbol(rec.variable)
        if bare:
            names.add(bare)
            names.add(bare.split(".", 1)[0])
    return names


def _infer_scope(
    store: DataStore,
    symbol: str,
    function: str,
    line: Optional[int],
) -> str:
    root = sanitize_symbol(symbol).split(".", 1)[0]
    if not root or not function:
        return "unknown"
    params = store.parse_function_parameters(function, near_line=line)
    if any((p.get("name") or "") == root for p in params):
        return "parameter"
    info = store.resolve_declaration_lookup(root)
    decl = info.get("declaration") if info.get("found") else None
    loc = getattr(decl, "location", "") if decl else ""
    dline = store.parse_location_line(loc) if loc else None
    if dline is not None:
        dfn, start, end = store.find_enclosing_function(dline)
        if dfn == function and start is not None and end is not None and start <= dline <= end:
            return "local"
        if info.get("kind") == "array" or (decl and getattr(decl, "array_size", None)):
            return "global"
        if dfn and dfn != function:
            return "global"
        if not dfn:
            return "global"
    for rec in store.data_flow_by_variable.get(root, []):
        if rec.function == function:
            return "local"
    return "unknown"


def _find_operand_mutating_calls(
    store: DataStore,
    *,
    function_name: str,
    anchor_line: Optional[int],
    operand_root: str,
    exclude: set[str],
) -> list[str]:
    """Callees, inside the alarm function, passed the operand (often as &op).

    callees_in_expression(index_expr) only sees calls written directly inside
    the subscript, e.g. Foo[Reader(checkword)]. It misses calls elsewhere in
    the same function body that mutate the operand in place before later
    loop iterations reuse it, e.g. `Update(&checkword, core, tsk);` inside
    the same while-loop that later re-reads `Foo[Reader(checkword)]`. Those
    calls are exactly the ones that decide whether the index can ever exceed
    the array's capacity, so they must be inspectable (and, for true/false
    verdicts, inspected) even though they never appear in the index text.
    """
    if not function_name or not operand_root:
        return []
    start, end = None, None
    if anchor_line is not None:
        _fn, start, end = store.find_enclosing_function(anchor_line)
    if start is None or end is None:
        span = store.function_span(function_name, near_line=anchor_line)
        if not span:
            return []
        start, end = span
    found: list[str] = []
    seen: set[str] = set()
    for _ln, text in store.get_source_slice(start, end):
        if not token_in_text(text, operand_root):
            continue
        for callee in callees_from_text(text):
            if (
                not callee
                or callee in _SKIP_IDENT
                or callee in exclude
                or callee in seen
            ):
                continue
            seen.add(callee)
            found.append(callee)
    return found


def _lookup_object(store: DataStore, name: str) -> IndexedObject:
    key = clean_indexed_object(sanitize_symbol(name))
    if not key or key == "*()":
        return IndexedObject(
            name=key or name,
            kind="unknown",
            declared_type="",
            array_size=None,
            is_pointer=True,
            location="",
            parse_note="Pointer dereference form; capacity may be dynamic.",
        )

    def _from_decl(
        decl: Any,
        *,
        display: str,
        kind: str,
        note: Optional[str] = None,
    ) -> IndexedObject:
        size = getattr(decl, "array_size", None) if decl else None
        is_ptr = bool(getattr(decl, "is_pointer", False)) if decl else False
        out_kind = kind
        if is_ptr and kind not in {"array", "field"}:
            out_kind = "pointer"
        pnote = note
        if pnote is None and decl is not None:
            pnote = getattr(decl, "parse_note", None)
        return IndexedObject(
            name=display,
            kind=out_kind,
            declared_type=getattr(decl, "declared_type", "") if decl else "",
            array_size=size if isinstance(size, int) else None,
            is_pointer=is_ptr,
            location=getattr(decl, "location", "") if decl else "",
            parse_note=pnote,
            raw_declaration_text=getattr(decl, "raw_declaration_text", "") if decl else "",
        )

    info = store.resolve_declaration_lookup(key)
    decl = info.get("declaration") if info.get("found") else None
    own_size = getattr(decl, "array_size", None) if decl else None
    if info.get("found") and isinstance(own_size, int):
        return _from_decl(decl, display=key, kind="array")

    # Struct.member: use that member's array size, never the first nested field.
    if "." in key:
        root, field = key.split(".", 1)
        tail = field.split(".")[-1]
        root_info = store.resolve_declaration_lookup(root)
        nested = list(root_info.get("nested_array_fields") or [])
        for fld in nested:
            fname = str(getattr(fld, "name", "") or "")
            ftail = fname.split(".")[-1]
            if fname in {key, field, tail} or ftail == tail or fname.endswith("." + tail):
                size = getattr(fld, "array_size", None)
                return _from_decl(
                    fld,
                    display=fname or key,
                    kind="array" if isinstance(size, int) else "field",
                    note=getattr(fld, "parse_note", None),
                )
        tail_info = store.resolve_declaration_lookup(tail)
        tail_decl = tail_info.get("declaration") if tail_info.get("found") else None
        tail_size = getattr(tail_decl, "array_size", None) if tail_decl else None
        if not isinstance(tail_size, int):
            for rec in store.declarations_all.get(tail, []) or []:
                sz = getattr(rec, "array_size", None)
                if isinstance(sz, int):
                    tail_decl = rec
                    tail_size = sz
                    break
        if isinstance(tail_size, int):
            return _from_decl(tail_decl, display=key, kind="array")

    if info.get("found") and decl is not None:
        kind = info.get("kind") or "unknown"
        if kind == "object":
            kind = "struct"
        elif kind == "array":
            kind = "array"
        elif getattr(decl, "is_pointer", False):
            kind = "pointer"
        elif kind in {"struct", "variable", "field"}:
            pass
        else:
            kind = "variable"
        return _from_decl(
            decl,
            display=key,
            kind=kind,
            note=(
                getattr(decl, "parse_note", None)
                or "Object/struct resolved; indexed member array was not."
            ),
        )

    root = key.split(".", 1)[0]
    if root and root != key:
        root_info = store.resolve_declaration_lookup(root)
        if root_info.get("found"):
            rdecl = root_info.get("declaration")
            return _from_decl(
                rdecl,
                display=key,
                kind="struct" if root_info.get("kind") == "object" else "unknown",
                note="Member path not declared as an array; size left unknown.",
            )
    return IndexedObject(
        name=key,
        kind="unknown",
        declared_type="",
        array_size=None,
        is_pointer=False,
        location="",
        parse_note="No declaration indexed for this object.",
    )


def _writes_before(
    store: DataStore,
    symbol: str,
    function: str,
    before_location: str,
) -> list[Any]:
    writes: list[Any] = list(
        store.get_prior_writes(symbol, before_location, function_scope=function)
    )
    before_line = store.parse_location_line(before_location)
    if before_line is None:
        return writes
    fn, start, end = store.find_enclosing_function(before_line)
    if fn != function or start is None or end is None:
        return writes
    hits = store.find_source_assignments(
        symbol,
        before_line=before_line,
        function_start=start,
        function_end=end,
    )
    seen_lines = {
        getattr(w, "line", None) if not isinstance(w, dict) else w.get("line")
        for w in writes
    }
    for hit in hits:
        ln = hit.get("line")
        if ln in seen_lines:
            continue
        writes.append(hit)
        seen_lines.add(ln)
    return writes


def _usable_arg_idents(idents: list[str], *, callee: str) -> list[str]:
    out: list[str] = []
    for raw in idents:
        x = (raw or "").strip()
        if not x or x == callee or x in _SKIP_IDENT:
            continue
        if x.endswith(("Type", "IterType", "PtrType")):
            continue
        # Generator placeholders like CName, not index variables like CfgIdx.
        if (
            "Idx" not in x
            and len(x) >= 3
            and x[0] == "C"
            and x[1].isupper()
            and x[2:].isalpha()
            and x[2:].islower()
        ):
            continue
        out.append(x)
    return out


def _looks_like_constant_init(text: str, token: str) -> bool:
    """True when the last assignment of ``token`` on the line is an integer literal."""
    tok = (token or "").strip()
    stmt = (text or "").split("//", 1)[0].strip()
    if not tok:
        return False
    return is_constant_init_text(stmt, tok)


def _collect_constant_inits(
    store: DataStore,
    symbol: str,
    function: str,
    before_line: Optional[int],
) -> list[str]:
    """Last assignment of the operand in this function, if it is a numeric literal."""
    if not function or before_line is None:
        return []
    fn, start, end = store.find_enclosing_function(int(before_line))
    if fn != function or start is None or end is None:
        return []
    hits = store.find_source_assignments(
        symbol,
        before_line=int(before_line),
        function_start=start,
        function_end=end,
    )
    if not hits:
        return []
    last = hits[-1]
    text = (last.get("assigned_expression_text") or "").strip()
    tok = sanitize_symbol(symbol).split(".")[-1]
    if not _looks_like_constant_init(text, tok):
        return []
    return [f"init in {function}: {text}"]


def _looks_like_bounds_guard(cond: str, token: str) -> bool:
    """Keep comparisons/for-limits on the operand; drop the OOB access itself."""
    text = cond or ""
    tok = (token or "").strip()
    if not tok or not token_in_text(text, tok):
        return False
    if text.strip().startswith("for") and token_in_text(text, tok):
        return True
    return condition_compares_token(text, tok)


def _collect_guards(
    store: DataStore,
    symbol: str,
    location: str,
    extra_tokens: Optional[list[str]] = None,
) -> list[str]:
    if not location:
        return []
    tokens = [sanitize_symbol(symbol).split(".")[-1], sanitize_symbol(symbol)]
    if extra_tokens:
        tokens.extend(extra_tokens)
    seen: set[str] = set()
    texts: list[str] = []
    for tok in tokens:
        tok = (tok or "").strip()
        if not tok or tok in _SKIP_IDENT or tok in seen:
            continue
        seen.add(tok)
        data = store.find_condition_guards(tok, location, max_depth=2, max_guards=16)
        for g in data.get("guards_found") or []:
            text = (g.get("condition_text") or "").strip()
            if not text:
                continue
            kind = g.get("kind") or "if"
            fn = g.get("function") or ""
            token_ok = any(
                _looks_like_bounds_guard(text, t)
                for t in tokens
                if t and t not in _SKIP_IDENT
            )
            if not token_ok:
                continue
            label = f"{kind} in {fn}: {text}" if fn else f"{kind}: {text}"
            if label not in texts:
                texts.append(label)
    return texts[:16]


def _param_index(store: DataStore, function: str, symbol: str, line: Optional[int]) -> Optional[int]:
    root = sanitize_symbol(symbol).split(".", 1)[0]
    for p in store.parse_function_parameters(function, near_line=line):
        if (p.get("name") or "") == root:
            return int(p.get("index", 0))
    return None


def _pick_caller(store: DataStore, function: str) -> tuple[Optional[Any], list[str]]:
    rows = list(store.control_by_callee.get(function, []))
    extras: list[str] = []
    seen: set[tuple[str, Optional[int]]] = set()
    unique = []
    for edge in rows:
        key = (edge.caller, edge.line)
        if not edge.caller or key in seen:
            continue
        seen.add(key)
        unique.append(edge)
    if not unique:
        return None, extras
    unique.sort(key=lambda e: (e.line is None, e.line or 10**12))
    extras = [e.caller for e in unique[1:6]]
    return unique[0], extras


def _build_path(
    store: DataStore,
    *,
    alarm_fn: str,
    alarm_loc: str,
    alarm_line: Optional[int],
    operand: str,
    gaps: list[str],
) -> tuple[list[PathStep], list[str]]:
    extra_callers: list[str] = []
    reverse: list[dict[str, Any]] = []
    fn = alarm_fn
    symbol = operand
    loc = alarm_loc
    line = alarm_line
    seen_fns: set[str] = set()

    for hop in range(8):
        if not fn or fn in seen_fns:
            if hop == 0:
                gaps.append("unresolved_function")
            break
        seen_fns.add(fn)
        scope = _infer_scope(store, symbol, fn, line)
        writes = _writes_before(store, symbol, fn, loc) if loc else []
        role = "alarm" if hop == 0 else "hop"
        access = "read" if hop == 0 else "write"
        note = ""
        arg_expr = None
        call_site = None

        if writes:
            last = writes[-1]
            wline = getattr(last, "line", None)
            wloc = getattr(last, "location", "")
            if isinstance(last, dict):
                wline = last.get("line")
                wloc = last.get("location") or loc
            if hop == 0:
                reverse.append(
                    {
                        "function": fn,
                        "role": "origin_and_alarm",
                        "symbol": symbol,
                        "scope": scope,
                        "line": line or wline,
                        "location": alarm_loc,
                        "access": "mixed",
                        "note": "Operand written in the alarm function before the access.",
                    }
                )
            else:
                reverse.append(
                    {
                        "function": fn,
                        "role": "origin",
                        "symbol": symbol,
                        "scope": "local" if scope == "parameter" else scope,
                        "line": wline,
                        "location": wloc or loc,
                        "access": "write",
                        "note": "Index origin write in this function.",
                    }
                )
            break

        if scope == "parameter":
            edge, extras = _pick_caller(store, fn)
            if extras and not extra_callers:
                extra_callers = extras
            if edge is None:
                note = "Parameter with no CF caller; origin unresolved."
                gaps.append("no_caller")
                reverse.append(
                    {
                        "function": fn,
                        "role": role,
                        "symbol": symbol,
                        "scope": scope,
                        "line": line,
                        "location": loc,
                        "access": access,
                        "note": note,
                    }
                )
                break
            pidx = _param_index(store, fn, symbol, line)
            extracted = None
            if pidx is not None and edge.line is not None:
                extracted = store.extract_call_argument_expression(fn, edge.line, pidx)
            if extracted:
                arg_expr = extracted.get("argument_expression")
                idents = _usable_arg_idents(
                    list(extracted.get("argument_identifiers") or []),
                    callee=fn,
                )
            else:
                gaps.append("arg_extract_failed")
                arg_expr = None
                idents = []
                note = (
                    "Caller found but argument expression was not extracted; "
                    "call-site line is the next window."
                )
            reverse.append(
                {
                    "function": fn,
                    "role": role,
                    "symbol": symbol,
                    "scope": scope,
                    "line": line,
                    "location": loc,
                    "access": access,
                    "argument_expression": arg_expr,
                    "call_site": edge.call_site,
                    "note": note or f"Passed from {edge.caller}.",
                }
            )
            fn = edge.caller
            loc = edge.call_site or loc
            line = edge.line
            if idents:
                symbol = idents[0]
            continue

        if hop == 0:
            global_hits: list[Any] = []
            if alarm_line is not None:
                global_hits = store.find_source_assignments_global(
                    symbol, before_line=alarm_line, max_hits=12
                )
            if global_hits:
                last = global_hits[-1]
                wfn = last.get("function") or ""
                wline = last.get("line")
                wloc = last.get("location") or loc
                if wfn and wfn != fn:
                    reverse.append(
                        {
                            "function": fn,
                            "role": "alarm",
                            "symbol": symbol,
                            "scope": scope,
                            "line": line,
                            "location": loc,
                            "access": "read",
                            "note": f"Index field last written in {wfn}.",
                        }
                    )
                    reverse.append(
                        {
                            "function": wfn,
                            "role": "origin",
                            "symbol": symbol,
                            "scope": "global",
                            "line": wline,
                            "location": wloc,
                            "access": "write",
                            "note": "Global/struct-field write of the index operand.",
                        }
                    )
                    break
            gaps.append("missing_df")
            reverse.append(
                {
                    "function": fn,
                    "role": "alarm",
                    "symbol": symbol,
                    "scope": scope,
                    "line": line,
                    "location": loc,
                    "access": "read",
                    "note": "No prior write indexed in this function.",
                }
            )
            if scope in {"local", "global", "unknown"}:
                break
        else:
            reverse.append(
                {
                    "function": fn,
                    "role": "origin",
                    "symbol": symbol,
                    "scope": scope,
                    "line": line,
                    "location": loc,
                    "access": "write",
                    "note": "Hop target with no indexed write; treat as origin window.",
                }
            )
            gaps.append("missing_df")
            break

    reverse.reverse()
    if reverse and reverse[0].get("role") not in {"origin", "origin_and_alarm"}:
        reverse[0]["role"] = "origin" if len(reverse) == 1 else reverse[0].get("role") or "origin"
        if len(reverse) == 1 and reverse[0]["role"] != "origin_and_alarm":
            reverse[0]["role"] = "origin_and_alarm" if reverse[0].get("function") == alarm_fn else "origin"
    if reverse and reverse[-1].get("function") == alarm_fn:
        if reverse[-1]["role"] not in {"alarm", "origin_and_alarm"}:
            reverse[-1]["role"] = "alarm" if len(reverse) > 1 else "origin_and_alarm"

    steps: list[PathStep] = []
    for i, raw in enumerate(reverse, start=1):
        extra_toks: list[str] = []
        if raw.get("argument_expression"):
            extra_toks = _usable_arg_idents(
                identifiers_from_text(str(raw.get("argument_expression"))),
                callee=str(raw.get("function") or ""),
            )
        guards = _collect_guards(
            store,
            str(raw.get("symbol") or operand),
            str(raw.get("location") or ""),
            extra_toks or None,
        )
        guards.extend(
            _collect_constant_inits(
                store,
                str(raw.get("symbol") or operand),
                str(raw.get("function") or ""),
                raw.get("line") or alarm_line,
            )
        )
        steps.append(
            PathStep(
                step=i,
                function=str(raw.get("function") or ""),
                role=str(raw.get("role") or "hop"),
                symbol=str(raw.get("symbol") or operand),
                scope=str(raw.get("scope") or "unknown"),
                line=raw.get("line"),
                location=str(raw.get("location") or ""),
                access=str(raw.get("access") or "other"),
                argument_expression=raw.get("argument_expression"),
                call_site=raw.get("call_site"),
                note=str(raw.get("note") or ""),
                guards=guards,
            )
        )
    if not steps:
        gaps.append("empty_path")
        steps.append(
            PathStep(
                step=1,
                function=alarm_fn,
                role="origin_and_alarm",
                symbol=operand,
                scope="unknown",
                line=alarm_line,
                location=alarm_loc,
                access="read",
                note="Fallback: alarm function only.",
            )
        )
    return steps, extra_callers


def compile_case(store: DataStore, order_id: int) -> CaseFile:
    alarm = store.get_alarm(order_id)
    if alarm is None:
        raise ValueError(f"Unknown alarm Order id: {order_id}")
    parsed = alarm.parsed_location
    alarm_line = parsed["line"] if parsed else None
    alarm_fn, _s, _e = (
        store.find_enclosing_function(alarm_line) if alarm_line is not None else (None, None, None)
    )
    alarm_fn = alarm_fn or ""
    alarm_text = ""
    if alarm_line is not None and 0 < alarm_line < len(store.source_lines):
        alarm_text = store.source_lines[alarm_line]

    pairs = extract_index_operands(alarm_text)
    col_start = parsed.get("col_start") if parsed else None
    indexed_name, index_expr = pick_alarm_index_access(alarm_text, col_start)
    if not indexed_name and pairs:
        indexed_name, index_expr = pairs[0]
    if not indexed_name:
        # Fall back to first DF name at the line as object; operand unknown.
        df_names = sorted(_df_names_at_line(store, alarm_line))
        indexed_name = df_names[0] if df_names else ""
        index_expr = ""

    obj = _lookup_object(store, indexed_name)
    params = {
        p.get("name") or ""
        for p in store.parse_function_parameters(alarm_fn, near_line=alarm_line)
        if p.get("name")
    }
    df_names = _df_names_at_line(store, alarm_line)
    declared = set(store.declarations.keys())
    operand = rank_operand_symbol(
        operand_text=index_expr or indexed_name,
        indexed_object=obj.name or indexed_name,
        param_names=params,
        df_names=df_names,
        declared_names=declared,
    )
    if not operand:
        operand = index_expr or indexed_name

    index_reader_helpers = [
        h
        for h in callees_in_expression(index_expr)
        if h != operand and h != obj.name
    ]
    operand_root = sanitize_symbol(operand).split(".", 1)[0]
    write_helpers = _find_operand_mutating_calls(
        store,
        function_name=alarm_fn,
        anchor_line=alarm_line,
        operand_root=operand_root,
        exclude={obj.name, operand, *index_reader_helpers},
    ) if operand_root else []
    helpers = list(dict.fromkeys([*index_reader_helpers, *write_helpers]))
    gaps: list[str] = []
    if not indexed_name:
        gaps.append("index_unparsed")
    if obj.array_size is None:
        gaps.append("capacity_unknown" if obj.kind == "pointer" else "size_unknown")
    if obj.kind == "unknown":
        gaps.append("object_undeclared")

    scope = _infer_scope(store, operand, alarm_fn, alarm_line)
    path, extra_callers = _build_path(
        store,
        alarm_fn=alarm_fn,
        alarm_loc=alarm.location,
        alarm_line=alarm_line,
        operand=operand,
        gaps=gaps,
    )

    # Dedup gaps while preserving order.
    seen_g: set[str] = set()
    uniq_gaps: list[str] = []
    for g in gaps:
        if g in seen_g:
            continue
        seen_g.add(g)
        uniq_gaps.append(g)

    all_guards: list[str] = []
    seen_guard: set[str] = set()
    seen_sym: set[str] = set()
    for step in path:
        step.sequence = step.step
        step.declared_size = obj.array_size
        step.datatype = obj.declared_type or ""
        step.first_access = step.symbol not in seen_sym
        seen_sym.add(step.symbol)
        for gtxt in step.guards:
            if gtxt in seen_guard:
                continue
            seen_guard.add(gtxt)
            all_guards.append(gtxt)

    return CaseFile(
        order_id=order_id,
        alarm_type=alarm.type or "",
        alarm_category=alarm.category or "",
        location=alarm.location or "",
        astree_message=alarm.message or "",
        alarm_function=alarm_fn,
        alarm_line=alarm_line,
        alarm_line_text=alarm_text.strip(),
        index_expression=index_expr,
        operand_symbol=operand,
        operand_scope_at_alarm=scope,
        indexed_object=obj,
        helpers=helpers,
        write_helpers=write_helpers,
        path=path,
        gaps=uniq_gaps,
        extra_callers=extra_callers,
        guards=all_guards,
    )