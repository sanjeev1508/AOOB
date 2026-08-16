"""Indexed loaders for Astrée CSVs and the C source used by AOOB tools."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Astree location: file:line.col_start-col_end  (end may be line.col)
_LOC_RE = re.compile(
    r"^(?P<file>[^:]+):(?P<line>\d+)\.(?P<col_start>\d+)"
    r"(?:-(?:(?P<end_line>\d+)\.)?(?P<col_end>\d+))?$"
)

# Rough C function / method opener for enclosing-snippet search.
# Allows GNU __attribute__((...)) between storage-class and return type / name.
_FUNC_START_RE = re.compile(
    r"^(?:(?:static|inline|extern|unsigned|signed|const|volatile)\s+|"
    r"__attribute__\s*\(\([^)]*\)\)\s*)*"
    r"(?:[\w\s\*]+?)\s+(\w+)\s*\([^;]*\)\s*\{?\s*$"
)

# Names too generic for file-wide assignment scans (would cross-contaminate).
_COMMON_INDEX_NAMES = frozenset(
    {
        "i",
        "j",
        "k",
        "n",
        "x",
        "y",
        "p",
        "q",
        "idx",
        "index",
        "Index",
        "tmp",
        "temp",
        "val",
        "value",
        "ret",
        "ptr",
        "buf",
        "len",
        "size",
        "count",
        "cnt",
        "id",
        "Id",
        "ch",
        "ChannelId",
    }
)

# Simple C integer typedef / builtin widths for abstract-domain FP hints.
_INT_TYPE_WIDTH_BITS: dict[str, int] = {
    "uint8": 8,
    "uint8_t": 8,
    "uint8_least": 8,
    "sint8": 8,
    "int8": 8,
    "int8_t": 8,
    "char": 8,
    "unsigned_char": 8,
    "uint16": 16,
    "uint16_t": 16,
    "uint16_least": 16,
    "sint16": 16,
    "int16": 16,
    "int16_t": 16,
    "uint32": 32,
    "uint32_t": 32,
    "uint32_least": 32,
    "sint32": 32,
    "int32": 32,
    "int32_t": 32,
}

_ATTR_STRIP_RE = re.compile(r"__attribute__\s*\(\([^)]*\)\)")
_FUNC_NAME_BLACKLIST = frozenset(
    {
        "if",
        "for",
        "while",
        "switch",
        "return",
        "sizeof",
        "__attribute__",
        "attribute",
        "defined",
    }
)

# Array declarator: Name[size] followed by initializer / terminator.
# Allow `=` with `{` on a later line (common in Astrée preprocessed dumps).
_ARRAY_DECL_RE = re.compile(
    r"\b(?P<name>[A-Za-z_]\w*)\s*\[\s*(?P<size>[^\]]+?)\s*\]"
    r"(?P<more>(?:\s*\[[^\]]*\])*)"
    r"\s*(?:=\s*\{?|;|,|\)|\s*$)"
)

# C integer literal in declarators: 2, 1u, 16UL, (10u), …
_C_INT_LIT_RE = re.compile(r"^(?P<digits>\d+)[uUlL]*$")


def _parse_array_size_token(size_tok: str) -> Optional[int]:
    """Literal size only — no expression evaluation. Strips one () layer."""
    t = (size_tok or "").strip()
    if t.startswith("(") and t.endswith(")"):
        t = t[1:-1].strip()
    m = _C_INT_LIT_RE.fullmatch(t)
    if not m:
        return None
    return int(m.group("digits"))

# Pointer-ish declarator: ... * Name [=|;|,)]
_PTR_DECL_RE = re.compile(
    r"(?P<head>[\w\s\*]+?)\b(?P<name>[A-Za-z_]\w*)\s*[;=,)]"
)


@dataclass(frozen=True)
class AlarmRecord:
    order: int
    type: str
    category: str
    location: str
    classification: str
    comment: str
    message: str

    @property
    def parsed_location(self) -> Optional[dict]:
        m = _LOC_RE.match(self.location.strip())
        if not m:
            return None
        return {
            "file": m.group("file"),
            "line": int(m.group("line")),
            "col_start": int(m.group("col_start")),
            "col_end": int(m.group("col_end") or m.group("col_start")),
            "end_line": int(m.group("end_line") or m.group("line")),
        }


@dataclass(frozen=True)
class DataFlowRecord:
    variable: str
    function: str
    access: str
    process: str
    data_races: str
    shared_variable: str
    byte_offset: str
    location: str
    line: Optional[int] = None


@dataclass(frozen=True)
class DeclarationRecord:
    name: str
    raw_declaration_text: str
    array_size: Optional[int]
    declared_type: str
    is_pointer: bool
    location: str
    parse_note: Optional[str] = None


@dataclass(frozen=True)
class ControlFlowRecord:
    caller: str
    callee: str
    call_site: str
    process: str
    line: Optional[int] = None


@dataclass
class DataStore:
    """In-memory indexes over the four allowed input artifacts."""

    root: Path
    alarms: dict[int, AlarmRecord] = field(default_factory=dict)
    data_flow_by_line: dict[int, list[DataFlowRecord]] = field(
        default_factory=lambda: defaultdict(list)
    )
    data_flow_by_variable: dict[str, list[DataFlowRecord]] = field(
        default_factory=lambda: defaultdict(list)
    )
    control_by_caller: dict[str, list[ControlFlowRecord]] = field(
        default_factory=lambda: defaultdict(list)
    )
    control_by_callee: dict[str, list[ControlFlowRecord]] = field(
        default_factory=lambda: defaultdict(list)
    )
    control_by_line: dict[int, list[ControlFlowRecord]] = field(
        default_factory=lambda: defaultdict(list)
    )
    source_lines: list[str] = field(default_factory=list)
    declarations: dict[str, DeclarationRecord] = field(default_factory=dict)
    # All declaration hits for a name (first wins in ``declarations``).
    declarations_all: dict[str, list[DeclarationRecord]] = field(
        default_factory=lambda: defaultdict(list)
    )
    # Variable → typedef/struct type name (from ``Type Name;``).
    object_types: dict[str, str] = field(default_factory=dict)
    # Typedef/struct type → array member fields (from typedef body).
    type_array_fields: dict[str, list[DeclarationRecord]] = field(
        default_factory=lambda: defaultdict(list)
    )

    @classmethod
    def load(cls, data_dir: Path | str) -> "DataStore":
        data_dir = Path(data_dir)
        store = cls(root=data_dir)
        store._load_alarms(data_dir / "Full_alarms.csv")
        store._load_data_flow(data_dir / "data flow.csv")
        store._load_control_flow(data_dir / "control flow.csv")
        store._load_source(data_dir / "input.c")
        store._build_declaration_index()
        return store

    def _read_semicolon_csv(self, path: Path) -> list[dict[str, str]]:
        text = path.read_text(encoding="utf-8", errors="replace")
        # Drop Astrée "sep=;" preamble / section titles before the real header.
        lines = text.splitlines()
        header_idx = None
        for i, line in enumerate(lines):
            if ";" in line and not line.lower().startswith("sep="):
                # Prefer a line that looks like a CSV header (letters + ;).
                if re.search(r"[A-Za-z]", line.split(";")[0]):
                    header_idx = i
                    # For control-flow report, skip until Caller;Callee...
                    if "Caller" in line or "Variable" in line or "Order" in line:
                        break
        if header_idx is None:
            raise ValueError(f"No CSV header found in {path}")
        payload = "\n".join(lines[header_idx:])
        reader = csv.DictReader(payload.splitlines(), delimiter=";")
        return [dict(row) for row in reader if any((v or "").strip() for v in row.values())]

    @staticmethod
    def parse_location_line(location: str) -> Optional[int]:
        m = _LOC_RE.match((location or "").strip())
        return int(m.group("line")) if m else None

    @staticmethod
    def _parse_order_id(raw: str) -> Optional[int]:
        """Parse Order values that may use thousand separators (e.g. ``1,112``)."""
        text = (raw or "").strip().replace(" ", "").replace(",", "")
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            return None

    def _load_alarms(self, path: Path) -> None:
        for row in self._read_semicolon_csv(path):
            order = self._parse_order_id(str(row.get("Order", "")))
            if order is None:
                continue
            self.alarms[order] = AlarmRecord(
                order=order,
                type=(row.get("Type") or "").strip(),
                category=(row.get("Category") or "").strip(),
                location=(row.get("Location") or "").strip(),
                classification=(row.get("Classification") or "").strip(),
                comment=(row.get("Comment") or "").strip(),
                message=(row.get("Message") or "").strip(),
            )

    def _load_data_flow(self, path: Path) -> None:
        for row in self._read_semicolon_csv(path):
            loc = (row.get("Location") or "").strip()
            line = self.parse_location_line(loc)
            rec = DataFlowRecord(
                variable=(row.get("Variable") or "").strip(),
                function=(row.get("Function") or "").strip(),
                access=(row.get("Access") or "").strip(),
                process=(row.get("Process") or "").strip(),
                data_races=(row.get("Data races") or "").strip(),
                shared_variable=(row.get("Shared variable") or "").strip(),
                byte_offset=(row.get("Byte offset") or "").strip(),
                location=loc,
                line=line,
            )
            if line is not None:
                self.data_flow_by_line[line].append(rec)
            if rec.variable:
                # Index bare name (before @scope) and full string.
                bare = rec.variable.split("@", 1)[0].strip('"')
                self.data_flow_by_variable[bare].append(rec)
                self.data_flow_by_variable[rec.variable].append(rec)

    def _load_control_flow(self, path: Path) -> None:
        for row in self._read_semicolon_csv(path):
            site = (row.get("Call site") or "").strip()
            line = self.parse_location_line(site)
            rec = ControlFlowRecord(
                caller=(row.get("Caller") or "").strip(),
                callee=(row.get("Callee") or "").strip(),
                call_site=site,
                process=(row.get("Process") or "").strip(),
                line=line,
            )
            if rec.caller:
                self.control_by_caller[rec.caller].append(rec)
            if rec.callee:
                self.control_by_callee[rec.callee].append(rec)
            if line is not None:
                self.control_by_line[line].append(rec)

    def _load_source(self, path: Path) -> None:
        # Keep a leading empty slot so line numbers are 1-based.
        self.source_lines = [""] + path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()

    def get_alarm(self, order_id: int) -> Optional[AlarmRecord]:
        return self.alarms.get(order_id)

    def get_source_slice(self, start: int, end: int) -> list[tuple[int, str]]:
        start = max(1, start)
        end = min(len(self.source_lines) - 1, end)
        return [(i, self.source_lines[i]) for i in range(start, end + 1)]

    @staticmethod
    def _split_c_args(arg_text: str) -> list[str]:
        """Split a comma-separated C argument list respecting nested () []."""
        parts: list[str] = []
        cur: list[str] = []
        depth = 0
        for ch in arg_text:
            if ch in "([":
                depth += 1
                cur.append(ch)
            elif ch in ")]":
                depth = max(0, depth - 1)
                cur.append(ch)
            elif ch == "," and depth == 0:
                parts.append("".join(cur).strip())
                cur = []
            else:
                cur.append(ch)
        if cur or parts:
            parts.append("".join(cur).strip())
        return [p for p in parts if p]

    def _window_text(self, start_line: int, end_line: int) -> str:
        start_line = max(1, start_line)
        end_line = min(len(self.source_lines) - 1, end_line)
        return "\n".join(
            self.source_lines[i] for i in range(start_line, end_line + 1)
        )

    def parse_function_parameters(
        self, function_name: str, near_line: Optional[int] = None
    ) -> list[dict]:
        """Parse parameter names/types from a function header (structural)."""
        fn = (function_name or "").strip()
        if not fn:
            return []
        # Prefer definition near ``near_line``; else any header match.
        candidates: list[int] = []
        if near_line is not None:
            for ln in range(max(1, near_line - 3), min(len(self.source_lines), near_line + 2)):
                if self._looks_like_function_header(ln) == fn:
                    candidates.append(ln)
                    break
        if not candidates:
            for ln in range(1, len(self.source_lines)):
                if self._looks_like_function_header(ln) == fn:
                    candidates.append(ln)
                    if len(candidates) >= 3:
                        break
        if not candidates:
            return []

        start = candidates[0]
        blob = self._window_text(start, start + 25)
        blob = _ATTR_STRIP_RE.sub(" ", blob)
        m = re.search(rf"\b{re.escape(fn)}\s*\(", blob)
        if not m:
            return []
        i = m.end() - 1  # at '('
        depth = 0
        end = None
        for j in range(i, len(blob)):
            if blob[j] == "(":
                depth += 1
            elif blob[j] == ")":
                depth -= 1
                if depth == 0:
                    end = j
                    break
        if end is None:
            return []
        raw_args = blob[i + 1 : end]
        params: list[dict] = []
        for idx, piece in enumerate(self._split_c_args(raw_args)):
            piece_n = re.sub(r"\s+", " ", piece).strip()
            if not piece_n or piece_n == "void":
                continue
            # Drop trailing array brackets on the name: Type name[]
            piece_n = re.sub(r"\[\s*\]", "", piece_n)
            # Name is rightmost identifier (handle pointers).
            nm = re.search(r"([A-Za-z_]\w*)\s*$", piece_n)
            if not nm:
                continue
            name = nm.group(1)
            typ = piece_n[: nm.start()].strip().rstrip("*").strip()
            typ = re.sub(r"\s*\*+\s*$", "", typ).strip()
            params.append(
                {
                    "index": idx,
                    "name": name,
                    "declared_type": typ or None,
                    "raw": piece_n[:160],
                }
            )
        return params

    def resolve_typedef_width(self, type_name: str, *, max_hops: int = 5) -> dict:
        """Follow simple ``typedef Underlying Name;`` chains for width hints."""
        key = (type_name or "").strip()
        if not key:
            return {"name": key, "found": False}
        chain = [key]
        cur = key
        for _ in range(max(1, max_hops)):
            width = _INT_TYPE_WIDTH_BITS.get(cur) or _INT_TYPE_WIDTH_BITS.get(
                cur.replace(" ", "_")
            )
            if width is not None:
                return {
                    "name": key,
                    "found": True,
                    "underlying": cur,
                    "width_bits": width,
                    "abstract_max": (1 << width) - 1,
                    "typedef_chain": chain,
                    "note": (
                        f"Type width {width} bits => abstract domain often reports "
                        f"[0, {(1 << width) - 1}]. That upper bound is the type "
                        "domain, not proof a runtime index reaches it."
                    ),
                }
            # Scan for typedef Underlying Name;
            hit = None
            pat = re.compile(
                rf"^typedef\s+(.+?)\s+{re.escape(cur)}\s*;\s*$"
            )
            for ln in range(1, len(self.source_lines)):
                text = self.source_lines[ln].strip()
                m = pat.match(text)
                if m:
                    hit = re.sub(r"\s+", " ", m.group(1)).strip()
                    # Drop storage / qualifiers noise
                    hit = re.sub(
                        r"^(?:const|volatile|unsigned|signed)\s+",
                        "",
                        hit,
                    ).strip()
                    # Keep last token if multi-word custom type left
                    if " " in hit and hit.split()[-1] not in _INT_TYPE_WIDTH_BITS:
                        # e.g. "unsigned char" → try full then last
                        joined = hit.replace(" ", "_")
                        if joined in _INT_TYPE_WIDTH_BITS:
                            hit = joined
                        elif hit.startswith("unsigned char"):
                            hit = "unsigned_char"
                        else:
                            hit = hit.split()[-1]
                    break
            if not hit or hit == cur:
                break
            chain.append(hit)
            cur = hit
        return {
            "name": key,
            "found": False,
            "typedef_chain": chain,
            "parse_note": "Could not resolve to a known fixed-width integer type.",
        }

    def extract_call_argument_expression(
        self, callee: str, call_line: int, arg_index: int
    ) -> Optional[dict]:
        """Extract the ``arg_index``-th argument expression at a call near ``call_line``."""
        fn = (callee or "").strip()
        if not fn or arg_index < 0:
            return None
        # Search a small window — calls may wrap across lines.
        for start in range(max(1, call_line - 2), call_line + 1):
            blob = self._window_text(start, start + 12)
            # Find fn( ... ) with matching parens (last occurrence near call_line).
            for m in re.finditer(rf"\b{re.escape(fn)}\s*\(", blob):
                i = m.end() - 1
                depth = 0
                end = None
                for j in range(i, len(blob)):
                    ch = blob[j]
                    if ch == "(":
                        depth += 1
                    elif ch == ")":
                        depth -= 1
                        if depth == 0:
                            end = j
                            break
                if end is None:
                    continue
                args = self._split_c_args(blob[i + 1 : end])
                if arg_index >= len(args):
                    continue
                expr = re.sub(r"\s+", " ", args[arg_index]).strip()
                idents = re.findall(r"\b([A-Za-z_]\w*)\b", expr)
                # Drop the callee name if it appeared (shouldn't in arg).
                idents = [x for x in idents if x != fn]
                return {
                    "argument_expression": expr[:240],
                    "argument_identifiers": idents[:12],
                    "arg_index": arg_index,
                    "call_text": re.sub(r"\s+", " ", blob[m.start() : end + 1])[:280],
                    "approx_line": start,
                }
        return None

    def find_call_site_arguments(
        self,
        callee_function: str,
        parameter_name: str,
        *,
        from_location: Optional[str] = None,
        max_sites: int = 12,
    ) -> dict:
        """CF callers → argument expressions passed for ``parameter_name``.

        Retrieval only. Use when a sliced symbol is a function parameter with
        no local writes — the concrete index origin is the caller argument.
        """
        callee = (callee_function or "").strip()
        param = (parameter_name or "").strip().split("@", 1)[0].strip('"')
        param_tail = param.split(".")[-1]
        near = self.parse_location_line(from_location) if from_location else None
        params = self.parse_function_parameters(callee, near_line=near)
        arg_index = None
        param_type = None
        for p in params:
            if p["name"] == param or p["name"] == param_tail:
                arg_index = int(p["index"])
                param_type = p.get("declared_type")
                break
        if arg_index is None and params:
            # Fallback: single-parameter function.
            if len(params) == 1:
                arg_index = int(params[0]["index"])
                param_type = params[0].get("declared_type")
                param_tail = params[0]["name"]

        width_info = (
            self.resolve_typedef_width(param_type) if param_type else {"found": False}
        )

        sites: list[dict] = []
        edges = list(self.control_by_callee.get(callee, []))
        seen_loc: set[tuple] = set()
        for edge in edges:
            if edge.line is None:
                continue
            key = (edge.caller, edge.line)
            if key in seen_loc:
                continue
            seen_loc.add(key)
            extracted = None
            if arg_index is not None:
                extracted = self.extract_call_argument_expression(
                    callee, edge.line, arg_index
                )
            site = {
                "caller": edge.caller,
                "call_site": edge.call_site,
                "line": edge.line,
                "location": (
                    edge.call_site
                    or f"ALL_bc_with_context.c:{edge.line}.1-1"
                ),
            }
            if extracted:
                site.update(extracted)
            sites.append(site)
            if len(sites) >= max_sites:
                break

        # Collect unique next-slice candidates from argument identifiers.
        next_syms: list[str] = []
        seen_s: set[str] = set()
        for s in sites:
            for ident in s.get("argument_identifiers") or []:
                if ident in _COMMON_INDEX_NAMES and ident == param_tail:
                    continue
                if ident in seen_s:
                    continue
                # Skip type-like tokens.
                if ident.endswith(("Type", "Idx", "IterType")) and ident[0].isupper():
                    if ident not in {param_tail}:
                        # still allow ethBufDataIdx-style names ending in Idx
                        if not ident[0].islower() and "Idx" in ident and len(ident) < 8:
                            continue
                seen_s.add(ident)
                next_syms.append(ident)

        return {
            "callee": callee,
            "parameter_name": param_tail,
            "parameter_index": arg_index,
            "parameter_type": param_type,
            "parameter_type_width": width_info if width_info.get("found") else None,
            "parameters_parsed": params,
            "call_sites": sites,
            "next_slice_candidates": next_syms[:16],
            "truncated": len(edges) > len(sites),
            "note": (
                "Raw retrieval of caller argument expressions for a callee "
                "parameter. Empty local writes on a parameter are expected — "
                "continue investigation on next_slice_candidates / "
                "argument_identifiers (not the parameter name alone). "
                "parameter_type_width.abstract_max matching an Astrée [0, hi] "
                "often means type-domain over-approx, not a proven runtime index."
            ),
        }

    def _function_end_from(self, func_line: int, max_span: int = 1200) -> int:
        """Brace-match from a candidate function start line; return end line."""
        n = len(self.source_lines) - 1
        depth = 0
        started = False
        end_line = min(n, func_line + max_span)
        for i in range(func_line, min(n, func_line + max_span) + 1):
            for ch in self.source_lines[i]:
                if ch == "{":
                    depth += 1
                    started = True
                elif ch == "}":
                    depth -= 1
                    if started and depth == 0:
                        return i
        return end_line

    def _looks_like_function_header(self, line_no: int) -> Optional[str]:
        """Return function name if ``line_no`` starts a (possibly multi-line) header."""
        if line_no < 1 or line_no >= len(self.source_lines):
            return None
        text = self.source_lines[line_no].strip()
        if not text or text.startswith(
            ("if", "for", "while", "switch", "else", "case", "#", "}")
        ):
            return None

        # Strip GNU attributes so they are not mistaken for the function name.
        cleaned = _ATTR_STRIP_RE.sub(" ", text)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        m = _FUNC_START_RE.match(cleaned)
        if m and m.group(1) not in _FUNC_NAME_BLACKLIST:
            return m.group(1)

        # Same-line fallback: last identifier before the final parameter list.
        # Avoid matching __attribute__ / casts by taking the rightmost \w+(...).
        if "(" in cleaned:
            candidates = list(
                re.finditer(r"\b([A-Za-z_]\w*)\s*\([^;]*\)\s*\{?\s*$", cleaned)
            )
            if not candidates:
                # Truncated long lines: name( still present without closing ')'.
                candidates = list(
                    re.finditer(r"\b([A-Za-z_]\w*)\s*\([^;]*$", cleaned)
                )
            for cm in reversed(candidates):
                name = cm.group(1)
                if name in _FUNC_NAME_BLACKLIST:
                    continue
                # Skip obvious type tokens when a better name follows earlier —
                # already iterating rightmost-first.
                if name.endswith(("Type", "type", "Idx", "IterType")) and len(
                    candidates
                ) > 1:
                    # Likely return type fragment; keep looking left only if
                    # this is the only candidate.
                    continue
                return name

        # Multi-line: Name\n( ... )\n{
        if re.fullmatch(r"[A-Za-z_]\w*", cleaned) or re.fullmatch(
            r"(?:static\s+|inline\s+|extern\s+)*[\w\s\*]+?\b([A-Za-z_]\w*)\s*",
            cleaned,
        ):
            m2 = re.search(r"\b([A-Za-z_]\w*)\s*$", cleaned)
            if not m2:
                return None
            name = m2.group(1)
            if name in _FUNC_NAME_BLACKLIST:
                return None
            for j in range(line_no + 1, min(line_no + 12, len(self.source_lines))):
                nxt = self.source_lines[j].strip()
                if not nxt:
                    continue
                if nxt.startswith("(") or nxt == "(":
                    depth = 0
                    saw_paren_close = False
                    for k in range(j, min(j + 40, len(self.source_lines))):
                        for ch in self.source_lines[k]:
                            if ch == "(":
                                depth += 1
                            elif ch == ")":
                                depth -= 1
                                if depth == 0:
                                    saw_paren_close = True
                            elif ch == "{" and saw_paren_close and depth == 0:
                                return name
                            elif ch == ";" and saw_paren_close and depth == 0:
                                return None
                    return None
                break
        return None

    def find_enclosing_function(
        self, line: int, max_lookback: int = 800
    ) -> tuple[Optional[str], int, int]:
        """Return (name, start_line, end_line) that actually contains ``line``."""
        n = len(self.source_lines) - 1
        start = max(1, line - max_lookback)
        for i in range(line, start - 1, -1):
            name = self._looks_like_function_header(i)
            if not name:
                continue
            end_line = self._function_end_from(i)
            if i <= line <= end_line:
                return name, i, end_line
        return None, max(1, line - 20), min(n, line + 20)

    def _build_declaration_index(self) -> None:
        """Scan ``input.c`` for array/pointer declarations (structural only)."""
        skip_prefixes = (
            "if",
            "for",
            "while",
            "switch",
            "return",
            "else",
            "case",
            "goto",
            "#",
        )
        for line_no in range(1, len(self.source_lines)):
            raw = self.source_lines[line_no]
            text = raw.strip()
            if not text or text.startswith(skip_prefixes):
                continue
            if text.startswith(("}", "{")):
                continue

            for m in _ARRAY_DECL_RE.finditer(text):
                name = m.group("name")
                size_tok = (m.group("size") or "").strip()
                more = m.group("more") or ""
                left = text[: m.start("name")].strip()
                # Skip obvious expression indexing (ptr->field[i], call(x)[i], …)
                # but keep typedef / custom type declarators (e.g. Foo_tst Name[1u]).
                if not left:
                    continue
                if left.endswith((".", "->", "(", "=", "+", "-", "/", "%", "&", ",")):
                    continue
                # Call-site / arg false matches: Func(&Name[0], …) or Foo(Name[i])
                if "(" in left and not re.search(r"\(\s*\*", left):
                    continue
                if not left:
                    continue
                if left.endswith("*") and not any(
                    k in left
                    for k in (
                        "uint",
                        "sint",
                        "int",
                        "char",
                        "bool",
                        "void",
                        "const",
                        "static",
                        "extern",
                        "struct",
                        "typedef",
                    )
                ):
                    # Pointer cast / deref expression, not a declaration.
                    if "(" in left:
                        continue

                array_size: Optional[int] = None
                parse_note: Optional[str] = None
                if more.strip():
                    parse_note = (
                        "Multi-dimensional declarator; only first dimension token "
                        f"captured ({size_tok!r}); further dimensions not resolved."
                    )
                lit_size = _parse_array_size_token(size_tok)
                if lit_size is not None:
                    array_size = lit_size
                    if array_size == 0:
                        parse_note = (
                            f"{parse_note} Literal size token {size_tok!r} parsed as 0 "
                            "— unusual for a real array declarator; treat with caution."
                        ).strip()
                else:
                    note = (
                        f"Non-literal array size token {size_tok!r}; "
                        "size not resolved to an integer."
                    )
                    parse_note = f"{parse_note} {note}".strip() if parse_note else note

                is_pointer = "*" in left
                declared_type = re.sub(r"\s+", " ", left).strip() or "unknown"
                if array_size is not None and not parse_note:
                    declared_type = f"{declared_type}[{array_size}]".strip()
                elif size_tok:
                    declared_type = f"{declared_type}[{size_tok}]".strip()

                loc = f"ALL_bc_with_context.c:{line_no}.1-{max(1, len(raw))}"
                rec = DeclarationRecord(
                    name=name,
                    raw_declaration_text=text[:240],
                    array_size=array_size,
                    declared_type=declared_type,
                    is_pointer=is_pointer,
                    location=loc,
                    parse_note=parse_note,
                )
                self.declarations_all[name].append(rec)
                existing = self.declarations.get(name)
                if existing is None:
                    self.declarations[name] = rec
                else:
                    # Prefer a positive literal size over missing/zero/ambiguous.
                    ex = existing.array_size
                    nw = rec.array_size
                    if (ex is None or ex == 0) and nw is not None and nw > 0:
                        self.declarations[name] = rec
                    elif ex is None and nw is not None:
                        self.declarations[name] = rec
                    elif (
                        existing.parse_note
                        and not rec.parse_note
                        and nw is not None
                    ):
                        self.declarations[name] = rec

            if "*" in text and "[" not in text:
                pm = _PTR_DECL_RE.search(text)
                if not pm:
                    continue
                name = pm.group("name")
                if name in self.declarations:
                    continue
                head = (pm.group("head") or "").strip()
                if not head or "*" not in head:
                    continue
                if not any(
                    k in head
                    for k in (
                        "uint",
                        "sint",
                        "int",
                        "char",
                        "bool",
                        "void",
                        "const",
                        "static",
                        "extern",
                        "struct",
                        "typedef",
                    )
                ):
                    continue
                loc = f"ALL_bc_with_context.c:{line_no}.1-{max(1, len(raw))}"
                rec = DeclarationRecord(
                    name=name,
                    raw_declaration_text=text[:240],
                    array_size=None,
                    declared_type=re.sub(r"\s+", " ", head).strip(),
                    is_pointer=True,
                    location=loc,
                    parse_note=None,
                )
                self.declarations_all[name].append(rec)
                self.declarations[name] = rec

        # Typedef struct/union member arrays + ``Type Name;`` object defs.
        self._index_typedef_and_objects()

    def _index_typedef_and_objects(self) -> None:
        """Index ``typedef {{ ... Field[N]; }} Type;`` and ``Type Var;`` (structural)."""
        typedef_start: Optional[int] = None
        pending_fields: list[tuple[str, Optional[int], str, int, str]] = []
        # field name, size, left type text, line, raw

        for line_no in range(1, len(self.source_lines)):
            text = self.source_lines[line_no].strip()
            if not text:
                continue

            if re.match(r"^typedef\s+(?:struct|union)\b", text):
                typedef_start = line_no
                pending_fields = []
                continue

            if typedef_start is not None:
                # Member array: Type field[N];
                for m in _ARRAY_DECL_RE.finditer(text):
                    fname = m.group("name")
                    size_tok = (m.group("size") or "").strip()
                    left = text[: m.start("name")].strip()
                    if not left or left.endswith((".", "->", "(", "=", "&")):
                        continue
                    if "(" in left and not re.search(r"\(\s*\*", left):
                        continue
                    size = _parse_array_size_token(size_tok)
                    pending_fields.append(
                        (fname, size, left, line_no, text[:240])
                    )

                # Closing: } TypeName;
                cm = re.match(r"^\}\s*([A-Za-z_]\w*)\s*;\s*$", text)
                if cm:
                    type_name = cm.group(1)
                    for fname, size, left, fln, raw in pending_fields:
                        left_n = re.sub(r"\s+", " ", left).strip()
                        rec = DeclarationRecord(
                            name=f"{type_name}.{fname}",
                            raw_declaration_text=raw,
                            array_size=size,
                            declared_type=(
                                f"{left_n}[{size}]"
                                if size is not None
                                else f"{left_n}[?]"
                            ),
                            is_pointer="*" in left,
                            location=(
                                f"ALL_bc_with_context.c:{fln}.1-"
                                f"{max(1, len(self.source_lines[fln]))}"
                            ),
                            parse_note=(
                                None
                                if size is not None
                                else f"Non-literal member size in {type_name}.{fname}"
                            ),
                        )
                        self.type_array_fields[type_name].append(rec)
                        # Also index Type.field and bare field if unambiguous later.
                        self.declarations_all[rec.name].append(rec)
                        if rec.name not in self.declarations or (
                            (self.declarations[rec.name].array_size or 0) == 0
                            and size
                        ):
                            self.declarations[rec.name] = rec
                    typedef_start = None
                    pending_fields = []
                    continue
                # Abandoned typedef if we hit another top-level typedef without close
                if text.startswith("typedef ") and "{" not in text:
                    typedef_start = None
                    pending_fields = []

            # Object definition: TypeName VarName;
            om = re.match(
                r"^(?:static\s+|extern\s+|const\s+|volatile\s+)*"
                r"([A-Za-z_]\w*)\s+([A-Za-z_]\w*)\s*;\s*$",
                text,
            )
            if om:
                typ, var = om.group(1), om.group(2)
                if typ in {
                    "return",
                    "sizeof",
                    "if",
                    "for",
                    "while",
                    "switch",
                    "goto",
                }:
                    continue
                if var not in self.object_types:
                    self.object_types[var] = typ
                # If Type has array fields, expose Var.field declarations.
                for frec in self.type_array_fields.get(typ, []):
                    field = frec.name.split(".")[-1]
                    qname = f"{var}.{field}"
                    qrec = DeclarationRecord(
                        name=qname,
                        raw_declaration_text=frec.raw_declaration_text,
                        array_size=frec.array_size,
                        declared_type=frec.declared_type,
                        is_pointer=frec.is_pointer,
                        location=frec.location,
                        parse_note=(
                            f"Array member of {typ} object {var}; "
                            f"from typedef field {frec.name}."
                        ),
                    )
                    self.declarations_all[qname].append(qrec)
                    self.declarations[qname] = qrec

    def get_declaration(self, symbol_name: str) -> Optional[DeclarationRecord]:
        key = (symbol_name or "").strip().split("@", 1)[0].strip('"')
        hit = self.declarations.get(key)
        if hit is not None:
            return hit
        # Struct.Field → try Type.field via object type, then bare field.
        if "." in key:
            obj, field = key.split(".", 1)
            typ = self.object_types.get(obj)
            if typ:
                typed = self.declarations.get(f"{typ}.{field}")
                if typed is not None:
                    return typed
            return self.declarations.get(field)
        return None

    def resolve_declaration_lookup(self, symbol_name: str) -> dict:
        """Rich declaration lookup for tools (arrays, objects, nested fields)."""
        key = (symbol_name or "").strip().split("@", 1)[0].strip('"')
        rec = self.get_declaration(key)
        if rec is not None and rec.array_size is not None:
            return {
                "symbol": key,
                "found": True,
                "kind": "array",
                "declaration": rec,
                "nested_array_fields": [],
            }

        # Object with nested array fields (union/struct instance).
        typ = self.object_types.get(key)
        if typ:
            fields = list(self.type_array_fields.get(typ, []))
            # Also pick up Var.field entries already indexed.
            for name, drec in self.declarations.items():
                if name.startswith(key + ".") and drec.array_size is not None:
                    if not any(f.name == name or f.name.endswith("." + name.split(".")[-1]) for f in fields):
                        fields.append(drec)
            return {
                "symbol": key,
                "found": True,
                "kind": "object",
                "object_type": typ,
                "declaration": DeclarationRecord(
                    name=key,
                    raw_declaration_text=f"{typ} {key};",
                    array_size=None,
                    declared_type=typ,
                    is_pointer=False,
                    location="",
                    parse_note=(
                        f"{key} is a {typ} object (not itself an array). "
                        "Indexed accesses usually target a member such as "
                        f"{key}.raw — see nested_array_fields."
                    ),
                ),
                "nested_array_fields": fields,
            }

        if rec is not None:
            return {
                "symbol": key,
                "found": True,
                "kind": "other",
                "declaration": rec,
                "nested_array_fields": [],
            }

        # Qualified path not found as array — try object.field suggestions.
        nested: list[DeclarationRecord] = []
        if "." not in key and key in self.object_types:
            nested = list(self.type_array_fields.get(self.object_types[key], []))
        return {
            "symbol": key,
            "found": False,
            "kind": None,
            "declaration": None,
            "nested_array_fields": nested,
        }

    @staticmethod
    def _symbol_name_forms(variable_name: str) -> dict[str, Optional[str]]:
        """Bare / qualified / parent forms for DF + source matching."""
        full = (variable_name or "").strip().split("@", 1)[0].strip('"')
        tail = full.split(".")[-1] if full else ""
        parent = full.rsplit(".", 1)[0] if "." in full else None
        return {"full": full, "tail": tail, "parent": parent}

    def declaration_name_suggestions(self, symbol_name: str, limit: int = 8) -> list[str]:
        """Structural near-matches for a missing declaration name (retrieval aid)."""
        key = (symbol_name or "").strip().split("@", 1)[0].strip('"')
        if len(key) < 6:
            return []
        key_l = key.lower()
        scored: list[tuple[int, str]] = []
        for name in self.declarations:
            n_l = name.lower()
            if key_l == n_l:
                continue
            score = 0
            if len(key_l) >= 8 and (key_l in n_l or n_l in key_l):
                score = 3
            else:
                pref = 0
                for a, b in zip(key_l, n_l):
                    if a != b:
                        break
                    pref += 1
                if pref >= max(10, len(key_l) // 2):
                    score = 1
            if score:
                scored.append((score, name))
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [n for _, n in scored[:limit]]

    def get_prior_writes(
        self,
        variable_name: str,
        before_location: str,
        *,
        function_scope: Optional[str] = None,
    ) -> list[DataFlowRecord]:
        """Return DF write events to ``variable_name`` with line strictly before
        ``before_location``. Optionally restrict to one function name.

        Matches bare name and qualified ``name@scope`` DF keys. For
        ``Struct.Field`` queries, also considers parent-struct DF writes whose
        assignment text mentions that field (Astrée DF often keys only the
        aggregate). Structural filter only — no reachability judgment beyond
        line order.
        """
        forms = self._symbol_name_forms(variable_name)
        bare = forms["full"] or ""
        before_line = self.parse_location_line(before_location)
        if before_line is None:
            return []
        out: list[DataFlowRecord] = []
        seen: set[tuple] = set()

        def _add_from_keys(keys: set[str], *, require_field_in_text: Optional[str] = None) -> None:
            for key in keys:
                for rec in self.data_flow_by_variable.get(key, []):
                    if (rec.access or "").lower() != "write":
                        continue
                    if rec.line is None or rec.line >= before_line:
                        continue
                    if function_scope and rec.function != function_scope:
                        scope = (
                            rec.variable.split("@", 1)[1]
                            if "@" in rec.variable
                            else ""
                        )
                        if scope != function_scope:
                            continue
                    if require_field_in_text:
                        expr = (
                            self.extract_assignment_text(rec.line, require_field_in_text)
                            if rec.line
                            else ""
                        )
                        # Also accept lines that mention Struct.Field / ->Field.
                        line_txt = (
                            self.source_lines[rec.line]
                            if rec.line and rec.line < len(self.source_lines)
                            else ""
                        )
                        field = require_field_in_text
                        if not (
                            f".{field}" in line_txt
                            or f"->{field}" in line_txt
                            or re.search(rf"\b{re.escape(field)}\b", line_txt)
                        ):
                            continue
                        if not (
                            "=" in line_txt
                            or f"{field}++" in line_txt
                            or f"{field}--" in line_txt
                            or f".{field}++" in line_txt
                            or f"->{field}++" in line_txt
                        ):
                            # Parent aggregate write unrelated to this field.
                            if field not in (expr or line_txt):
                                continue
                    sig = (rec.variable, rec.function, rec.location, rec.line, rec.process)
                    if sig in seen:
                        continue
                    seen.add(sig)
                    out.append(rec)

        # Query bare + any qualified keys that share the bare prefix.
        keys = {bare}
        for k in self.data_flow_by_variable:
            if k == bare or k.startswith(bare + "@"):
                keys.add(k)
        # Also try field tail alone.
        tail = forms["tail"]
        if tail and tail != bare:
            keys.add(tail)
            for k in self.data_flow_by_variable:
                if k == tail or k.startswith(tail + "@"):
                    keys.add(k)
        _add_from_keys(keys)

        # Parent aggregate DF rows filtered to this field (common Astrée pattern).
        parent = forms["parent"]
        if parent and tail:
            parent_keys = {parent}
            for k in self.data_flow_by_variable:
                if k == parent or k.startswith(parent + "@"):
                    parent_keys.add(k)
            _add_from_keys(parent_keys, require_field_in_text=tail)

        out.sort(key=lambda r: (r.line or 0, r.function, r.location))
        return out

    def df_lookup_stats(self, variable_name: str, before_location: str) -> dict:
        """Explain how a variable name resolved in the DF index (retrieval only)."""
        bare = (variable_name or "").strip().split("@", 1)[0].strip('"')
        before_line = self.parse_location_line(before_location)
        keys = sorted(
            {k for k in self.data_flow_by_variable if k == bare or k.startswith(bare + "@")}
        )
        records: list[DataFlowRecord] = []
        seen: set[tuple] = set()
        for key in keys:
            for rec in self.data_flow_by_variable.get(key, []):
                sig = (rec.variable, rec.function, rec.location, rec.line, rec.process, rec.access)
                if sig in seen:
                    continue
                seen.add(sig)
                records.append(rec)
        writes = [r for r in records if (r.access or "").lower() == "write"]
        before = [
            r for r in writes if before_line is not None and r.line is not None and r.line < before_line
        ]
        after = [
            r for r in writes if before_line is not None and r.line is not None and r.line >= before_line
        ]
        return {
            "queried_name": bare,
            "df_keys_matched": keys,
            "df_record_count": len(records),
            "df_write_count": len(writes),
            "writes_before_location": len(before),
            "writes_at_or_after_location": len(after),
            "sample_writes_at_or_after": [
                {
                    "variable": r.variable,
                    "function": r.function,
                    "line": r.line,
                    "location": r.location,
                }
                for r in after[:5]
            ],
        }

    def extract_assignment_text(self, line: int, variable_name: str) -> str:
        """Return source text mentioning an assignment involving ``variable_name``."""
        if line < 1 or line >= len(self.source_lines):
            return ""
        bare = variable_name.split("@", 1)[0].strip('"')
        for ln in [line, line - 1, line + 1, line - 2, line + 2]:
            if ln < 1 or ln >= len(self.source_lines):
                continue
            text = self.source_lines[ln].strip()
            if bare in text and (
                "=" in text or f"{bare}++" in text or f"{bare}--" in text
            ):
                return text[:300]
        return self.source_lines[line].strip()[:300]

    def find_source_assignments(
        self,
        variable_name: str,
        *,
        before_line: int,
        function_start: int,
        function_end: int,
    ) -> list[dict]:
        """Scan source in a function span for assignments/initializers to a name.

        Retrieval only — returns matching lines before ``before_line``.
        Supports bare names and ``Struct.Field`` / ``Struct->Field`` forms.
        """
        forms = self._symbol_name_forms(variable_name)
        full = forms["full"] or ""
        tail = forms["tail"] or full
        targets = []
        if full:
            targets.append(full)
        if tail and tail not in targets:
            targets.append(tail)
        patterns: list[re.Pattern[str]] = []
        for name in targets:
            esc = re.escape(name)
            patterns.extend(
                [
                    re.compile(rf"(?:->|\.){re.escape(tail)}\s*=(?!=)")
                    if name == full and "." in full
                    else re.compile(rf"\b{esc}\s*=(?!=)"),
                    re.compile(rf"(?:->|\.){re.escape(tail)}\s*\+\+"),
                    re.compile(rf"(?:->|\.){re.escape(tail)}\s*--"),
                    re.compile(rf"\b{esc}\s*\+\+"),
                    re.compile(rf"\b{esc}\s*--"),
                    re.compile(rf"\b{esc}\s*[+\-*/&|^%]?="),
                ]
            )
            if "." in name:
                # Explicit Struct.Field / Struct->Field
                root, field = name.rsplit(".", 1)
                patterns.append(
                    re.compile(
                        rf"\b{re.escape(root)}\s*(?:\.|->)\s*{re.escape(field)}\s*=(?!=)"
                    )
                )
                patterns.append(
                    re.compile(
                        rf"\b{re.escape(root)}\s*(?:\.|->)\s*{re.escape(field)}\s*\+\+"
                    )
                )
                patterns.append(
                    re.compile(
                        rf"\b{re.escape(root)}\s*(?:\.|->)\s*{re.escape(field)}\s*--"
                    )
                )

        hits: list[dict] = []
        lo = max(1, function_start)
        hi = min(before_line - 1, function_end, len(self.source_lines) - 1)
        for ln in range(lo, hi + 1):
            text = self.source_lines[ln].strip()
            if not text:
                continue
            if tail not in text and full not in text:
                continue
            if any(p.search(text) for p in patterns):
                hits.append(
                    {
                        "function": None,  # filled by caller
                        "location": f"ALL_bc_with_context.c:{ln}.1-{max(1, len(self.source_lines[ln]))}",
                        "line": ln,
                        "process": None,
                        "assigned_expression_text": text[:300],
                        "call_depth": 0,
                        "source": "source_scan",
                    }
                )
        return hits

    def find_source_assignments_global(
        self,
        variable_name: str,
        *,
        before_line: int,
        max_hits: int = 40,
    ) -> list[dict]:
        """File-wide source assignment scan when DF has no field-level keys.

        Retrieval only — capped. Prefer path-local ``find_source_assignments``
        when a function span is known. Skipped for generic names like ``Index``
        / ``i`` that would cross-contaminate unrelated functions.
        """
        forms = self._symbol_name_forms(variable_name)
        full = forms["full"] or ""
        tail = forms["tail"] or full
        if not tail:
            return []
        if tail in _COMMON_INDEX_NAMES or (full in _COMMON_INDEX_NAMES):
            return []
        if len(tail) <= 2:
            return []
        patterns: list[re.Pattern[str]] = [
            re.compile(rf"(?:->|\.)\s*{re.escape(tail)}\s*=(?!=)"),
            re.compile(rf"(?:->|\.)\s*{re.escape(tail)}\s*\+\+"),
            re.compile(rf"(?:->|\.)\s*{re.escape(tail)}\s*--"),
            re.compile(rf"\b{re.escape(tail)}\s*=(?!=)"),
            re.compile(rf"\b{re.escape(tail)}\s*\+\+"),
            re.compile(rf"\b{re.escape(tail)}\s*--"),
        ]
        if full and "." in full:
            root, field = full.rsplit(".", 1)
            patterns.append(
                re.compile(
                    rf"\b{re.escape(root)}\s*(?:\.|->)\s*{re.escape(field)}\s*=(?!=)"
                )
            )
            patterns.append(
                re.compile(
                    rf"\b{re.escape(root)}\s*(?:\.|->)\s*{re.escape(field)}\s*\+\+"
                )
            )

        hits: list[dict] = []
        hi = min(before_line - 1, len(self.source_lines) - 1)
        for ln in range(1, hi + 1):
            text = self.source_lines[ln].strip()
            if not text or tail not in text:
                continue
            if not any(p.search(text) for p in patterns):
                continue
            fn, _, _ = self.find_enclosing_function(ln)
            hits.append(
                {
                    "function": fn,
                    "location": (
                        f"ALL_bc_with_context.c:{ln}.1-"
                        f"{max(1, len(self.source_lines[ln]))}"
                    ),
                    "line": ln,
                    "process": None,
                    "assigned_expression_text": text[:300],
                    "call_depth": None,
                    "source": "source_scan_global",
                }
            )
            if len(hits) >= max_hits:
                break
        return hits

    def backward_slice(
        self,
        variable_name: str,
        from_location: str,
        *,
        max_depth: int = 8,
        max_writes: int = 80,
    ) -> dict:
        """Structural backward walk over DF writes + same-function source assigns + CF callers.

        Returns writes found, unresolved graph-boundary paths, and whether the
        walk stopped due to depth / write-count limits. No safety verdict.

        For each CF caller frame, writes are filtered against that caller's
        call-site location (not only the original alarm line), so functions that
        appear later in the preprocessed file are still searchable.
        """
        bare = (variable_name or "").strip().split("@", 1)[0].strip('"')
        lookup = self.df_lookup_stats(bare, from_location)
        target_line = self.parse_location_line(from_location)
        if target_line is None:
            return {
                "variable": bare,
                "target_location": from_location,
                "writes_found": [],
                "unresolved_paths": [
                    {
                        "reason": "could not parse from_location line number",
                        "function": None,
                        "location": from_location,
                    }
                ],
                "truncated": False,
                "lookup": lookup,
            }

        start_fn, fn_start, fn_end = self.find_enclosing_function(target_line)
        # Queue items: (fn, depth, f_start, f_end, before_location)
        queue: list[
            tuple[Optional[str], int, Optional[int], Optional[int], str]
        ] = [(start_fn, 0, fn_start, fn_end, from_location)]
        seen_fns: set[Optional[str]] = set()
        writes_found: list[dict] = []
        unresolved: list[dict] = []
        truncated = False
        seen_write_keys: set[tuple] = set()

        while queue:
            fn, depth, f_start, f_end, before_loc = queue.pop(0)
            if fn in seen_fns:
                continue
            seen_fns.add(fn)
            if depth > max_depth:
                truncated = True
                unresolved.append(
                    {
                        "reason": "max_depth reached before exploring further callers",
                        "function": fn,
                        "location": before_loc,
                    }
                )
                continue

            if fn:
                candidates = self.get_prior_writes(
                    bare, before_loc, function_scope=fn
                )
            else:
                candidates = self.get_prior_writes(bare, before_loc)

            for rec in candidates:
                key = ("df", rec.function, rec.location, rec.line, rec.process)
                if key in seen_write_keys:
                    continue
                seen_write_keys.add(key)
                expr = (
                    self.extract_assignment_text(rec.line, bare)
                    if rec.line is not None
                    else ""
                )
                writes_found.append(
                    {
                        "function": rec.function,
                        "location": rec.location,
                        "line": rec.line,
                        "process": rec.process,
                        "assigned_expression_text": expr,
                        "call_depth": depth,
                        "source": "data_flow",
                        "before_location_used": before_loc,
                    }
                )
                if len(writes_found) >= max_writes:
                    truncated = True
                    break

            # Same-function source scan (catches local / field assigns missing from DF).
            if (
                not truncated
                and f_start is not None
                and f_end is not None
            ):
                before_line = self.parse_location_line(before_loc) or target_line
                for hit in self.find_source_assignments(
                    bare,
                    before_line=before_line,
                    function_start=f_start,
                    function_end=f_end,
                ):
                    key = ("src", hit["location"], hit["line"])
                    if key in seen_write_keys:
                        continue
                    seen_write_keys.add(key)
                    hit = dict(hit)
                    hit["function"] = fn
                    hit["call_depth"] = depth
                    hit["before_location_used"] = before_loc
                    writes_found.append(hit)
                    if len(writes_found) >= max_writes:
                        truncated = True
                        break

            if truncated and len(writes_found) >= max_writes:
                break

            if fn and depth < max_depth:
                callers = self.control_by_callee.get(fn, [])
                if not callers:
                    unresolved.append(
                        {
                            "reason": (
                                "reached function with no visible caller in "
                                "control-flow graph"
                            ),
                            "function": fn,
                            "location": before_loc,
                        }
                    )
                else:
                    for edge in callers:
                        if edge.caller and edge.caller not in seen_fns:
                            c_start = c_end = None
                            if edge.line is not None:
                                c_fn, c_start, c_end = self.find_enclosing_function(
                                    edge.line
                                )
                                if c_fn != edge.caller:
                                    c_start = c_end = None
                            # Filter caller writes relative to this call site.
                            caller_before = edge.call_site or before_loc
                            queue.append(
                                (edge.caller, depth + 1, c_start, c_end, caller_before)
                            )
            elif fn and depth >= max_depth:
                truncated = True
                callers = self.control_by_callee.get(fn, [])
                if not callers:
                    unresolved.append(
                        {
                            "reason": (
                                "reached function with no visible caller in "
                                "control-flow graph"
                            ),
                            "function": fn,
                            "location": before_loc,
                        }
                    )
                else:
                    unresolved.append(
                        {
                            "reason": "max_depth reached with callers remaining unexplored",
                            "function": fn,
                            "location": before_loc,
                        }
                    )

        uniq: list[dict] = []
        seen_u: set[tuple] = set()
        for u in unresolved:
            k = (u.get("reason"), u.get("function"))
            if k in seen_u:
                continue
            seen_u.add(k)
            uniq.append(u)

        writes_found.sort(
            key=lambda w: (
                w.get("call_depth", 0) if w.get("call_depth") is not None else 99,
                w.get("line") or 0,
                w.get("function") or "",
            )
        )

        # When DF has no field-level keys (common for Struct.Field), supplement
        # with a capped file-wide source scan so the agent still sees writes.
        # Skipped for generic names (Index, i, …) — those must stay path-local.
        if len(writes_found) < 3 and target_line is not None:
            for hit in self.find_source_assignments_global(
                bare, before_line=target_line, max_hits=max_writes
            ):
                key = ("srcg", hit["location"], hit["line"])
                if key in seen_write_keys:
                    continue
                if any(w.get("line") == hit.get("line") for w in writes_found):
                    continue
                seen_write_keys.add(key)
                writes_found.append(hit)
                if len(writes_found) >= max_writes:
                    truncated = True
                    break
            writes_found.sort(
                key=lambda w: (
                    w.get("call_depth", 0) if w.get("call_depth") is not None else 99,
                    w.get("line") or 0,
                    w.get("function") or "",
                )
            )

        # Parameter note: name appears in the enclosing function signature.
        param_note = None
        call_site_info = None
        if start_fn and fn_start and not writes_found:
            header = " ".join(
                self.source_lines[ln].strip()
                for ln in range(fn_start, min(fn_start + 5, len(self.source_lines)))
            )
            if re.search(rf"\b{re.escape(bare.split('.')[-1])}\b", header):
                param_note = (
                    f"{bare.split('.')[-1]!r} appears in the signature of "
                    f"{start_fn} (likely a parameter). No prior writes found in "
                    "this function — origin is at call sites, not a local loop "
                    "in an unrelated function."
                )
                call_site_info = self.find_call_site_arguments(
                    start_fn,
                    bare,
                    from_location=from_location,
                    max_sites=10,
                )

        result = {
            "variable": bare,
            "target_location": from_location,
            "start_function": start_fn,
            "writes_found": writes_found,
            "unresolved_paths": uniq,
            "truncated": truncated,
            "functions_visited": [f for f in seen_fns if f],
            "lookup": lookup,
        }
        if param_note:
            result["parameter_note"] = param_note
        if call_site_info is not None:
            result["call_site_arguments"] = call_site_info
            result["navigation_hint"] = (
                "Parameter with no local writes. Use call_site_arguments."
                "next_slice_candidates (argument expressions at callers) as the "
                "next get_backward_slice / get_condition_guards targets. Do not "
                "treat empty writes_found as proof of OOB; do not treat an Astrée "
                "[0, type_max] vs small array alone as classification true."
            )
        return result

    def get_all_writes_to_symbol(self, symbol_name: str, *, max_writes: int = 120) -> dict:
        """Flat DF write listing for a symbol (no path / depth filter).

        Retrieval only — reachability to an alarm site is NOT implied.
        For Struct.Field names with no DF keys, supplements with a capped
        source scan of field assignments.
        """
        forms = self._symbol_name_forms(symbol_name)
        bare = forms["full"] or ""
        tail = forms["tail"] or bare
        keys = sorted(
            {
                k
                for k in self.data_flow_by_variable
                if k == bare
                or k.startswith(bare + "@")
                or (tail and (k == tail or k.startswith(tail + "@")))
                or (
                    forms["parent"]
                    and (k == forms["parent"] or k.startswith(forms["parent"] + "@"))
                )
            }
        )
        writes: list[dict] = []
        seen: set[tuple] = set()
        for key in keys:
            for rec in self.data_flow_by_variable.get(key, []):
                if (rec.access or "").lower() != "write":
                    continue
                # Parent aggregate keys: keep only lines that mention the field.
                if forms["parent"] and key.startswith(forms["parent"]):
                    line_txt = (
                        self.source_lines[rec.line]
                        if rec.line and rec.line < len(self.source_lines)
                        else ""
                    )
                    if tail and tail not in line_txt:
                        continue
                sig = (rec.variable, rec.function, rec.location, rec.line, rec.process)
                if sig in seen:
                    continue
                seen.add(sig)
                writes.append(
                    {
                        "variable": rec.variable,
                        "function": rec.function,
                        "location": rec.location,
                        "line": rec.line,
                        "process": rec.process,
                        "assigned_expression_text": self.extract_assignment_text(
                            rec.line or 0, tail or bare
                        )
                        if rec.line
                        else "",
                        "source": "data_flow",
                    }
                )
        # Source supplement when DF is empty / sparse for fields.
        if len(writes) < 5:
            for hit in self.find_source_assignments_global(
                bare or tail,
                before_line=len(self.source_lines),
                max_hits=max_writes,
            ):
                sig = ("src", hit.get("location"), hit.get("line"))
                if sig in seen:
                    continue
                if any(w.get("line") == hit.get("line") for w in writes):
                    continue
                seen.add(sig)
                writes.append(hit)
        writes.sort(key=lambda w: (w.get("line") or 0, w.get("function") or ""))
        truncated = len(writes) > max_writes
        return {
            "symbol": bare,
            "df_keys_matched": keys,
            "write_count": len(writes),
            "writes": writes[:max_writes],
            "truncated": truncated,
        }

    def resolve_symbolic_constant(self, name: str) -> dict:
        """Look up a #define or enum member literal value in input.c.

        Retrieval only — no guessing for macros/expressions that are not plain
        integer literals. Sequential enum members without ``=`` get their
        C ordinal (0, 1, 2, …) when the enclosing enum block can be parsed.
        """
        key = (name or "").strip().split("@", 1)[0].strip('"')
        if not key or not re.match(r"^[A-Za-z_]\w*$", key):
            return {
                "name": key,
                "found": False,
                "parse_note": "Name must be a bare C identifier.",
            }

        # 1) #define NAME literal
        define_re = re.compile(
            rf"^\s*#\s*define\s+{re.escape(key)}\s+(.+)$"
        )
        for ln in range(1, len(self.source_lines)):
            text = self.source_lines[ln]
            m = define_re.match(text)
            if not m:
                continue
            rhs = m.group(1).strip()
            # Strip trailing line comments
            rhs = re.split(r"\s+//|\s+/\*", rhs, maxsplit=1)[0].strip()
            lit = self._parse_int_literal(rhs)
            if lit is not None:
                return {
                    "name": key,
                    "found": True,
                    "kind": "define",
                    "value": lit["value"],
                    "value_text": lit["text"],
                    "location": f"ALL_bc_with_context.c:{ln}.1-{max(1, len(text))}",
                    "raw_text": text.strip()[:240],
                    "parse_note": None,
                }
            return {
                "name": key,
                "found": True,
                "kind": "define",
                "value": None,
                "value_text": rhs[:120],
                "location": f"ALL_bc_with_context.c:{ln}.1-{max(1, len(text))}",
                "raw_text": text.strip()[:240],
                "parse_note": (
                    "Found #define but RHS is not a plain integer literal "
                    "(macro/expression) — value left unresolved."
                ),
            }

        # 2) Enum member (explicit = or sequential ordinal)
        word_re = re.compile(rf"\b{re.escape(key)}\b")
        for ln in range(1, len(self.source_lines)):
            text = self.source_lines[ln]
            if not word_re.search(text):
                continue
            # Prefer lines that look like enum member lists / assignments.
            if "enum" not in text and "=" not in text and "," not in text:
                # Could still be inside a multi-line enum — check nearby header.
                if not self._line_inside_enum(ln):
                    continue
            enum_info = self._resolve_enum_member_at_line(key, ln)
            if enum_info is not None:
                return enum_info

        return {
            "name": key,
            "found": False,
            "parse_note": (
                "No #define or enum member with this exact name found in input.c."
            ),
            "name_suggestions": self.declaration_name_suggestions(key),
        }

    @staticmethod
    def _parse_int_literal(text: str) -> Optional[dict]:
        t = (text or "").strip()
        if not t:
            return None
        # Unwrap one layer of parentheses: (5u)
        if t.startswith("(") and t.endswith(")"):
            t = t[1:-1].strip()
        m = re.fullmatch(
            r"([+\-]?(?:0[xX][0-9A-Fa-f]+|\d+))(?:[uUlL]{0,3})",
            t,
        )
        if not m:
            return None
        raw = m.group(1)
        try:
            val = int(raw, 0)
        except ValueError:
            return None
        return {"value": val, "text": t}

    def _line_inside_enum(self, line: int) -> bool:
        lo = max(1, line - 40)
        for ln in range(line, lo - 1, -1):
            t = self.source_lines[ln]
            if re.search(r"\benum\b", t):
                return True
            if "}" in t and "enum" not in t:
                return False
        return False

    def _resolve_enum_member_at_line(self, key: str, hit_line: int) -> Optional[dict]:
        # Scan upward for enum / typedef enum opener.
        start = None
        for ln in range(hit_line, max(0, hit_line - 80), -1):
            t = self.source_lines[ln]
            if re.search(r"\benum\b", t):
                start = ln
                break
            if ln < hit_line and re.match(r"^\s*}\s*\w*\s*;", t):
                return None
        if start is None:
            return None

        # Collect body until closing brace.
        body_parts: list[str] = []
        end_line = start
        for ln in range(start, min(len(self.source_lines), start + 120)):
            body_parts.append(self.source_lines[ln])
            end_line = ln
            # Stop after enum's closing `};` (not the opening line alone).
            if ln > start and "}" in self.source_lines[ln]:
                break
        body = " ".join(body_parts)
        # Members: IDENT or IDENT = expr, separated by commas.
        # Strip comments roughly.
        body = re.sub(r"/\*.*?\*/", " ", body)
        body = re.sub(r"//.*?$", " ", body, flags=re.M)
        # Focus between { and }
        brace = re.search(r"\{(.*)\}", body, flags=re.S)
        if not brace:
            return None
        inner = brace.group(1)
        members: list[tuple[str, Optional[int], Optional[str]]] = []
        next_val = 0
        for part in inner.split(","):
            part = part.strip()
            if not part:
                continue
            m = re.match(
                r"([A-Za-z_]\w*)\s*(?:=\s*([^,\s][^,]*))?",
                part,
            )
            if not m:
                continue
            name = m.group(1)
            rhs = (m.group(2) or "").strip() or None
            if rhs:
                lit = self._parse_int_literal(rhs)
                if lit is None:
                    members.append((name, None, rhs))
                    # Cannot continue auto-numbering reliably.
                    next_val = None  # type: ignore[assignment]
                else:
                    members.append((name, lit["value"], lit["text"]))
                    next_val = lit["value"] + 1
            else:
                if next_val is None:
                    members.append((name, None, None))
                else:
                    members.append((name, next_val, str(next_val)))
                    next_val += 1

        for name, value, value_text in members:
            if name != key:
                continue
            return {
                "name": key,
                "found": True,
                "kind": "enum_member",
                "value": value,
                "value_text": value_text,
                "enum_location": f"ALL_bc_with_context.c:{start}.1-{end_line}",
                "enum_member_count": len(members),
                "raw_text": self.source_lines[hit_line].strip()[:240],
                "parse_note": (
                    None
                    if value is not None
                    else "Enum member found but value could not be resolved to a literal."
                ),
            }
        return None

    def find_condition_guards(
        self,
        variable_name: str,
        from_location: str,
        *,
        max_depth: int = 8,
        max_guards: int = 40,
    ) -> dict:
        """Find if/switch/assert/ternary conditions mentioning ``variable_name``.

        Walks the same CF caller climb idea as backward_slice, scanning source
        in each visited function for conditional text before ``from_location``
        (or before each call-site location). Retrieval only — does not judge
        whether a guard prevents the access.
        """
        bare = (variable_name or "").strip().split("@", 1)[0].strip('"')
        # Also match bare field: foo.bar → bar
        tokens = {bare, bare.split(".")[-1]}
        tokens = {t for t in tokens if t}
        target_line = self.parse_location_line(from_location)
        if target_line is None:
            return {
                "variable": bare,
                "from_location": from_location,
                "guards_found": [],
                "error": "could not parse from_location",
            }

        depth_lim = max(0, min(int(max_depth), 32))
        start_fn, fn_start, fn_end = self.find_enclosing_function(target_line)
        queue: list[
            tuple[Optional[str], int, Optional[int], Optional[int], str]
        ] = [(start_fn, 0, fn_start, fn_end, from_location)]
        seen_fns: set[Optional[str]] = set()
        guards: list[dict] = []
        truncated = False

        while queue and len(guards) < max_guards:
            fn, depth, f_start, f_end, before_loc = queue.pop(0)
            if fn in seen_fns:
                continue
            seen_fns.add(fn)
            before_line = self.parse_location_line(before_loc) or target_line
            if f_start and f_end:
                hits = self._scan_guards_in_span(
                    tokens, f_start, min(f_end, before_line), function=fn, call_depth=depth
                )
                guards.extend(hits)
                if len(guards) >= max_guards:
                    truncated = True
                    break

            if depth >= depth_lim:
                continue
            if not fn:
                continue
            callers = self.control_by_callee.get(fn, [])[:30]
            for edge in callers:
                c_line = edge.line
                if c_line is None:
                    continue
                c_fn, c_start, c_end = self.find_enclosing_function(c_line)
                if c_fn in seen_fns:
                    continue
                queue.append(
                    (
                        c_fn,
                        depth + 1,
                        c_start,
                        c_end,
                        edge.call_site
                        or f"ALL_bc_with_context.c:{c_line}.1-1",
                    )
                )

        # De-dupe by location + text
        uniq: list[dict] = []
        seen_g: set[tuple] = set()
        for g in guards:
            k = (g.get("location"), g.get("condition_text"))
            if k in seen_g:
                continue
            seen_g.add(k)
            uniq.append(g)

        return {
            "variable": bare,
            "from_location": from_location,
            "start_function": start_fn,
            "guards_found": uniq[:max_guards],
            "functions_visited": [f for f in seen_fns if f],
            "truncated": truncated or len(uniq) > max_guards,
        }

    def _scan_guards_in_span(
        self,
        tokens: set[str],
        start: int,
        end: int,
        *,
        function: Optional[str],
        call_depth: int,
    ) -> list[dict]:
        hits: list[dict] = []
        lo = max(1, start)
        hi = min(end, len(self.source_lines) - 1)
        # Multi-line if: accumulate when an if( opens without ) on same line.
        pending: Optional[dict] = None
        for ln in range(lo, hi + 1):
            raw = self.source_lines[ln]
            text = raw.strip()
            if not text or text.startswith("#"):
                continue
            combined = text
            kind = None
            if pending:
                pending["lines"].append(text)
                pending["end_line"] = ln
                blob = " ".join(pending["lines"])
                if blob.count("(") <= blob.count(")"):
                    combined = blob
                    kind = pending["kind"]
                    start_line = pending["start_line"]
                    pending = None
                else:
                    continue
            else:
                start_line = ln
                if re.search(r"\belse\s+if\b\s*\(", text) or re.search(
                    r"\bif\s*\(", text
                ):
                    kind = "if"
                elif re.search(r"\bswitch\s*\(", text):
                    kind = "switch"
                elif re.search(r"\bfor\s*\(", text):
                    kind = "for"
                elif re.search(r"\bwhile\s*\(", text):
                    kind = "while"
                elif re.search(r"\bassert\s*\(", text) or re.search(
                    r"\bASSERT\s*\(", text
                ):
                    kind = "assert"
                elif "?" in text and ":" in text:
                    kind = "ternary"
                else:
                    continue
                # Multi-line condition
                if kind in {"if", "switch", "assert", "for", "while"} and text.count(
                    "("
                ) > text.count(")"):
                    pending = {
                        "kind": kind,
                        "start_line": ln,
                        "end_line": ln,
                        "lines": [text],
                    }
                    continue

            if kind is None:
                continue
            if not any(re.search(rf"\b{re.escape(tok)}\b", combined) for tok in tokens):
                continue
            # Keep a compact condition text (first ~240 chars).
            cond = re.sub(r"\s+", " ", combined).strip()[:240]
            hits.append(
                {
                    "condition_text": cond,
                    "kind": kind,
                    "function": function,
                    "location": (
                        f"ALL_bc_with_context.c:{start_line}.1-"
                        f"{max(1, len(self.source_lines[start_line]))}"
                    ),
                    "line": start_line,
                    "call_depth": call_depth,
                }
            )
        return hits
