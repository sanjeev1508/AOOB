"""Indexed loaders for Astrée CSVs and the C source used by AOOB tools."""

from __future__ import annotations

import bisect
import csv
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

from tree_sitter import Language, Node, Parser
import tree_sitter_c as tsc

_C_LANGUAGE = Language(tsc.language())
_C_PARSER = Parser(_C_LANGUAGE)

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
    }
)

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

_SNIP_PREFIX = "void __aoob_snip(void) {\n"
_SNIP_SKIP = frozenset({"__aoob_snip"})


def _iter_nodes(root: Node) -> Iterator[Node]:
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        for i in range(node.child_count - 1, -1, -1):
            child = node.child(i)
            if child is not None:
                stack.append(child)


def _node_text(source: bytes, node: Optional[Node]) -> str:
    if node is None:
        return ""
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def collapse_ws(text: str) -> str:
    return " ".join((text or "").split())


def ident_token(name: str) -> bool:
    if not name:
        return False
    if not (name[0].isalpha() or name[0] == "_"):
        return False
    return all(ch.isalnum() or ch == "_" for ch in name)


def token_in_text(text: str, token: str) -> bool:
    if not token or not text:
        return False
    start = 0
    n = len(token)
    while True:
        idx = text.find(token, start)
        if idx < 0:
            return False
        left_ok = idx == 0 or not (text[idx - 1].isalnum() or text[idx - 1] == "_")
        right = idx + n
        right_ok = right >= len(text) or not (text[right].isalnum() or text[right] == "_")
        if left_ok and right_ok:
            return True
        start = idx + 1


def parse_location(loc: str) -> Optional[dict]:
    """Parse ``file:line.col`` / ``file:line.col-col`` / ``file:line.col-line.col``."""
    text = (loc or "").strip()
    if ":" not in text:
        return None
    file_part, rest = text.rsplit(":", 1)
    dash = rest.find("-")
    start = rest if dash < 0 else rest[:dash]
    end = rest[dash + 1 :] if dash >= 0 else ""
    if "." not in start:
        return None
    line_s, col_s = start.split(".", 1)
    if not line_s.isdigit() or not col_s.isdigit():
        return None
    line = int(line_s)
    col_start = int(col_s)
    end_line = line
    col_end = col_start
    if end:
        if "." in end:
            el, ec = end.split(".", 1)
            if el.isdigit():
                end_line = int(el)
            if ec.isdigit():
                col_end = int(ec)
        elif end.isdigit():
            col_end = int(end)
    return {
        "file": file_part,
        "line": line,
        "col_start": col_start,
        "col_end": col_end,
        "end_line": end_line,
    }


def parse_c_source(text: str) -> tuple[bytes, Any]:
    blob = text.encode("utf-8")
    return blob, _C_PARSER.parse(blob)


def parse_snippet(text: str) -> tuple[bytes, Any]:
    wrapped = f"{_SNIP_PREFIX}{text}\n}}\n"
    return parse_c_source(wrapped)


def _eval_int_literal(text: str) -> Optional[int]:
    raw = collapse_ws(text).replace(" ", "")
    if not raw:
        return None
    while raw.startswith("(") and raw.endswith(")") and len(raw) > 2:
        raw = raw[1:-1]
    suffixes = ("ULL", "LLU", "UL", "LU", "LL", "U", "L")
    upper = raw.upper()
    for suf in suffixes:
        if upper.endswith(suf):
            raw = raw[: -len(suf)]
            upper = raw.upper()
            break
    if raw.startswith(("0x", "0X")):
        try:
            return int(raw, 16)
        except ValueError:
            return None
    if raw.isdigit():
        return int(raw)
    return None


def _eval_size_node(source: bytes, node: Optional[Node]) -> tuple[Optional[int], str]:
    if node is None:
        return None, "missing array size expression"
    text = collapse_ws(_node_text(source, node))
    lit = _eval_int_literal(text)
    if lit is not None:
        return lit, ""
    if node.type in {"parenthesized_expression", "cast_expression"}:
        for child in node.named_children:
            if child.type == "type_descriptor":
                continue
            size, note = _eval_size_node(source, child)
            if size is not None:
                return size, note
    if node.type == "sizeof_expression":
        return None, f"non-literal array size: {text[:80]}"
    return None, f"non-literal array size: {text[:80]}"


def _unwrap_declarator(node: Optional[Node]) -> Optional[Node]:
    cur = node
    while cur is not None and cur.type in {
        "pointer_declarator",
        "function_declarator",
        "parenthesized_declarator",
        "abstract_pointer_declarator",
    }:
        nxt = cur.child_by_field_name("declarator")
        if nxt is None and cur.named_children:
            nxt = cur.named_children[0]
        cur = nxt
    return cur


def _declarator_name(source: bytes, node: Optional[Node]) -> str:
    cur = _unwrap_declarator(node)
    if cur is None:
        return ""
    if cur.type == "array_declarator":
        return _declarator_name(source, cur.child_by_field_name("declarator"))
    if cur.type in {"identifier", "field_identifier", "type_identifier"}:
        return _node_text(source, cur)
    for child in _iter_nodes(cur):
        if child.type in {"identifier", "field_identifier"}:
            return _node_text(source, child)
    return ""


def _find_array_declarator(node: Optional[Node]) -> Optional[Node]:
    cur = node
    seen = 0
    while cur is not None and seen < 12:
        seen += 1
        if cur.type == "array_declarator":
            return cur
        if cur.type in {
            "pointer_declarator",
            "function_declarator",
            "parenthesized_declarator",
        }:
            cur = cur.child_by_field_name("declarator")
            continue
        break
    if node is None:
        return None
    for child in _iter_nodes(node):
        if child.type == "array_declarator":
            return child
    return None


def _count_initializer_elements(node: Optional[Node]) -> Optional[int]:
    if node is None:
        return None
    if node.type == "initializer_list":
        return sum(1 for ch in node.named_children if ch.type != "comment")
    if node.type == "init_declarator":
        return _count_initializer_elements(node.child_by_field_name("value"))
    return None


def _type_prefix(source: bytes, node: Node) -> str:
    bits: list[str] = []
    for child in node.children:
        if child.type in {
            "field_declaration_list",
            ";",
            ",",
            "comment",
            "attribute_specifier",
        }:
            continue
        if child.type in {
            "init_declarator",
            "array_declarator",
            "pointer_declarator",
            "function_declarator",
            "identifier",
            "field_identifier",
            "parenthesized_declarator",
        }:
            break
        bits.append(_node_text(source, child))
    return collapse_ws(" ".join(bits)) or "unknown"


def _function_name_from_definition(source: bytes, node: Node) -> str:
    decl = node.child_by_field_name("declarator")
    name = _declarator_name(source, decl)
    if name and name not in _FUNC_NAME_BLACKLIST:
        return name
    for child in _iter_nodes(node):
        if child.type == "function_declarator":
            inner = child.child_by_field_name("declarator")
            name = _declarator_name(source, inner)
            if name and name not in _FUNC_NAME_BLACKLIST:
                return name
    return ""


def _looks_constant_expr(node: Node) -> bool:
    const_types = {
        "number_literal",
        "char_literal",
        "string_literal",
        "null",
        "true",
        "false",
    }
    if node.type in const_types:
        return True
    if node.type in {"parenthesized_expression", "cast_expression", "unary_expression"}:
        kids = [ch for ch in node.named_children if ch.type != "type_descriptor"]
        return bool(kids) and all(_looks_constant_expr(ch) for ch in kids)
    if node.type == "binary_expression":
        kids = list(node.named_children)
        return bool(kids) and all(_looks_constant_expr(ch) for ch in kids)
    if node.type == "sizeof_expression":
        return True
    return False


def identifiers_from_text(text: str) -> list[str]:
    blob, tree = parse_snippet((text or "") + ";")
    out: list[str] = []
    seen: set[str] = set()
    for node in _iter_nodes(tree.root_node):
        if node.type not in {"identifier", "field_identifier"}:
            continue
        if node.start_point[0] != 1:
            continue
        name = _node_text(blob, node)
        if not name or name in _SNIP_SKIP or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def member_paths_from_text(text: str) -> list[str]:
    blob, tree = parse_snippet((text or "") + ";")
    out: list[str] = []
    seen: set[str] = set()

    def _path(node: Node) -> str:
        if node.type in {"identifier", "field_identifier"}:
            return _node_text(blob, node)
        if node.type in {"field_expression", "pointer_expression"}:
            arg = node.child_by_field_name("argument")
            field = node.child_by_field_name("field")
            if arg is None or field is None:
                named = node.named_children
                if len(named) >= 2:
                    arg, field = named[0], named[-1]
                else:
                    return collapse_ws(_node_text(blob, node)).replace("->", ".")
            left = _path(arg)
            right = _node_text(blob, field)
            if left and right:
                return f"{left}.{right}"
        if node.type == "subscript_expression":
            arg = node.child_by_field_name("argument")
            return _path(arg) if arg is not None else ""
        if node.type == "parenthesized_expression" and node.named_children:
            return _path(node.named_children[0])
        return ""

    for node in _iter_nodes(tree.root_node):
        if node.type in {"field_expression"}:
            if node.start_point[0] != 1:
                continue
            path = _path(node)
            if path and path not in seen and "__aoob_snip" not in path:
                seen.add(path)
                out.append(path)
    return out


def callees_from_text(text: str) -> list[str]:
    blob, tree = parse_snippet((text or "") + ";")
    names: list[str] = []
    seen: set[str] = set()
    for node in _iter_nodes(tree.root_node):
        if node.type != "call_expression":
            continue
        fn = node.child_by_field_name("function")
        if fn is None:
            continue
        raw = collapse_ws(_node_text(blob, fn))
        ident = raw.replace("->", ".").split(".")[-1]
        if (
            not ident
            or not ident.isidentifier()
            or ident in _SNIP_SKIP
            or (fn is not None and fn.start_point[0] != 1)
        ):
            continue
        if ident not in seen:
            seen.add(ident)
            names.append(ident)
    return names


def subscripts_from_text(line_text: str) -> list[dict[str, Any]]:
    """subscript_expression nodes in one source line (AST, line-break safe)."""
    blob, tree = parse_snippet(line_text or "")
    out: list[dict[str, Any]] = []
    for node in _iter_nodes(tree.root_node):
        if node.type != "subscript_expression":
            continue
        if node.start_point[0] != 1:
            continue
        arg = node.child_by_field_name("argument")
        idx = node.child_by_field_name("index")
        if arg is None or idx is None:
            named = node.named_children
            if len(named) >= 2:
                arg, idx = named[0], named[1]
            else:
                continue
        bracket_col = None
        for ch in node.children:
            if ch.type == "[":
                bracket_col = ch.start_point[1]
                break
        if bracket_col is None:
            bracket_col = node.start_point[1]
        indexed_raw = collapse_ws(_node_text(blob, arg)).replace("->", ".")
        indexed_raw = indexed_raw.replace(" ", "")
        out.append(
            {
                "indexed": indexed_raw,
                "operand": collapse_ws(_node_text(blob, idx)),
                "bracket_col": bracket_col,
                "text": collapse_ws(_node_text(blob, node)),
            }
        )
    return out


def is_constant_init_text(text: str, token: str) -> bool:
    if not token or not text:
        return False
    blob, tree = parse_snippet(text if text.rstrip().endswith(";") else text + ";")
    for node in _iter_nodes(tree.root_node):
        if node.type == "assignment_expression":
            left = node.child_by_field_name("left")
            right = node.child_by_field_name("right")
            lhs = collapse_ws(_node_text(blob, left)) if left else ""
            if token_in_text(lhs, token.split(".")[-1]) or lhs.endswith(token) or token in lhs:
                return bool(right) and _looks_constant_expr(right)
        if node.type == "init_declarator":
            d = node.child_by_field_name("declarator")
            v = node.child_by_field_name("value")
            name = _declarator_name(blob, d)
            if name == token or name.endswith(token.split(".")[-1]):
                return bool(v) and _looks_constant_expr(v)
    return False


def _cmp_op(node: Node) -> str:
    for ch in node.children:
        if ch.type in {"<", ">", "<=", ">=", "==", "!="}:
            return ch.type
    return ""


def _arith_op(node: Node) -> str:
    for ch in node.children:
        if ch.type in {"+", "-", "*", "/", "%"}:
            return ch.type
    return ""


def _unwrap_value(node: Optional[Node]) -> Optional[Node]:
    """Peel parens, casts, and unary +/-/~ so the compared value is visible."""
    cur = node
    seen = 0
    while cur is not None and seen < 8:
        seen += 1
        if cur.type == "parenthesized_expression":
            cur = cur.named_children[0] if cur.named_children else None
            continue
        if cur.type == "cast_expression":
            cur = cur.child_by_field_name("value") or (
                cur.named_children[-1] if cur.named_children else None
            )
            continue
        if cur.type == "unary_expression":
            inner = cur.named_children[0] if cur.named_children else None
            if inner is None:
                break
            cur = inner
            continue
        break
    return cur


def _direct_token_expr(blob: bytes, node: Optional[Node], token: str) -> bool:
    """True when *node* is the token, or arithmetic on the token (idx + 1).

    The token may not hide only inside a subscript index or a call argument.
    ``5u <= CurProfile`` and ``idx + 1 < 8`` qualify; ``Sorted[CurProfile]``
    does not.
    """
    cur = _unwrap_value(node)
    if cur is None or not token:
        return False
    tail = token.split(".")[-1]
    if cur.type == "identifier":
        name = _node_text(blob, cur)
        return name == token or name == tail
    if cur.type == "field_expression":
        field = cur.child_by_field_name("field")
        fname = _node_text(blob, field) if field is not None else ""
        path = collapse_ws(_node_text(blob, cur)).replace("->", ".")
        return (
            path == token
            or path.endswith("." + tail)
            or fname == token
            or fname == tail
        )
    if cur.type == "binary_expression" and _arith_op(cur):
        return _direct_token_expr(
            blob, cur.child_by_field_name("left"), token
        ) or _direct_token_expr(blob, cur.child_by_field_name("right"), token)
    return False


def _other_derives_from_token(blob: bytes, node: Optional[Node], token: str) -> bool:
    """Other comparison side is a subscript / member / call that mentions token.

    Filters ``PhysAddrPtr[i] != Sorted[CurProfile]->Mac[i]`` once one side
    has already been accepted as a direct token (it usually hasn't — both
    sides fail _direct_token_expr — but this also drops ``idx != Foo[idx]``).
    """
    cur = _unwrap_value(node)
    if cur is None or not token:
        return False
    if cur.type not in {"subscript_expression", "field_expression", "call_expression"}:
        return False
    return token_in_text(_node_text(blob, cur), token.split(".")[-1] or token)


def _looks_bound_rhs(blob: bytes, node: Optional[Node]) -> bool:
    """Literal, sizeof, enum/define-like ident, or a *Size/*Num/*Count member.

    Accept cases (see condition_compares_token):
    - ``5u <= CurProfile`` — number_literal
    - ``idx < SlacNumProfiles`` — identifier / field containing Num
    Reject is decided by the caller when the token is not a direct operand.
    """
    cur = _unwrap_value(node)
    if cur is None:
        return False
    if cur.type in {"number_literal", "sizeof_expression"}:
        return True
    markers = ("Num", "Size", "Count", "Max", "Cnt")
    if cur.type == "identifier":
        name = _node_text(blob, cur)
        if any(m in name for m in markers):
            return True
        letters = name.replace("_", "")
        return bool(letters) and letters.isupper()
    if cur.type == "field_expression":
        field = cur.child_by_field_name("field")
        fname = _node_text(blob, field) if field is not None else ""
        return any(m in fname for m in markers)
    return False


def condition_compares_token(condition: str, token: str) -> bool:
    """True only when a relational/equality actually *bounds* the token.

    Inline cases:
        ``5u <= CurProfile``  → ACCEPT (literal bounds CurProfile)
        ``PhysAddrPtr[i] != Sorted[CurProfile]->Mac[i]``  → REJECT
            (token only appears as a subscript index on both sides)
        ``while LOOKUP[GetCoreIdx(checkword)] != &end``  → REJECT
            (token only appears inside the index / call; also the alarm line)
        ``Signals_SigNumValid(SigNum) != 0``  → REJECT
            (left is a call wrapper, not the token; do not special-case it in)
    """
    if not token_in_text(condition or "", token):
        return False
    blob, tree = parse_snippet(f"if ({condition}) {{}}")
    tail = token.split(".")[-1] or token
    for node in _iter_nodes(tree.root_node):
        if node.type != "binary_expression":
            continue
        if not _cmp_op(node):
            continue
        left = node.child_by_field_name("left")
        right = node.child_by_field_name("right")
        left_tok = _direct_token_expr(blob, left, token) or _direct_token_expr(
            blob, left, tail
        )
        right_tok = _direct_token_expr(blob, right, token) or _direct_token_expr(
            blob, right, tail
        )
        if not left_tok and not right_tok:
            continue
        other = right if left_tok else left
        if _other_derives_from_token(blob, other, token):
            continue
        # Direct token vs anything that is not another use of the token as
        # an index/call. Literals / sizeof / *Num members are the common
        # bound; a plain identifier on the other side is also a bound form
        # (``idx < limit``) and is not the MAC-compare pattern.
        if _looks_bound_rhs(blob, other) or not _other_derives_from_token(
            blob, other, token
        ):
            return True
    return False


def call_site_writes_operand(line_text: str, callee: str, operand: str) -> bool:
    """True when this call assigns the operand or receives ``&operand``.

    Inline cases:
        ``MoCEMM_Co_UpdateChkwordCoreTsk(&checkword, core, tsk);`` → write
        ``Mo_Inst_GetTskIdx(checkword)`` → read (by-value argument)
        ``MoCMem_Co_ChkSingleU8(checkword, &result)`` → read (writes result)
        ``EthTrcv_30_Ar7000_Slac_VConfirming___625(TrcvIdx, SlacCurProfileValidation)``
            → read (Mode passed by value)
    """
    if not callee or not operand or not line_text:
        return False
    blob, tree = parse_snippet(line_text if line_text.rstrip().endswith(";") else line_text + ";")
    tail = operand.split(".")[-1] or operand

    def _call_name(call: Node) -> str:
        fn = call.child_by_field_name("function")
        raw = collapse_ws(_node_text(blob, fn))
        return raw.replace("->", ".").split(".")[-1]

    def _is_amp_operand(arg: Optional[Node]) -> bool:
        cur = _unwrap_value(arg)
        if cur is None or cur.type != "pointer_expression":
            return False
        if not any(ch.type == "&" for ch in cur.children):
            return False
        inner = cur.named_children[0] if cur.named_children else None
        return _direct_token_expr(blob, inner, operand) or _direct_token_expr(
            blob, inner, tail
        )

    for node in _iter_nodes(tree.root_node):
        if node.type == "assignment_expression":
            right = node.child_by_field_name("right")
            right_u = _unwrap_value(right)
            if right_u is not None and right_u.type == "call_expression":
                if _call_name(right_u) == callee and (
                    _direct_token_expr(blob, node.child_by_field_name("left"), operand)
                    or _direct_token_expr(blob, node.child_by_field_name("left"), tail)
                ):
                    return True
        if node.type != "call_expression" or _call_name(node) != callee:
            continue
        args = node.child_by_field_name("arguments")
        if args is None:
            continue
        for ch in args.named_children:
            if _is_amp_operand(ch):
                return True
    return False


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
        return parse_location(self.location)


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
    declarations_all: dict[str, list[DeclarationRecord]] = field(
        default_factory=lambda: defaultdict(list)
    )
    object_types: dict[str, str] = field(default_factory=dict)
    type_array_fields: dict[str, list[DeclarationRecord]] = field(
        default_factory=lambda: defaultdict(list)
    )
    source_file_tag: str = "input.c"
    source_bytes: bytes = b""
    source_tree: Any = None
    _functions_by_name: dict[str, list[tuple[int, int]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    _functions_by_start: dict[int, tuple[str, int, int]] = field(default_factory=dict)
    _function_spans: list[tuple[int, int, str]] = field(default_factory=list)
    _fn_start_keys: list[int] = field(default_factory=list)
    _assigns: list[dict[str, Any]] = field(default_factory=list)
    _guards: list[dict[str, Any]] = field(default_factory=list)
    _calls_by_line: dict[int, list[dict[str, Any]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    _source_slice_cache: dict[tuple[int, int], list[tuple[int, str]]] = field(
        default_factory=dict, repr=False
    )
    _function_snippet_cache: dict[tuple[str, Optional[int]], dict[str, Any]] = field(
        default_factory=dict, repr=False
    )
    _micro_window_cache: dict[tuple[str, int, int], dict[str, Any]] = field(
        default_factory=dict, repr=False
    )

    @classmethod
    def load(cls, data_dir: Path | str) -> "DataStore":
        data_dir = Path(data_dir)
        store = cls(root=data_dir)
        store._load_alarms(data_dir / "Full_alarms.csv")
        store._infer_source_file_tag()
        store._load_data_flow(data_dir / "data flow.csv")
        store._load_control_flow(data_dir / "control flow.csv")
        store._load_source(data_dir / "input.c")
        store._build_declaration_index()
        return store

    def _infer_source_file_tag(self) -> None:
        """Use the source filename from the first parseable alarm Location."""
        for alarm in self.alarms.values():
            loc = (alarm.location or "").strip()
            if ":" not in loc:
                continue
            tag = loc.split(":", 1)[0].strip()
            if tag:
                self.source_file_tag = tag
                return

    def synth_location(self, line: int | None, col_end: int | None = None) -> str:
        """Build a Location-shaped string using this dump's source filename."""
        if line is None:
            return ""
        end = max(1, int(col_end if col_end is not None else 1))
        return f"{self.source_file_tag}:{int(line)}.1-{end}"

    def _read_semicolon_csv(self, path: Path) -> list[dict[str, str]]:
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        header_idx = None
        for i, line in enumerate(lines):
            if ";" in line and not line.lower().startswith("sep="):
                first = line.split(";")[0]
                if any(ch.isalpha() for ch in first):
                    header_idx = i
                    if "Caller" in line or "Variable" in line or "Order" in line:
                        break
        if header_idx is None:
            raise ValueError(f"No CSV header found in {path}")
        payload = "\n".join(lines[header_idx:])
        reader = csv.DictReader(payload.splitlines(), delimiter=";")
        return [dict(row) for row in reader if any((v or "").strip() for v in row.values())]

    @staticmethod
    def parse_location_line(location: str) -> Optional[int]:
        parsed = parse_location(location or "")
        if not parsed:
            return None
        line = parsed.get("line")
        return int(line) if isinstance(line, int) else None

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
        text = path.read_text(encoding="utf-8", errors="replace")
        self.source_lines = [""] + text.splitlines()
        self.source_bytes, self.source_tree = parse_c_source(text)
        self._index_ast_spans()

    def _index_ast_spans(self) -> None:
        self._functions_by_name.clear()
        self._functions_by_start.clear()
        self._function_spans.clear()
        self._assigns.clear()
        self._guards.clear()
        self._calls_by_line.clear()
        if self.source_tree is None:
            return
        src = self.source_bytes
        for node in _iter_nodes(self.source_tree.root_node):
            ntype = node.type
            if ntype == "function_definition":
                name = _function_name_from_definition(src, node)
                if not name:
                    continue
                start = node.start_point[0] + 1
                end = node.end_point[0] + 1
                self._functions_by_name[name].append((start, end))
                self._functions_by_start[start] = (name, start, end)
                self._function_spans.append((start, end, name))
            elif ntype == "assignment_expression":
                left = node.child_by_field_name("left")
                right = node.child_by_field_name("right")
                op = "="
                for ch in node.children:
                    t = ch.type
                    if t in {"=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>="}:
                        op = t
                        break
                self._assigns.append(
                    {
                        "line": node.start_point[0] + 1,
                        "lhs": collapse_ws(_node_text(src, left)),
                        "op": op,
                        "rhs": collapse_ws(_node_text(src, right)),
                        "text": collapse_ws(_node_text(src, node))[:300],
                    }
                )
            elif ntype == "init_declarator":
                val = node.child_by_field_name("value")
                if val is None:
                    continue
                d = node.child_by_field_name("declarator")
                self._assigns.append(
                    {
                        "line": node.start_point[0] + 1,
                        "lhs": _declarator_name(src, d),
                        "op": "=",
                        "rhs": collapse_ws(_node_text(src, val)),
                        "text": collapse_ws(_node_text(src, node))[:300],
                    }
                )
            elif ntype == "update_expression":
                arg = node.child_by_field_name("argument")
                if arg is None and node.named_children:
                    arg = node.named_children[0]
                self._assigns.append(
                    {
                        "line": node.start_point[0] + 1,
                        "lhs": collapse_ws(_node_text(src, arg)),
                        "op": "++/--",
                        "rhs": "",
                        "text": collapse_ws(_node_text(src, node))[:300],
                    }
                )
            elif ntype in {
                "if_statement",
                "while_statement",
                "for_statement",
                "switch_statement",
                "conditional_expression",
            }:
                cond = node.child_by_field_name("condition")
                kind = {
                    "if_statement": "if",
                    "while_statement": "while",
                    "for_statement": "for",
                    "switch_statement": "switch",
                    "conditional_expression": "ternary",
                }[ntype]
                if cond is None and ntype == "for_statement":
                    named = [ch for ch in node.named_children if ch.type != "compound_statement"]
                    cond = named[1] if len(named) >= 2 else None
                if cond is None:
                    continue
                self._guards.append(
                    {
                        "line": node.start_point[0] + 1,
                        "kind": kind,
                        "condition": collapse_ws(_node_text(src, cond))[:240],
                    }
                )
            elif ntype == "call_expression":
                fn = node.child_by_field_name("function")
                args_n = node.child_by_field_name("arguments")
                fn_text = collapse_ws(_node_text(src, fn))
                fn_name = fn_text.replace("->", ".").split(".")[-1]
                arg_nodes = [
                    ch
                    for ch in (args_n.named_children if args_n is not None else [])
                    if ch.type != "comment"
                ]
                line = node.start_point[0] + 1
                self._calls_by_line[line].append(
                    {
                        "function": fn_name,
                        "function_text": fn_text,
                        "args": [collapse_ws(_node_text(src, a)) for a in arg_nodes],
                        "call_text": collapse_ws(_node_text(src, node))[:280],
                        "line": line,
                        "end_line": node.end_point[0] + 1,
                    }
                )
        self._function_spans.sort(key=lambda s: s[0])
        self._fn_start_keys = [s[0] for s in self._function_spans]
        self._assigns.sort(key=lambda a: a["line"])
        self._guards.sort(key=lambda g: g["line"])

    def function_span(
        self, name: str, near_line: Optional[int] = None
    ) -> Optional[tuple[int, int]]:
        spans = self._functions_by_name.get(name) or []
        if not spans:
            return None
        if near_line is None:
            return spans[0]
        return min(spans, key=lambda sp: abs(sp[0] - near_line))

    def get_alarm(self, order_id: int) -> Optional[AlarmRecord]:
        return self.alarms.get(order_id)

    def get_source_slice(self, start: int, end: int) -> list[tuple[int, str]]:
        start = max(1, start)
        end = min(len(self.source_lines) - 1, end)
        key = (start, end)
        cached = self._source_slice_cache.get(key)
        if cached is not None:
            return cached
        rows = [(i, self.source_lines[i]) for i in range(start, end + 1)]
        self._source_slice_cache[key] = rows
        return rows

    def cached_function_snippet(
        self, function_name: str, *, anchor_line: Optional[int] = None
    ) -> Optional[dict[str, Any]]:
        """Process-level cache for full function snippets (cross-alarm reuse)."""
        key = (function_name, anchor_line)
        return self._function_snippet_cache.get(key)

    def store_function_snippet(
        self,
        function_name: str,
        *,
        anchor_line: Optional[int] = None,
        payload: dict[str, Any],
    ) -> None:
        key = (function_name, anchor_line)
        self._function_snippet_cache[key] = payload

    def cached_micro_window(
        self, function_name: str, start_line: int, end_line: int
    ) -> Optional[dict[str, Any]]:
        """Process-level cache for micro-window payloads (cross-alarm reuse)."""
        return self._micro_window_cache.get((function_name, start_line, end_line))

    def store_micro_window(
        self,
        function_name: str,
        start_line: int,
        end_line: int,
        payload: dict[str, Any],
    ) -> None:
        self._micro_window_cache[(function_name, start_line, end_line)] = payload

    def _window_text(self, start_line: int, end_line: int) -> str:
        start_line = max(1, start_line)
        end_line = min(len(self.source_lines) - 1, end_line)
        return "\n".join(self.source_lines[i] for i in range(start_line, end_line + 1))

    def parse_function_parameters(
        self, function_name: str, near_line: Optional[int] = None
    ) -> list[dict]:
        """Parse parameter names/types from a function_definition parameter_list."""
        fn = (function_name or "").strip()
        if not fn or self.source_tree is None:
            return []
        spans = self._functions_by_name.get(fn) or []
        if not spans:
            return []
        if near_line is not None:
            start, end = min(spans, key=lambda sp: abs(sp[0] - near_line))
        else:
            start, end = spans[0]
        src = self.source_bytes
        for node in _iter_nodes(self.source_tree.root_node):
            if node.type != "function_definition":
                continue
            if node.start_point[0] + 1 != start:
                continue
            if _function_name_from_definition(src, node) != fn:
                continue
            decl = node.child_by_field_name("declarator")
            plist = None
            cur = decl
            while cur is not None:
                if cur.type == "function_declarator":
                    plist = cur.child_by_field_name("parameters")
                    break
                cur = cur.child_by_field_name("declarator")
            if plist is None:
                return []
            params: list[dict] = []
            idx = 0
            for child in plist.named_children:
                if child.type != "parameter_declaration":
                    continue
                name = _declarator_name(src, child.child_by_field_name("declarator"))
                typ = _type_prefix(src, child)
                raw = collapse_ws(_node_text(src, child))
                if raw == "void" or not name:
                    idx += 1
                    continue
                params.append(
                    {
                        "index": idx,
                        "name": name,
                        "declared_type": typ or None,
                        "raw": raw[:160],
                    }
                )
                idx += 1
            return params
        return []

    def extract_call_argument_expression(
        self, callee: str, call_line: int, arg_index: int
    ) -> Optional[dict]:
        """Extract the ``arg_index``-th argument of a call_expression near ``call_line``."""
        fn = (callee or "").strip()
        if not fn or arg_index < 0:
            return None
        for ln in range(max(1, call_line - 2), call_line + 13):
            for rec in self._calls_by_line.get(ln, []):
                if rec["line"] > call_line + 10:
                    continue
                hit = rec["function"] == fn or fn in rec["function_text"]
                if not hit:
                    # indirect call: table[i](...) — still accept if this line matches
                    if rec["function"] and rec["function"] != fn and "[" not in rec["function_text"]:
                        continue
                    if not rec["args"]:
                        continue
                args = rec["args"]
                if arg_index >= len(args):
                    continue
                expr = args[arg_index]
                idents = [x for x in identifiers_from_text(expr) if x != fn]
                return {
                    "argument_expression": expr[:240],
                    "argument_identifiers": idents[:12],
                    "arg_index": arg_index,
                    "call_text": rec["call_text"],
                    "approx_line": rec["line"],
                }
        # Indirect / function-pointer: any call on this line with enough args.
        for rec in self._calls_by_line.get(call_line, []):
            if arg_index < len(rec["args"]):
                expr = rec["args"][arg_index]
                idents = [x for x in identifiers_from_text(expr) if x != fn]
                return {
                    "argument_expression": expr[:240],
                    "argument_identifiers": idents[:12],
                    "arg_index": arg_index,
                    "call_text": rec["call_text"],
                    "approx_line": rec["line"],
                }
        return None

    def _function_end_from(self, func_line: int, max_span: int = 1200) -> int:
        info = self._functions_by_start.get(func_line)
        if info:
            return min(info[2], func_line + max_span)
        n = len(self.source_lines) - 1
        return min(n, func_line + max_span)

    def _looks_like_function_header(self, line_no: int) -> Optional[str]:
        info = self._functions_by_start.get(line_no)
        if info:
            return info[0]
        return None

    def find_enclosing_function(
        self, line: int, max_lookback: int = 800
    ) -> tuple[Optional[str], int, int]:
        """Return (name, start_line, end_line) that actually contains ``line``."""
        n = len(self.source_lines) - 1
        if not self._function_spans:
            return None, max(1, line - 20), min(n, line + 20)
        i = bisect.bisect_right(self._fn_start_keys, line) - 1
        while i >= 0:
            start, end, name = self._function_spans[i]
            if start < line - max_lookback and end < line:
                break
            if start <= line <= end:
                return name, start, end
            i -= 1
        return None, max(1, line - 20), min(n, line + 20)

    def _store_decl(self, rec: DeclarationRecord, *, prefer_size: bool = True) -> None:
        self.declarations_all[rec.name].append(rec)
        existing = self.declarations.get(rec.name)
        if existing is None:
            self.declarations[rec.name] = rec
            return
        if not prefer_size:
            return
        ex = existing.array_size
        nw = rec.array_size
        if (ex is None or ex == 0) and nw is not None and nw > 0:
            self.declarations[rec.name] = rec
        elif ex is None and nw is not None:
            self.declarations[rec.name] = rec
        elif existing.parse_note and not rec.parse_note and nw is not None:
            self.declarations[rec.name] = rec

    def _decl_from_array(
        self,
        name: str,
        type_text: str,
        array_node: Node,
        line: int,
        *,
        is_pointer: bool,
        init_node: Optional[Node] = None,
        raw: str = "",
    ) -> DeclarationRecord:
        size, note = _eval_size_node(
            self.source_bytes, array_node.child_by_field_name("size")
        )
        if size is None and init_node is not None:
            counted = _count_initializer_elements(init_node)
            if counted:
                size = counted
                note = "implicit size from initializer_list"
        extra = array_node.child_by_field_name("declarator")
        # Multi-dimensional: nested array_declarator
        more = False
        cur = extra
        while cur is not None:
            if cur.type == "array_declarator":
                more = True
                break
            cur = cur.child_by_field_name("declarator") if cur.type != "identifier" else None
        parse_note: Optional[str] = note or None
        if more:
            extra_note = (
                "Multi-dimensional declarator; only first dimension token captured; "
                "further dimensions not resolved."
            )
            parse_note = f"{extra_note} {note}".strip() if note else extra_note
        declared_type = type_text or "unknown"
        if size is not None and not parse_note:
            declared_type = f"{declared_type}[{size}]".strip()
        elif array_node.child_by_field_name("size") is not None:
            tok = collapse_ws(
                _node_text(self.source_bytes, array_node.child_by_field_name("size"))
            )
            declared_type = f"{declared_type}[{tok}]".strip()
        loc = self.synth_location(line, 1)
        return DeclarationRecord(
            name=name,
            raw_declaration_text=(raw or "")[:240],
            array_size=size,
            declared_type=declared_type,
            is_pointer=is_pointer,
            location=loc,
            parse_note=parse_note,
        )

    def _harvest_field_declaration(self, node: Node, type_name: str) -> None:
        src = self.source_bytes
        type_text = _type_prefix(src, node)
        raw = collapse_ws(_node_text(src, node))
        is_ptr_type = "*" in type_text
        for child in node.named_children:
            if child.type in {"primitive_type", "type_identifier", "sized_type_specifier", "struct_specifier", "union_specifier", "enum_specifier"}:
                continue
            array_node = _find_array_declarator(child)
            name = _declarator_name(src, child)
            if not name and child.type == "field_identifier":
                name = _node_text(src, child)
            if not name:
                continue
            line = child.start_point[0] + 1
            is_ptr = is_ptr_type or child.type == "pointer_declarator"
            if array_node is not None:
                rec = self._decl_from_array(
                    name,
                    type_text,
                    array_node,
                    line,
                    is_pointer=is_ptr,
                    raw=raw,
                )
                if type_name:
                    typed = DeclarationRecord(
                        name=f"{type_name}.{name}",
                        raw_declaration_text=rec.raw_declaration_text,
                        array_size=rec.array_size,
                        declared_type=rec.declared_type,
                        is_pointer=rec.is_pointer,
                        location=rec.location,
                        parse_note=rec.parse_note,
                    )
                    self.type_array_fields[type_name].append(typed)
                    self._store_decl(typed)
                self._store_decl(rec)
            elif type_name:
                rec = DeclarationRecord(
                    name=f"{type_name}.{name}",
                    raw_declaration_text=raw[:240],
                    array_size=None,
                    declared_type=type_text,
                    is_pointer=is_ptr,
                    location=self.synth_location(line, 1),
                    parse_note=None,
                )
                self._store_decl(rec, prefer_size=True)

    def _struct_tag(self, node: Node) -> str:
        for child in node.named_children:
            if child.type in {"type_identifier", "identifier"}:
                return _node_text(self.source_bytes, child)
        return ""

    def _harvest_struct(self, node: Node, typedef_name: str = "") -> None:
        tag = self._struct_tag(node)
        body = None
        for child in node.named_children:
            if child.type == "field_declaration_list":
                body = child
                break
        if body is None:
            return
        keys = [k for k in (typedef_name, tag) if k]
        if typedef_name and tag and typedef_name != tag:
            keys = [typedef_name, tag]
        if not keys:
            return
        for field_n in body.named_children:
            if field_n.type == "field_declaration":
                for key in keys:
                    self._harvest_field_declaration(field_n, key)

    def _harvest_declaration_node(self, node: Node) -> None:
        src = self.source_bytes
        type_text = _type_prefix(src, node)
        raw = collapse_ws(_node_text(src, node))
        simple_type = type_text.split()[-1].replace("*", "") if type_text else ""
        is_ptr_type = "*" in type_text
        for child in node.named_children:
            init_n = None
            decl_n = child
            if child.type == "init_declarator":
                decl_n = child.child_by_field_name("declarator")
                init_n = child.child_by_field_name("value")
            if decl_n is None:
                continue
            if decl_n.type in {
                "primitive_type",
                "type_identifier",
                "sized_type_specifier",
                "struct_specifier",
                "union_specifier",
                "storage_class_specifier",
                "type_qualifier",
            }:
                continue
            name = _declarator_name(src, decl_n)
            if not name:
                continue
            line = decl_n.start_point[0] + 1
            array_node = _find_array_declarator(decl_n)
            is_ptr = is_ptr_type or decl_n.type == "pointer_declarator"
            if array_node is not None:
                rec = self._decl_from_array(
                    name,
                    type_text,
                    array_node,
                    line,
                    is_pointer=is_ptr,
                    init_node=init_n,
                    raw=raw,
                )
                self._store_decl(rec)
                continue
            if is_ptr:
                rec = DeclarationRecord(
                    name=name,
                    raw_declaration_text=raw[:240],
                    array_size=None,
                    declared_type=type_text,
                    is_pointer=True,
                    location=self.synth_location(line, 1),
                    parse_note=None,
                )
                if name not in self.declarations:
                    self._store_decl(rec)
            if simple_type and ident_token(simple_type) and ident_token(name):
                if name not in self.object_types:
                    self.object_types[name] = simple_type

    def _build_declaration_index(self) -> None:
        """Walk the C AST for array/pointer/typedef field declarations."""
        if self.source_tree is None:
            return
        for node in _iter_nodes(self.source_tree.root_node):
            if node.type in {"struct_specifier", "union_specifier"}:
                self._harvest_struct(node)
            elif node.type == "type_definition":
                spec = None
                typedef_name = ""
                for child in node.named_children:
                    if child.type in {"struct_specifier", "union_specifier"}:
                        spec = child
                    elif child.type == "type_identifier":
                        typedef_name = _node_text(self.source_bytes, child)
                    elif "declarator" in child.type:
                        typedef_name = _declarator_name(self.source_bytes, child) or typedef_name
                if spec is not None:
                    self._harvest_struct(spec, typedef_name)
        for node in _iter_nodes(self.source_tree.root_node):
            if node.type == "declaration":
                self._harvest_declaration_node(node)
        # Copy typedef array fields onto object instances: Type var;
        for var, typ in list(self.object_types.items()):
            for frec in self.type_array_fields.get(typ, []):
                field = frec.name.split(".")[-1]
                qname = f"{var}.{field}"
                if qname in self.declarations and self.declarations[qname].array_size:
                    continue
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
                self._store_decl(qrec)

    def get_declaration(self, symbol_name: str) -> Optional[DeclarationRecord]:
        key = (symbol_name or "").strip().split("@", 1)[0].strip('"')
        hit = self.declarations.get(key)
        if hit is not None:
            return hit
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

        typ = self.object_types.get(key)
        if typ:
            fields = list(self.type_array_fields.get(typ, []))
            for name, drec in self.declarations.items():
                if name.startswith(key + ".") and drec.array_size is not None:
                    if not any(
                        f.name == name or f.name.endswith("." + name.split(".")[-1])
                        for f in fields
                    ):
                        fields.append(drec)
            if not fields:
                seen_member: set[str] = set()
                for name in self.declarations:
                    if not name.startswith(key + "."):
                        continue
                    member = name.split(".", 1)[1]
                    if not member or member in seen_member:
                        continue
                    seen_member.add(member)
                    fields.append(self.declarations[name])
                for df_name in self.data_flow_by_variable:
                    if not df_name.startswith(key + "."):
                        continue
                    member = df_name.split(".", 1)[1].split("@", 1)[0].strip('"')
                    if not member or member in seen_member:
                        continue
                    seen_member.add(member)
                    fields.append(
                        DeclarationRecord(
                            name=f"{key}.{member}",
                            raw_declaration_text="",
                            array_size=None,
                            declared_type="unknown",
                            is_pointer=False,
                            location="",
                            parse_note=(
                                "Member inferred from data-flow keys; declaration "
                                "line not resolved."
                            ),
                        )
                    )
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
        full = (variable_name or "").strip().split("@", 1)[0].strip('"')
        tail = full.split(".")[-1] if full else ""
        parent = full.rsplit(".", 1)[0] if "." in full else None
        return {"full": full, "tail": tail, "parent": parent}

    def get_prior_writes(
        self,
        variable_name: str,
        before_location: str,
        *,
        function_scope: Optional[str] = None,
    ) -> list[DataFlowRecord]:
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
                        line_txt = (
                            self.source_lines[rec.line]
                            if rec.line and rec.line < len(self.source_lines)
                            else ""
                        )
                        field = require_field_in_text
                        if not (
                            f".{field}" in line_txt
                            or f"->{field}" in line_txt
                            or token_in_text(line_txt, field)
                        ):
                            continue
                        if not (
                            "=" in line_txt
                            or f"{field}++" in line_txt
                            or f"{field}--" in line_txt
                            or f".{field}++" in line_txt
                            or f"->{field}++" in line_txt
                        ):
                            if field not in (expr or line_txt):
                                continue
                    sig = (rec.variable, rec.function, rec.location, rec.line, rec.process)
                    if sig in seen:
                        continue
                    seen.add(sig)
                    out.append(rec)

        keys = {bare}
        for k in self.data_flow_by_variable:
            if k == bare or k.startswith(bare + "@"):
                keys.add(k)
        tail = forms["tail"]
        if tail and tail != bare:
            keys.add(tail)
            for k in self.data_flow_by_variable:
                if k == tail or k.startswith(tail + "@"):
                    keys.add(k)
        _add_from_keys(keys)

        parent = forms["parent"]
        if parent and tail:
            parent_keys = {parent}
            for k in self.data_flow_by_variable:
                if k == parent or k.startswith(parent + "@"):
                    parent_keys.add(k)
            _add_from_keys(parent_keys, require_field_in_text=tail)

        out.sort(key=lambda r: (r.line or 0, r.function, r.location))
        return out

    def extract_assignment_text(self, line: int, variable_name: str) -> str:
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

    def _lhs_matches(self, lhs: str, full: str, tail: str) -> bool:
        compact = collapse_ws(lhs).replace("->", ".").replace(" ", "")
        if not compact:
            return False
        if full and (compact == full.replace("->", ".") or compact.endswith("." + full.split(".")[-1])):
            if compact == full.replace(" ", "") or compact.endswith("." + tail) or compact == tail:
                return True
        if tail and (compact == tail or compact.endswith("." + tail)):
            return True
        if full and "." in full:
            root, field = full.rsplit(".", 1)
            if compact == f"{root}.{field}" or compact.endswith(f".{field}"):
                return True
        return False

    def find_source_assignments(
        self,
        variable_name: str,
        *,
        before_line: int,
        function_start: int,
        function_end: int,
    ) -> list[dict]:
        forms = self._symbol_name_forms(variable_name)
        full = forms["full"] or ""
        tail = forms["tail"] or full
        hits: list[dict] = []
        lo = max(1, function_start)
        hi = min(before_line - 1, function_end, len(self.source_lines) - 1)
        for rec in self._assigns:
            ln = rec["line"]
            if ln < lo:
                continue
            if ln > hi:
                break
            if not self._lhs_matches(rec["lhs"], full, tail):
                continue
            text = rec["text"] or self.source_lines[ln].strip()[:300]
            hits.append(
                {
                    "function": None,
                    "location": self.synth_location(ln, max(1, len(self.source_lines[ln]))),
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
        forms = self._symbol_name_forms(variable_name)
        full = forms["full"] or ""
        tail = forms["tail"] or full
        if not tail:
            return []
        if tail in _COMMON_INDEX_NAMES or (full in _COMMON_INDEX_NAMES):
            return []
        if len(tail) <= 2:
            return []
        hits: list[dict] = []
        hi = min(before_line - 1, len(self.source_lines) - 1)
        for rec in reversed(self._assigns):
            ln = rec["line"]
            if ln > hi or ln < 1:
                continue
            if not self._lhs_matches(rec["lhs"], full, tail):
                continue
            fn, _, _ = self.find_enclosing_function(ln)
            text = rec["text"] or self.source_lines[ln].strip()[:300]
            hits.append(
                {
                    "function": fn,
                    "location": self.synth_location(ln, max(1, len(self.source_lines[ln]))),
                    "line": ln,
                    "process": None,
                    "assigned_expression_text": text[:300],
                    "call_depth": None,
                    "source": "source_scan_global",
                }
            )
            if len(hits) >= max_hits:
                break
        hits.reverse()
        return hits

    def find_condition_guards(
        self,
        variable_name: str,
        from_location: str,
        *,
        max_depth: int = 8,
        max_guards: int = 40,
    ) -> dict:
        bare = (variable_name or "").strip().split("@", 1)[0].strip('"')
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
        queue: list[tuple[Optional[str], int, Optional[int], Optional[int], str]] = [
            (start_fn, 0, fn_start, fn_end, from_location)
        ]
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
                        edge.call_site or self.synth_location(c_line),
                    )
                )

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
        for rec in self._guards:
            ln = rec["line"]
            if ln < lo:
                continue
            if ln > hi:
                break
            cond = rec["condition"]
            if not any(token_in_text(cond, tok) for tok in tokens):
                continue
            hits.append(
                {
                    "condition_text": cond,
                    "kind": rec["kind"],
                    "function": function,
                    "location": self.synth_location(
                        ln, max(1, len(self.source_lines[ln]) if ln < len(self.source_lines) else 1)
                    ),
                    "line": ln,
                    "call_depth": call_depth,
                }
            )
        return hits

    def subscripts_on_line(self, line: int) -> list[dict[str, Any]]:
        if line < 1 or line >= len(self.source_lines):
            return []
        return subscripts_from_text(self.source_lines[line])
