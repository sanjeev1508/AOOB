"""LangGraph ReAct agent over local Ollama or NVIDIA NIM for AOOB investigation."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Annotated, Any, Literal, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.utils import trim_messages
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from aoob_agent.data_store import DataStore
from aoob_agent.report import AlarmInvestigationReport
from aoob_agent.tools import TOOLS, bind_store, extract_index_operands

DEFAULT_NVIDIA_MODEL = "meta/llama-3.1-70b-instruct"
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"

# Soft ceiling so multi-tool investigations (decl + index slice + guards + …) can finish.
_MAX_TOOL_MESSAGES = 12

# Index-origin retrieval tools — process gate only (never forces true/false).
_ORIGIN_TOOLS = frozenset(
    {
        "get_backward_slice",
        "get_variable_manipulation_sequence",
        "get_all_writes_to_symbol",
        "get_call_site_arguments",
    }
)

INDEX_ORIGIN_NUDGE = """You have not yet traced a real INDEX operand.

Required next step (retrieval only — not a verdict):
- Read index_operand_candidates from the latest get_affected_symbols result.
- Call get_backward_slice (preferred) on ONE of those candidates (e.g. checkword,
  Struct.Field, idxMRPChnl_u16) — NOT indexed_objects (the array), NOT sentinel
  pointers, NOT the table name.
- Prefer get_declaration_bounds on indexed_objects (e.g. Obj.raw) for the bound.
- For Struct.Field indexes, pass the qualified name or the field; if the slice is
  empty, call get_all_writes_to_symbol on the same name.
- If the alarm snippet shows shifts/masks on the index, call
  get_index_expression_structure on that expression before treating Astrée [lo,hi]
  as the final index range.
- Do NOT finalize yet. Do NOT argue array_size < Astrée abstract hi as proof of
  classification "true".
- After a correct index slice returns, you may continue or finish; classification
  remains yours.
"""

CALL_SITE_NUDGE = """Index looks like a function PARAMETER (empty local writes).

Required next step (retrieval only — not a verdict):
- Read call_site_arguments / next_slice_candidates from the latest
  get_backward_slice (or call get_call_site_arguments on the enclosing function +
  parameter name).
- Call get_backward_slice on ONE argument identifier from next_slice_candidates
  (the expression callers actually pass), NOT the parameter name again.
- Also call get_condition_guards on that argument identifier at a caller
  call_site location when available.
- If parameter_type_width.abstract_max matches Astrée's index hi (e.g. 255 for
  uint8), treat that as type-domain over-approx unless call-site evidence shows a
  concrete oversized value can reach the access.
- Do NOT finalize as classification "true" / medium from Astrée hi vs array_size
  alone while call-site arguments remain unexamined.
"""

GUARD_NUDGE = """Before finishing an OOB investigation, retrieve condition guards.

Required next step (retrieval only — not a verdict):
- Call get_condition_guards on the INDEX operand you are tracing (prefer a
  call-site argument identifier if the index is a parameter) at the alarm or
  call-site location. This covers if / for / while / switch / assert.
- If guards name sentinels or limits, call resolve_symbolic_constant on them.
- Do NOT claim "value is not checked" if a guard already appears in the snippet
  or in get_condition_guards output — acknowledge it.
- Then you may finalize; classification remains yours.
"""

NEXT_HOP_NUDGE = """The last origin tool named a specific identifier as the next value to retrieve.

Required next step (retrieval only — not a verdict):
- Call get_backward_slice (preferred) or get_all_writes_to_symbol / get_condition_guards
  on THAT identifier — the one named in assigned_expression_text or
  next_slice_candidates — not a different symbol.
- If the tool returns found=false, empty df_keys_matched, or a parse_note that the
  name is a macro / missing DF row / unresolved, stop retrying that name.
- Do NOT finalize while that identifier is still open and not explicitly unresolvable.
"""

SYSTEM_PROMPT = """You are an Astrée out-of-bounds (AOOB) alarm investigation agent.

You receive an alarm Order id. You MUST use your tools to gather evidence — do not
invent source locations, variables, declarations, or value origins. There are no
hardcoded triage rules in code: YOU decide tools, arguments, interpretation,
classification (true / false / review), comment, and confidence from evidence alone.

Available tools (retrieval-only — they never classify bug/no-bug):
1) get_affected_symbols — alarm metadata + DF + source at the site.
   alarm.message (when present) is Astrée's diagnostic text, e.g.
   "[0, 7] not included in array index range [0, 1]". That is NOT a human label
   and NOT a concrete runtime index value (see "Abstract intervals" below).
   Use indexed_objects for declaration lookups; index_operand_candidates for slices.
2) get_function_snippet — enclosing function + CF edges
3) get_variable_manipulation_sequence — ordered DF read/write events
4) get_declaration_bounds — declaration / literal array_size / parse_note /
   nested_array_fields for union/struct objects. Prefer indexed_objects names
   (e.g. Obj.raw) when the access is Obj.raw[i].
5) get_backward_slice — earlier writes + CF callers for ONE symbol at a location.
   Pass the INDEX / pointer operand. If parameter_note / call_site_arguments
   appear, the index origin is at callers — follow next_slice_candidates.
6) get_call_site_arguments — caller argument expressions for a callee parameter
   (use when the index is a parameter with no local writes).
7) get_condition_guards — if/switch/assert/ternary text mentioning a variable
   before the access (and in callers). Raw conditions only — you judge relevance.
8) resolve_symbolic_constant — literal value of an enum member / #define when
   resolvable. Prefer this over inferring meaning from the identifier spelling.
9) get_index_expression_structure — parse shifts/masks/arithmetic/fields in an
   index expression (structure only; no range evaluation).
10) get_all_writes_to_symbol — flat exhaustive DF write listing (cross-check when
   a backward slice is truncated). Reachability still needs slice/CF reasoning.

Abstract intervals (critical):
- Astrée messages like "[lo, hi] not included in [a, b]" report ABSTRACT DOMAIN
  bounds (over-approximations), not proof that the index equals hi at runtime.
- Never argue "array size is N which is less than hi, therefore true bug" from
  the message alone. hi is an analyzer upper bound, not a measured index.
- Always separate: (A) declared object size, (B) what concrete/source evidence
  says about the index/pointer origin, (C) what Astrée's abstract interval says.
  Classification must weigh all three; (C) alone is insufficient for "true".
- Type-domain pattern: if the index parameter's type width max equals Astrée hi
  (e.g. uint8 / uint8_least → 255) and the array is much smaller, that mismatch
  is a common over-approx shape unless call-site evidence shows a value ≥
  array size can actually reach the access. That pattern is classification
  "review" (or "false" only if tools also show a bound/guard/transform that
  keeps the access inside the object). Never "true" at medium/high on this
  pattern alone.

Index-expression arithmetic (critical):
- Before comparing an Astrée [lo, hi] to an array bound, inspect the actual index
  expression in the snippet. Shifts, masks, offsets, casts, and field extracts
  change the effective index range.
- Astrée's [lo, hi] often describes an operand (or abstract field), not necessarily
  the final subscript after transforms. Example: if the access is
  ``arr[(idDFC.id) >> 4u]``, do NOT treat a raw [0, 255] on id as proof that the
  subscript can be 255 — call get_index_expression_structure and APPLY the
  arithmetic: a right-shift by N shrinks an unsigned operand's max by about 2^N
  (e.g. 8-bit value >> 4 ⇒ at most 15). Compare the *transformed* range to the
  bound before concluding "true".
- Mention the transform explicitly in comment / index_origin_summary when present.

Loop-exit / loop-counter intervals:
- Astrée intervals for loop counters often include the loop-*exit* value (the
  bound that fails the loop test), even when the array access sits inside the
  loop body and never executes at that value.
- Example pattern: counter runs while ``i < N`` / ``i < numBlocks`` with access
  ``arr[i]`` and Astrée reports ``[0, N]`` vs declared ``[0, N-1]``. That alone
  is NOT proof of a reachable OOB — call get_condition_guards (for/while/if) and
  reason whether the access is protected by the loop/guard condition.
- Prefer classification "review" (not "false") when the only alarming signal
  is a one-past-end loop-exit interval, unless a loop/index guard in the tools
  clearly keeps the access inside the object ("false") or a reachable OOB is
  shown ("true").

Sentinel / invalid constants:
- A write that initializes a variable to a named "invalid" / "idle" / default
  constant is NOT by itself evidence of a real OOB bug.
- ALWAYS call get_condition_guards on the index before concluding the access is
  unbounded / classification "true". Resolve named constants with
  resolve_symbolic_constant — do not rely on identifier spelling alone.
- If you cannot find a guard and cannot otherwise show the sentinel is reachable
  as an index at the access, do not exceed confidence "low".

Parameter / call-site navigation (critical):
- Empty writes_found on a parameter is EXPECTED. Do not treat it as "unchecked
  unbounded index".
- When call_site_arguments / next_slice_candidates are present, you MUST attempt
  at least one slice (or all-writes) on a caller argument identifier before
  finishing — that is the real index origin path.
- Call-site expressions like ``startIdx + BufIdx`` / config table lookups are
  evidence of a constrained constructed index; without a concrete oversized
  index; without a concrete oversized write, prefer classification "review"
  (not "true"). Use "false" only if tools also show a bound, guard, or
  transform that keeps the constructed index inside the object.

Unresolved evidence:
- If evidence_completeness is "not traced", confidence MUST be "low". Do NOT use
  medium/high.
- Do NOT invent a strong "true" from the Astrée string alone.
- Do NOT invent a strong "false" that invents facts not in tools.
- Incomplete origin (not traced / parameter with no call-site follow-up) is
  classification "review", not a false-positive finding.
- You MAY classify "false" only when tools show the access is actually
  constrained (guard, in-range init that is the value used, shift/mask that
  fits, constructed index inside the object).
- A traced safe-looking initializer (e.g. init=0) plus leftover Astrée
  type-max [0, 255] is still "review" unless a guard/transform proves the
  access cannot exceed the bound. Do not use "false" as a safe harbor.
- classification "true" at medium/high only when tool evidence shows a
  reachable too-large index/pointer, not the abstract interval alone.

Investigation discipline:
- For OOB array/pointer alarms, gather BOTH container bounds AND index/pointer
  origin. Prefer get_declaration_bounds on indexed_objects AND get_backward_slice
  on index_operand_candidates (then call-site args if parameter).
- When the snippet shows a non-trivial index expression, call
  get_index_expression_structure and use the arithmetic consequence.
- Before classification "true", call get_condition_guards on the index (and
  resolve_symbolic_constant on any named limits/sentinels in those guards).
  If a guard is already visible in the function snippet, you MUST acknowledge it
  in the comment — never claim "value is not checked" when an if/for/while
  check is in the tool output.
- Do not stop after message + declaration only. If you have not attempted an
  index-origin tool, keep going unless you can state a concrete reason that
  further tools cannot help.
- When a slice shows a local initializer (e.g. checkword = 0), prior sanitizing
  writes, or only unresolved/truncated paths, say so explicitly.
- Slicing the array name itself often yields empty writes_found for const
  tables; that is not evidence about the index. Re-slice the index symbol /
  Struct.Field.
- If get_backward_slice is truncated or empty for a field, use
  get_all_writes_to_symbol as a cross-check.
- declared_size null / missing / suppressed 0 means unresolved size — NOT proof
  the object has zero extent. Never argue "size 0 ⇒ true OOB".

Classification & confidence (you author these; tools/CSV never supply them):
- classification "true" = a real reachable OOB bug, supported by tool facts
  beyond Astrée [lo, hi] vs declared size.
- classification "false" = a false positive: tools show the access is bounded
  or the alarming interval cannot reach the site (guard, transform, constrained
  origin). Incomplete evidence is NOT "false".
- classification "review" = you cannot tell reachable OOB from analyzer FP.
  Use this when origin is untraced/partial, call sites were not followed, OR
  origin is traced but the leftover Astrée interval is still unexplained
  (init=0 + [0,255] type-max is this case — do not dump it to "false").
- Never write a comment of the form "array size N is less than Astrée hi" as
  proof of classification "true". That comparison misreads abstract domains.
- Prefer strong "true" (medium/high) only when evidence beyond the bare Astrée
  string supports that a too-large index/pointer can actually arise.
- confidence "high" only when index/pointer origin is meaningfully traced AND
  consistent with your classification. If evidence_completeness is
  "not traced", classification should be "review" and confidence "low".
  If "partially traced", prefer "review" or low/medium — not a confident false.
- evidence_completeness "fully traced" only if you actually followed the
  index/pointer to a coherent origin story from tool outputs — not merely
  because you called three tools. Parameter with only empty local writes =
  "not traced" or "partially traced" (if call-site args were examined).

Report fields: include classification, comment, astree_message (copy
alarm.message), bounds_evidence.index_origin_summary, confidence, tools_used.

Do NOT paste a finished JSON report in ordinary assistant turns. Keep using
tools until done; the dedicated report step builds the structured object.
"""

REPORT_PROMPT = """Based ONLY on the tool results provided, produce the final
investigation report. Do not invent facts not present in tool outputs.
Keep function_snippet to the alarm neighborhood.

Return a single JSON object matching the schema — no prose wrappers, no
markdown, no `field = value` lines.

Astrée alarm.message intervals are ABSTRACT over-approximations. Do not treat
the upper bound as a concrete index when writing classification/comment.
Forbidden rationale: "array_size < Astrée hi ⇒ true bug".
If a backward slice shows a concrete initializer or constrained writes that
conflict with reading the abstract hi as a runtime index, reflect that in
classification/comment/confidence.

Index transforms: if tools show shifts/masks/offsets on the index operand,
your comment must account for them — apply the arithmetic (>> N shrinks max)
and do not compare the raw Astrée operand interval to the array bound as if no
transform existed.

Loop guards: if the index is a loop counter and Astrée's hi equals the loop
limit / array size, check get_condition_guards / snippet for the loop test —
do not treat the exit value as a value that reaches an in-loop access.

Parameters: empty local writes + call_site_arguments showing constructed indices
(config start + buffer id, etc.) without an oversized concrete value ⇒
classification "review", not "true". "false" only if a bound/guard/transform
is in the tools. If parameter_type_width.abstract_max equals Astrée hi,
that is type-domain over-approx — not proof of true, and not by itself proof
of false.

Sentinels: if tools show a named invalid/default constant write, do not treat
that alone as proof of "true". Cite guards (or their absence) and any resolved
constant value. Call get_condition_guards before "true".

Unresolved: if evidence_completeness is "not traced", classification MUST be
"review" and confidence "low". Do not use "false" as a default when the only
signal is Astrée abstract vs declared size. A traced init=0 with leftover
[0, type_max] is "review" unless a guard proves the access stays in-bound.

If evidence_completeness is "fully traced" and tools already returned a
declaration bound, a condition guard, or an index transform (shift/mask/offset),
classification must be "true" or "false" — not "review". Unfamiliar helper
names are not a reason to hedge past that evidence.

Never treat declared_size 0 / null as proof of OOB.

Required agent-authored fields:
- classification: "true" (real bug), "false" (false positive), or "review"
  (cannot tell). Incomplete evidence is "review", not "false".
- comment: short rationale that cites tool facts (not the Astrée string alone).
- astree_message: copy alarm.message from tools when present.
- confidence: "low" | "medium" | "high" — calibrate to how well index origin
  was traced and how consistent that origin is with your classification.
- bounds_evidence.evidence_completeness: "fully traced" | "partially traced" |
  "not traced" — "fully traced" only if index/pointer origin was actually
  reconstructed from tools.
- bounds_evidence.index_origin_summary: what tools showed about the index
  (including "not traced" / unresolved / local init = 0 / shifts / guards /
  call-site argument expressions).

List every tool you relied on in tools_used.
"""


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    order_id: int
    report: Optional[dict]
    tool_rounds: int
    origin_nudges: int
    call_site_nudges: int
    guard_nudges: int
    next_hop_nudges: int
    force_retries: int


def load_env(project_root: Path) -> None:
    load_dotenv(project_root / ".env", override=False)


def ollama_model_name() -> str:
    """Model tag from OLLAMA_LOCAL_PATH or OLLAMA_MODEL (e.g. qwen2.5:7b)."""
    return (os.getenv("OLLAMA_LOCAL_PATH") or os.getenv("OLLAMA_MODEL") or "").strip()


def llm_backend_override() -> Optional[str]:
    """AOOB_LLM_BACKEND=nvidia|ollama forces a backend even if both are configured."""
    raw = (os.getenv("AOOB_LLM_BACKEND") or "").strip().lower()
    return raw if raw in {"nvidia", "ollama"} else None


def using_ollama() -> bool:
    override = llm_backend_override()
    if override == "nvidia":
        return False
    if override == "ollama":
        if not ollama_model_name():
            raise RuntimeError(
                "AOOB_LLM_BACKEND=ollama but OLLAMA_LOCAL_PATH / OLLAMA_MODEL is unset."
            )
        return True
    return bool(ollama_model_name())


def resolve_model_name(model: Optional[str] = None) -> str:
    if model:
        return model
    if using_ollama():
        return ollama_model_name()
    return os.getenv("NVIDIA_MODEL", DEFAULT_NVIDIA_MODEL) or DEFAULT_NVIDIA_MODEL


def llm_backend_label() -> str:
    if using_ollama():
        return f"Ollama ({resolve_model_name()})"
    return f"NVIDIA ({resolve_model_name()})"


def build_llm(model: Optional[str] = None) -> Any:
    name = resolve_model_name(model)
    timeout = float(os.getenv("OLLAMA_TIMEOUT" if using_ollama() else "NVIDIA_TIMEOUT", "300"))
    if using_ollama():
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:
            raise RuntimeError(
                "OLLAMA_LOCAL_PATH is set but langchain-ollama is not installed. "
                "Run: py -3.13 -m pip install langchain-ollama"
            ) from exc
        base_url = (
            os.getenv("OLLAMA_HOST")
            or os.getenv("OLLAMA_BASE_URL")
            or DEFAULT_OLLAMA_HOST
        ).rstrip("/")
        num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "16384"))
        _log(
            f"[llm] backend=ollama model={name} base_url={base_url} "
            f"num_ctx={num_ctx} timeout={timeout}s"
        )
        return ChatOllama(
            model=name,
            base_url=base_url,
            temperature=0.1,
            num_ctx=num_ctx,
            num_predict=2048,
            client_kwargs={"timeout": timeout},
        )

    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "NVIDIA_API_KEY missing (and OLLAMA_LOCAL_PATH is unset). "
            "Set one of them in .env."
        )
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    _log(f"[llm] backend=nvidia model={name} timeout={timeout}s")
    return ChatNVIDIA(
        model=name,
        api_key=api_key,
        temperature=0.1,
        max_tokens=2048,
        timeout=timeout,
    )


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _count_tool_messages(messages: list) -> int:
    return sum(1 for m in messages if isinstance(m, ToolMessage))


def _tool_names_used(messages: list) -> set[str]:
    names: set[str] = set()
    for m in messages:
        if isinstance(m, ToolMessage) and getattr(m, "name", None):
            names.add(str(m.name))
        if isinstance(m, AIMessage):
            for tc in getattr(m, "tool_calls", None) or []:
                if tc.get("name"):
                    names.add(str(tc["name"]))
    return names


def _parse_tool_json(content: str) -> Optional[dict]:
    try:
        data = json.loads(content)
        return data if isinstance(data, dict) else None
    except Exception:  # noqa: BLE001
        # Truncated tool payloads may be incomplete JSON — ignore.
        return None


def _index_candidates_from_messages(messages: list) -> list[str]:
    """Pull index_operand_candidates from get_affected_symbols tool results."""
    found: list[str] = []
    seen: set[str] = set()
    for m in messages:
        if not isinstance(m, ToolMessage):
            continue
        if getattr(m, "name", None) != "get_affected_symbols":
            continue
        data = _parse_tool_json(str(m.content))
        if not data:
            continue
        for c in data.get("index_operand_candidates") or []:
            c = str(c).strip()
            if c and c not in seen:
                seen.add(c)
                found.append(c)
    return found


def _indexed_objects_from_messages(messages: list) -> set[str]:
    objs: set[str] = set()
    for m in messages:
        if not isinstance(m, ToolMessage):
            continue
        if getattr(m, "name", None) not in {
            "get_affected_symbols",
            "get_declaration_bounds",
        }:
            continue
        data = _parse_tool_json(str(m.content))
        if not data:
            continue
        for o in data.get("indexed_objects") or []:
            objs.add(str(o))
            objs.add(str(o).split(".")[-1])
        if data.get("symbol"):
            objs.add(str(data["symbol"]))
            objs.add(str(data["symbol"]).split(".")[-1])
        if data.get("found") and data.get("symbol"):
            objs.add(str(data["symbol"]))
    return objs


def _origin_slice_targets(messages: list) -> list[str]:
    """variable_name args used with index-origin tools."""
    targets: list[str] = []
    for m in messages:
        if not isinstance(m, AIMessage):
            continue
        for tc in getattr(m, "tool_calls", None) or []:
            if (tc.get("name") or "") not in _ORIGIN_TOOLS:
                continue
            args = tc.get("args") or {}
            name = args.get("variable_name") or args.get("symbol_name")
            if name:
                targets.append(str(name).strip().split("@", 1)[0].strip('"'))
            pname = args.get("parameter_name")
            if pname:
                targets.append(str(pname).strip().split("@", 1)[0].strip('"'))
    return targets


def _target_matches_candidate(target: str, candidates: list[str]) -> bool:
    t = target.strip()
    t_tail = t.split(".")[-1]
    for c in candidates:
        if t == c or t_tail == c or t == c.split(".")[-1] or t_tail == c.split(".")[-1]:
            return True
        if c.endswith("." + t) or c.endswith("." + t_tail):
            return True
    return False


def _has_useful_index_origin_tool(messages: list) -> bool:
    """True only if an origin tool targeted a real index candidate (not the array)."""
    targets = _origin_slice_targets(messages)
    if not targets:
        return False
    candidates = _index_candidates_from_messages(messages)
    indexed = _indexed_objects_from_messages(messages)
    if candidates:
        return any(_target_matches_candidate(t, candidates) for t in targets)
    # Fallback when parser found no candidates: any origin tool whose target is
    # not the indexed array/object.
    return any(t not in indexed and t.split(".")[-1] not in indexed for t in targets)


def _call_site_followup_info(messages: list) -> Optional[dict]:
    """If a slice reported a parameter with next_slice_candidates, return them."""
    for m in reversed(messages):
        if not isinstance(m, ToolMessage):
            continue
        if getattr(m, "name", None) not in {
            "get_backward_slice",
            "get_call_site_arguments",
        }:
            continue
        data = _parse_tool_json(str(m.content))
        if not data:
            continue
        csa = data.get("call_site_arguments")
        if isinstance(csa, dict):
            nxt = [str(x) for x in (csa.get("next_slice_candidates") or []) if x]
            if nxt or data.get("parameter_note"):
                return {
                    "next_slice_candidates": nxt,
                    "parameter": data.get("variable") or csa.get("parameter_name"),
                    "callee": data.get("start_function") or csa.get("callee"),
                }
        nxt = [str(x) for x in (data.get("next_slice_candidates") or []) if x]
        if nxt:
            return {
                "next_slice_candidates": nxt,
                "parameter": data.get("parameter_name") or data.get("variable"),
                "callee": data.get("callee") or data.get("start_function"),
            }
        if data.get("parameter_note"):
            return {
                "next_slice_candidates": [],
                "parameter": data.get("variable"),
                "callee": data.get("start_function"),
            }
    return None


def _has_call_site_argument_followup(messages: list) -> bool:
    """True if an origin tool targeted a call-site argument (not only the param)."""
    info = _call_site_followup_info(messages)
    if not info:
        return True  # no parameter path → nothing to follow up
    nxt = info.get("next_slice_candidates") or []
    if not nxt:
        return "get_call_site_arguments" in _tool_names_used(messages)
    targets = _origin_slice_targets(messages)
    param = str(info.get("parameter") or "")
    param_tail = param.split(".")[-1]
    for t in targets:
        if t == param or t.split(".")[-1] == param_tail:
            continue
        if _target_matches_candidate(t, nxt):
            return True
    return False


def _needs_call_site_followup(messages: list) -> bool:
    info = _call_site_followup_info(messages)
    if not info:
        return False
    return not _has_call_site_argument_followup(messages)


def _saw_parameter_note(messages: list) -> bool:
    for m in messages:
        if not isinstance(m, ToolMessage):
            continue
        data = _parse_tool_json(str(m.content))
        if data and (data.get("parameter_note") or data.get("call_site_arguments")):
            return True
    return False


def _is_oob_alarm(store: DataStore, order_id: int) -> bool:
    alarm = store.get_alarm(order_id)
    if alarm is None:
        return False
    cat = (alarm.category or "").lower()
    return "out-of-bound" in cat or "out of bound" in cat or "array_out_of_bounds" in cat


def _build_origin_nudge(messages: list) -> str:
    cands = _index_candidates_from_messages(messages)
    wrong = _origin_slice_targets(messages)
    lines = [INDEX_ORIGIN_NUDGE]
    if cands:
        lines.append(
            "index_operand_candidates from tools: "
            + ", ".join(cands[:12])
        )
    if wrong:
        lines.append(
            "You already sliced these (not accepted as index operands): "
            + ", ".join(wrong)
            + ". Slice a candidate instead."
        )
    return "\n".join(lines)


def _build_call_site_nudge(messages: list) -> str:
    info = _call_site_followup_info(messages) or {}
    lines = [CALL_SITE_NUDGE]
    nxt = info.get("next_slice_candidates") or []
    if nxt:
        lines.append(
            "next_slice_candidates from tools: " + ", ".join(nxt[:12])
        )
    if info.get("callee") and info.get("parameter"):
        lines.append(
            "Suggested get_call_site_arguments("
            f"callee_function={info['callee']!r}, "
            f"parameter_name={info['parameter']!r}) if args missing."
        )
    return "\n".join(lines)


def _build_guard_nudge(messages: list) -> str:
    info = _call_site_followup_info(messages) or {}
    nxt = info.get("next_slice_candidates") or []
    cands = nxt or _index_candidates_from_messages(messages)
    lines = [GUARD_NUDGE]
    if cands:
        lines.append(
            "Suggested variable_name for get_condition_guards: "
            + cands[0]
        )
    return "\n".join(lines)


def _parameter_path_open(messages: list) -> bool:
    return _needs_call_site_followup(messages) or (
        _saw_parameter_note(messages)
        and not _has_call_site_argument_followup(messages)
    )


def _guards_missing(messages: list) -> bool:
    return "get_condition_guards" not in _tool_names_used(messages)


_IDENT_TOKEN_RE = re.compile(r"\b([A-Za-z_]\w*)\b")
_SKIP_IDENT_TOKENS = frozenset(
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
    }
)
_UNRESOLVABLE_RE = re.compile(
    r"(no declaration|no #define|no data-flow|name must be a bare|"
    r"index miss|macro-built|could not resolve|value left unresolved|"
    r"no DF keys|unknown symbol|not a plain integer)",
    re.IGNORECASE,
)
_TRANSFORM_RE = re.compile(r">>|<<|(?<!&)&(?!&)|(?<!\|)\|(?!\|)|\^")
_CF_BOUNDARY_RE = re.compile(r"caller|max_depth|truncated", re.IGNORECASE)

_SITE_TOOL_NAMES = (
    "get_affected_symbols",
    "get_function_snippet",
    "get_declaration_bounds",
)
_HOP_TOOL_NAMES = (
    "get_backward_slice",
    "get_all_writes_to_symbol",
    "get_call_site_arguments",
)
_ORIGIN_CORE_NAMES = (
    "get_backward_slice",
    "get_all_writes_to_symbol",
    "get_declaration_bounds",
)
_CALLSITE_TOOL_NAMES = (
    "get_call_site_arguments",
    "get_backward_slice",
    "get_condition_guards",
)
_GUARD_TOOL_NAMES = (
    "get_condition_guards",
    "get_index_expression_structure",
    "resolve_symbolic_constant",
)
_FOLLOW_TOOL_NAMES = (
    "get_backward_slice",
    "get_index_expression_structure",
    "get_condition_guards",
)


def max_visible_tools() -> Optional[int]:
    """Hard cap on schemas bound this turn (AOOB_MAX_VISIBLE_TOOLS). None = all."""
    raw = (os.getenv("AOOB_MAX_VISIBLE_TOOLS") or "").strip()
    if not raw:
        return None
    try:
        n = int(raw)
    except ValueError:
        return None
    return max(1, min(n, len(TOOLS)))


def next_hop_gate_enabled() -> bool:
    """Fix #2. Isolation probe sets AOOB_NEXT_HOP_GATE=0 to hold this constant."""
    raw = (os.getenv("AOOB_NEXT_HOP_GATE") or "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def close_case_reprompt_enabled() -> bool:
    raw = (os.getenv("AOOB_CLOSE_CASE_REPROMPT") or "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _ident_norm(name: str) -> str:
    return (name or "").strip().split("@", 1)[0].strip().strip('"')


def _ident_tail(name: str) -> str:
    return _ident_norm(name).split(".")[-1]


def _idents_from_c_text(text: str) -> list[str]:
    """Identifiers in an assignment / argument expression (not a verdict)."""
    found: list[str] = []
    seen: set[str] = set()

    def _add(raw: str) -> None:
        name = _ident_norm(raw)
        if not name or name.lower() in _SKIP_IDENT_TOKENS or name in seen:
            return
        if len(name) < 2:
            return
        if (name.startswith("Get") or name.startswith("Mo_Inst_Get")) and (
            name.endswith("Idx") or re.search(r"Get\w*Idx$", name)
        ):
            return
        seen.add(name)
        found.append(name)

    ops = extract_index_operands([text or ""])
    for c in ops.get("index_operand_candidates") or []:
        _add(str(c))
    for m in _IDENT_TOKEN_RE.finditer(text or ""):
        _add(m.group(1))
    return found


def _payload_about_symbol(data: dict, symbol: str) -> bool:
    tail = _ident_tail(symbol)
    if not tail:
        return False
    for key in ("variable", "symbol", "parameter_name", "name"):
        val = data.get(key)
        if val and _ident_tail(str(val)) == tail:
            return True
    lookup = data.get("lookup")
    if isinstance(lookup, dict):
        q = lookup.get("queried_name")
        if q and _ident_tail(str(q)) == tail:
            return True
    return False


def _df_keys_matched(data: dict) -> Optional[list]:
    if "df_keys_matched" in data:
        keys = data.get("df_keys_matched")
        return keys if isinstance(keys, list) else None
    lookup = data.get("lookup")
    if isinstance(lookup, dict) and "df_keys_matched" in lookup:
        keys = lookup.get("df_keys_matched")
        return keys if isinstance(keys, list) else None
    return None


def _payload_writes(data: dict) -> list:
    writes = data.get("writes_found") or data.get("writes") or []
    return writes if isinstance(writes, list) else []


def _looks_like_aggregate(name: str) -> bool:
    tail = _ident_tail(name)
    low = tail.lower()
    return low.endswith(
        ("_ast", "_pst", "_table", "_buf", "_buffer", "_array", "_queue", "_lut")
    )


def _simple_rhs_ident(text: str) -> Optional[str]:
    """Bare identifier on an assignment RHS (optional cast). None if the RHS is complex."""
    bits = re.split(r"(?<![=!<>])=(?![=])", text or "", maxsplit=1)
    rhs = (bits[1] if len(bits) == 2 else text or "").strip().rstrip(";").strip()
    rhs = re.sub(r"/\*.*?\*/", " ", rhs)
    rhs = re.sub(r"//.*$", "", rhs)
    rhs = re.sub(r"^\(\s*[A-Za-z_]\w*(?:\s*\*)?\s*\)\s*", "", rhs).strip()
    m = re.fullmatch(r"[A-Za-z_]\w*(?:\s*\.\s*[A-Za-z_]\w*)?", rhs)
    if not m:
        return None
    return re.sub(r"\s+", "", m.group(0))


def _named_next_from_payload(data: dict) -> list[str]:
    """Identifiers a tool result named as the next origin to retrieve.

    Only call-site candidates, a bare assignment RHS, or subscript operands —
    not every token in a complex write (that chained 2475 into array/macro names).
    """
    sliced = _ident_norm(
        str(data.get("variable") or data.get("symbol") or data.get("parameter_name") or "")
    )
    named: list[str] = []
    seen: set[str] = set()

    def _add(raw: str) -> None:
        name = _ident_norm(raw)
        if not name or name.lower() in _SKIP_IDENT_TOKENS:
            return
        if name == sliced or _ident_tail(name) == _ident_tail(sliced):
            return
        if _looks_like_aggregate(name):
            return
        key = _ident_tail(name)
        if key in seen:
            return
        seen.add(key)
        named.append(name)

    csa = data.get("call_site_arguments")
    if isinstance(csa, dict):
        for c in csa.get("next_slice_candidates") or []:
            _add(str(c))
    for c in data.get("next_slice_candidates") or []:
        _add(str(c))
    for w in _payload_writes(data):
        if not isinstance(w, dict):
            continue
        text = str(w.get("assigned_expression_text") or "")
        simple = _simple_rhs_ident(text)
        if simple:
            _add(simple)
            continue
        bits = re.split(r"(?<![=!<>])=(?![=])", text, maxsplit=1)
        rhs = bits[1] if len(bits) == 2 else text
        ops = extract_index_operands([rhs])
        for c in ops.get("index_operand_candidates") or []:
            _add(str(c))
    operand = data.get("operand")
    if operand and not _looks_like_aggregate(str(operand)):
        _add(str(operand))
    return named


def symbol_explicitly_unresolvable(messages: list, symbol: str) -> bool:
    """True when a tool reported this name as a miss / macro / empty DF — not CF depth."""
    if not symbol:
        return False
    for m in messages:
        if not isinstance(m, ToolMessage):
            continue
        data = _parse_tool_json(str(m.content))
        if not data or not _payload_about_symbol(data, symbol):
            continue
        parse_note = str(data.get("parse_note") or "")
        if data.get("found") is False and parse_note:
            return True
        if (
            data.get("found") is True
            and data.get("value") is None
            and "unresolved" in parse_note.lower()
        ):
            return True
        if data.get("parameter_note") or data.get("call_site_arguments"):
            continue
        writes = _payload_writes(data)
        keys = _df_keys_matched(data)
        if not writes and keys == []:
            return True
        if parse_note and not writes and _UNRESOLVABLE_RE.search(parse_note):
            return True
        # Targeted origin lookup of a macro/enum with no writes: tool has
        # answered; do not chain retries on the ALL_CAPS spelling.
        tool_name = getattr(m, "name", None)
        tail = _ident_tail(symbol)
        if (
            not writes
            and not data.get("parameter_note")
            and tool_name in _ORIGIN_TOOLS
            and tail.isupper()
            and "_" in tail
        ):
            return True
        for u in data.get("unresolved_paths") or []:
            reason = str(u.get("reason") if isinstance(u, dict) else u)
            if _CF_BOUNDARY_RE.search(reason):
                continue
            if not writes and _UNRESOLVABLE_RE.search(reason):
                return True
    return False


def _symbol_origin_succeeded(messages: list, symbol: str) -> bool:
    for m in messages:
        if not isinstance(m, ToolMessage):
            continue
        data = _parse_tool_json(str(m.content))
        if not data or not _payload_about_symbol(data, symbol):
            continue
        if _payload_writes(data):
            return True
        if data.get("call_site_arguments"):
            return True
        conds = (
            data.get("conditions")
            or data.get("guards")
            or data.get("guards_found")
            or []
        )
        if conds:
            return True
        if data.get("found") is True and data.get("operations"):
            return True
        if data.get("found") is True and data.get("value") is not None:
            return True
        if data.get("found") is True and data.get("kind") in {"define", "enum_member"}:
            return True
    return False


def pending_named_hop(messages: list) -> Optional[str]:
    """First identifier named by origin tools that is not yet retrieved or released."""
    indexed = _indexed_objects_from_messages(messages)
    indexed_tails = {_ident_tail(o) for o in indexed}
    named: list[str] = []
    seen: set[str] = set()
    for m in messages:
        if not isinstance(m, ToolMessage):
            continue
        if getattr(m, "name", None) not in (
            _ORIGIN_TOOLS
            | {"get_index_expression_structure", "get_condition_guards"}
        ):
            continue
        data = _parse_tool_json(str(m.content))
        if not data:
            continue
        for ident in _named_next_from_payload(data):
            tail = _ident_tail(ident)
            if not tail or tail in indexed_tails or ident in indexed:
                continue
            if tail in seen:
                continue
            seen.add(tail)
            named.append(ident)
    for ident in named:
        if symbol_explicitly_unresolvable(messages, ident):
            continue
        if _symbol_origin_succeeded(messages, ident):
            continue
        return ident
    return None


def tools_returned_closing_evidence(messages: list) -> bool:
    """Bound, guard, or index transform actually present in a tool payload."""
    for m in messages:
        if not isinstance(m, ToolMessage):
            continue
        data = _parse_tool_json(str(m.content))
        if not data:
            continue
        name = getattr(m, "name", None)
        if name == "get_declaration_bounds":
            size = data.get("array_size")
            if data.get("found") and size not in (None, 0, "0"):
                return True
            nested = data.get("nested_array_fields") or []
            if any(
                isinstance(f, dict) and f.get("array_size") not in (None, 0, "0")
                for f in nested
            ):
                return True
        if name == "get_condition_guards":
            conds = (
                data.get("conditions")
                or data.get("guards")
                or data.get("guards_found")
                or []
            )
            if conds:
                return True
        if name == "get_index_expression_structure" and data.get("operations"):
            return True
        for w in _payload_writes(data):
            if not isinstance(w, dict):
                continue
            if _TRANSFORM_RE.search(str(w.get("assigned_expression_text") or "")):
                return True
    return False


def select_visible_tool_names(
    messages: list, cap: Optional[int] = None
) -> list[str]:
    """At most `cap` tool schemas, swapped by investigation stage."""
    all_names = [t.name for t in TOOLS]
    if cap is None:
        cap = max_visible_tools()
    if cap is None or cap >= len(all_names):
        return all_names
    used = _tool_names_used(messages)
    pending = pending_named_hop(messages) if next_hop_gate_enabled() else None
    if pending:
        preferred = _HOP_TOOL_NAMES
    elif "get_affected_symbols" not in used:
        preferred = _SITE_TOOL_NAMES
    elif not _has_useful_index_origin_tool(messages):
        preferred = _ORIGIN_CORE_NAMES
    elif _parameter_path_open(messages):
        preferred = _CALLSITE_TOOL_NAMES
    elif _guards_missing(messages):
        preferred = _GUARD_TOOL_NAMES
    else:
        preferred = _FOLLOW_TOOL_NAMES
    out: list[str] = []
    for name in list(preferred) + all_names:
        if name in out:
            continue
        out.append(name)
        if len(out) >= cap:
            break
    return out


def _visible_tools(messages: list) -> list:
    by_name = {t.name: t for t in TOOLS}
    return [by_name[n] for n in select_visible_tool_names(messages) if n in by_name]


def _bind_llm_tools(llm: Any, tools: list, *, force: bool):
    bound = llm.bind_tools(tools)
    if not force:
        return bound
    try:
        return llm.bind_tools(tools, tool_choice="any")
    except TypeError:
        return bound


def _build_next_hop_nudge(messages: list) -> str:
    pending = pending_named_hop(messages)
    lines = [NEXT_HOP_NUDGE]
    if pending:
        lines.append(f"Required identifier: {pending}")
        lines.append(
            f"Suggested get_backward_slice(variable_name={pending!r}, "
            "from_location=<alarm location>)."
        )
    return "\n".join(lines)


def _force_tools_needed(
    store: DataStore, order_id: int, messages: list, tool_n: int
) -> bool:
    if not _is_oob_alarm(store, order_id):
        return False
    if tool_n >= _MAX_TOOL_MESSAGES - 1:
        return False
    if not _has_useful_index_origin_tool(messages):
        return True
    if next_hop_gate_enabled() and pending_named_hop(messages):
        return True
    if _parameter_path_open(messages):
        return True
    if tool_n <= 8 and _guards_missing(messages):
        return True
    return False


def _pack_report_evidence(messages: list) -> str:
    """Keep site bootstrap + latest origin/guard evidence, then recent tools."""
    ordered: list[ToolMessage] = [
        m for m in messages if isinstance(m, ToolMessage)
    ]
    if not ordered:
        return ""

    selected: list[ToolMessage] = []
    seen: set[int] = set()

    def _add(msg: ToolMessage) -> None:
        key = id(msg)
        if key in seen:
            return
        seen.add(key)
        selected.append(msg)

    for msg in ordered:
        if getattr(msg, "name", None) == "get_affected_symbols":
            _add(msg)
            break

    sticky = _ORIGIN_TOOLS | {
        "get_condition_guards",
        "get_declaration_bounds",
        "get_index_expression_structure",
    }
    for msg in reversed(ordered):
        if getattr(msg, "name", None) in sticky:
            _add(msg)
            break

    for msg in reversed(ordered):
        if len(selected) >= 8:
            break
        _add(msg)

    index = {id(m): i for i, m in enumerate(ordered)}
    selected.sort(key=lambda m: index.get(id(m), 0))
    return "\n\n---\n\n".join(
        f"TOOL[{m.name}]:\n{str(m.content)[:4000]}" for m in selected
    )


def _fill_report_metadata(
    data: dict,
    store: DataStore,
    order_id: int,
    messages: list,
) -> dict:
    """Stamp store facts and gate-derived completeness. Never sets classification."""
    alarm = store.get_alarm(order_id)
    data["alarm_order_id"] = order_id
    data["tools_used"] = sorted(_tool_names_used(messages))
    if alarm is not None:
        data["alarm_type"] = alarm.type or ""
        data["alarm_category"] = alarm.category or ""
        data["location"] = alarm.location or ""
        data["astree_message"] = alarm.message or data.get("astree_message") or ""

    be = data.get("bounds_evidence")
    if not isinstance(be, dict):
        be = {}
    has_origin = _has_useful_index_origin_tool(messages)
    needs_cs = _parameter_path_open(messages)
    completeness = (be.get("evidence_completeness") or "").strip()
    if not has_origin:
        be["evidence_completeness"] = "not traced"
    elif needs_cs:
        if completeness in {"", "fully traced"}:
            be["evidence_completeness"] = "partially traced"
    elif not completeness:
        be["evidence_completeness"] = "partially traced"
    be.setdefault("symbol", "")
    if "declared_size" not in be:
        be["declared_size"] = None
    be.setdefault("index_origin_summary", "")
    data["bounds_evidence"] = be
    return data


def _maybe_close_case_reprompt(
    llm: Any,
    data: dict,
    messages: list,
    pack: str,
) -> dict:
    """One extra model turn when the draft hedges past evidence already in tools.

    Does not write true/false in Python. If the model still returns review, keep it.
    """
    if not close_case_reprompt_enabled():
        return data
    be = data.get("bounds_evidence")
    completeness = be.get("evidence_completeness") if isinstance(be, dict) else None
    if data.get("classification") != "review":
        return data
    if completeness != "fully traced":
        return data
    if not tools_returned_closing_evidence(messages):
        return data
    _log(
        "[report] fully traced + bound/guard/transform still classified review "
        "→ one re-prompt (model-authored true/false; no Python override)"
    )
    try:
        raw = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "Your draft marks evidence_completeness as fully traced, and "
                        "tools already returned a declaration bound, a condition guard, "
                        "or an index transform. classification MUST be \"true\" or "
                        "\"false\" — not \"review\". Hedging because a helper name is "
                        "unfamiliar is not allowed when the index operand and a bound "
                        "are in the tool evidence. Keep comment internally consistent "
                        "with those tool facts (do not ignore a shift/mask/guard you "
                        "already recorded). Return ONLY JSON with keys classification, "
                        "comment, and confidence. No other keys."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Draft report:\n{json.dumps(data)[:3500]}\n\n"
                        f"Tool evidence excerpt:\n{pack[:3500]}"
                    )
                ),
            ]
        )
    except Exception as exc:  # noqa: BLE001
        _log(f"[report] close-case re-prompt failed ({exc}); keeping review")
        return data
    text = raw.content if hasattr(raw, "content") else str(raw)
    if isinstance(text, list):
        text = "\n".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in text
        )
    try:
        patch = _normalize_report_dict(extract_json(str(text)))
    except Exception as exc:  # noqa: BLE001
        _log(f"[report] close-case re-prompt parse failed: {exc}")
        return data
    cls = patch.get("classification")
    if cls not in {"true", "false"}:
        _log(
            f"[report] close-case re-prompt still {cls!r}; keeping review "
            "(no Python flip)"
        )
        if patch.get("comment"):
            data["comment"] = patch["comment"]
        return data
    data["classification"] = cls
    if patch.get("comment"):
        data["comment"] = patch["comment"]
    if patch.get("confidence"):
        data["confidence"] = patch["confidence"]
    _log(f"[report] close-case re-prompt authored classification={cls}")
    return data


def build_agent(store: DataStore, model: Optional[str] = None):
    bind_store(store)
    llm = build_llm(model=model)
    report_llm = llm

    def agent_node(state: AgentState) -> dict:
        msgs = list(state["messages"])
        tool_n = _count_tool_messages(msgs)
        trimmed = trim_messages(
            msgs,
            max_tokens=10000,
            strategy="last",
            token_counter="approximate",
            start_on="human",
            include_system=True,
            allow_partial=True,
        )
        visible = _visible_tools(msgs)
        force = _force_tools_needed(store, state["order_id"], msgs, tool_n)
        runner = _bind_llm_tools(llm, visible, force=force)
        pending = pending_named_hop(msgs) if next_hop_gate_enabled() else None
        _log(
            f"[agent] invoke model msgs={len(trimmed)} tools_so_far={tool_n} "
            f"force_tool={force} visible={len(visible)} "
            f"names={[t.name for t in visible]}"
            + (f" next_hop={pending}" if pending else "")
        )
        try:
            response = runner.invoke(trimmed)
        except Exception as exc:  # noqa: BLE001
            if force:
                _log(f"[agent] tool_choice=any failed ({exc}); retry unbound")
                response = _bind_llm_tools(llm, visible, force=False).invoke(trimmed)
            else:
                raise
        tool_calls = getattr(response, "tool_calls", None) or []
        _log(f"[agent] tool_calls={len(tool_calls)}")
        return {"messages": [response]}

    def tools_node(state: AgentState) -> dict:
        result = ToolNode(TOOLS).invoke(state)
        rounds = int(state.get("tool_rounds") or 0) + 1
        _log(f"[tools] completed round={rounds}")
        return {**result, "tool_rounds": rounds}

    def nudge_node(state: AgentState) -> dict:
        # Prefer origin → named next-hop → call-site follow-up → guards.
        if not _has_useful_index_origin_tool(state["messages"]):
            n = int(state.get("origin_nudges") or 0) + 1
            text = _build_origin_nudge(state["messages"])
            _log(f"[nudge] useful index-origin slice required (nudge={n})")
            return {
                "origin_nudges": n,
                "messages": [HumanMessage(content=text)],
            }
        pending = pending_named_hop(state["messages"]) if next_hop_gate_enabled() else None
        if pending:
            n = int(state.get("next_hop_nudges") or 0) + 1
            text = _build_next_hop_nudge(state["messages"])
            _log(f"[nudge] next-hop identifier {pending!r} required (nudge={n})")
            return {
                "next_hop_nudges": n,
                "messages": [HumanMessage(content=text)],
            }
        if _needs_call_site_followup(state["messages"]):
            n = int(state.get("call_site_nudges") or 0) + 1
            text = _build_call_site_nudge(state["messages"])
            _log(f"[nudge] call-site argument follow-up required (nudge={n})")
            return {
                "call_site_nudges": n,
                "messages": [HumanMessage(content=text)],
            }
        n = int(state.get("guard_nudges") or 0) + 1
        text = _build_guard_nudge(state["messages"])
        _log(f"[nudge] get_condition_guards required (nudge={n})")
        return {
            "guard_nudges": n,
            "messages": [HumanMessage(content=text)],
        }

    def retry_node(state: AgentState) -> dict:
        n = int(state.get("force_retries") or 0) + 1
        _log(f"[retry] gate still open — force another tool turn (retry={n})")
        return {"force_retries": n}

    def after_agent(state: AgentState) -> Literal["tools", "nudge", "retry", "report"]:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        tool_n = _count_tool_messages(state["messages"])
        if tool_n >= _MAX_TOOL_MESSAGES:
            return "report"
        oob = _is_oob_alarm(store, state["order_id"])
        can_nudge = tool_n < _MAX_TOOL_MESSAGES - 1
        force_n = int(state.get("force_retries") or 0)
        msgs = state["messages"]

        if (
            oob
            and can_nudge
            and not _has_useful_index_origin_tool(msgs)
        ):
            if int(state.get("origin_nudges") or 0) < 3:
                _log("[route] missing useful index-origin slice → nudge")
                return "nudge"
            if force_n < 3:
                _log("[route] origin still missing after nudges → retry/force")
                return "retry"

        pending = pending_named_hop(msgs) if next_hop_gate_enabled() else None
        if oob and can_nudge and pending:
            # Stay open until the named ident is retrieved or a tool marks it
            # unresolvable. Nudge text is capped; force continues for a budget
            # — not a "this name is untraceable" release. Model refusal still
            # has to reach the report node (incomplete), or recursion_limit fires.
            if int(state.get("next_hop_nudges") or 0) < 2:
                _log(f"[route] named next-hop {pending!r} still open → nudge")
                return "nudge"
            if force_n < 8:
                _log(f"[route] named next-hop {pending!r} still open → retry/force")
                return "retry"
            _log(
                f"[route] named next-hop {pending!r} still open after force budget "
                "→ continue (name not marked unresolvable; model did not retrieve it)"
            )

        if oob and can_nudge and _needs_call_site_followup(msgs):
            if int(state.get("call_site_nudges") or 0) < 2:
                _log("[route] missing call-site argument follow-up → nudge")
                return "nudge"
            if force_n < 3:
                _log("[route] call-site still open after nudges → retry/force")
                return "retry"

        if (
            oob
            and can_nudge
            and tool_n <= 8
            and _guards_missing(msgs)
        ):
            if int(state.get("guard_nudges") or 0) < 1:
                _log("[route] missing get_condition_guards → nudge")
                return "nudge"
            if force_n < 2:
                _log("[route] guards still missing after nudge → retry/force")
                return "retry"
        return "report"

    def after_tools(state: AgentState) -> Literal["agent", "report"]:
        if _count_tool_messages(state["messages"]) >= _MAX_TOOL_MESSAGES:
            _log(f"[route] tool-message ceiling {_MAX_TOOL_MESSAGES} → report")
            return "report"
        return "agent"

    def report_node(state: AgentState) -> dict:
        _log("[report] building structured JSON report ...")
        pack = _pack_report_evidence(state["messages"])
        used = sorted(_tool_names_used(state["messages"]))
        human = HumanMessage(
            content=(
                f"Alarm order id: {state['order_id']}\n"
                f"Tools actually called: {used}\n\n"
                f"Evidence from tools:\n{pack}"
            )
        )
        system = SystemMessage(
            content=(
                REPORT_PROMPT
                + "\nSchema reminder (types only):\n"
                + json_schema_brief()
            )
        )

        data: Optional[dict] = None
        try:
            structured = report_llm.with_structured_output(AlarmInvestigationReport)
            result = structured.invoke([system, human])
            if isinstance(result, AlarmInvestigationReport):
                data = result.model_dump()
            elif isinstance(result, dict):
                data = result
        except Exception as exc:  # noqa: BLE001
            _log(f"[report] with_structured_output failed ({exc}); JSON extract fallback")

        if data is None:
            raw = report_llm.invoke([system, human])
            content = raw.content if hasattr(raw, "content") else str(raw)
            if isinstance(content, list):
                content = "\n".join(
                    b.get("text", "") if isinstance(b, dict) else str(b)
                    for b in content
                )
            data = extract_json(str(content))

        data["alarm_order_id"] = state["order_id"]
        data = _normalize_report_dict(data)
        data = _fill_report_metadata(
            data, store, state["order_id"], state["messages"]
        )

        # If index origin was never retrieved, keep confidence cautious unless
        # the model already chose low — do not flip classification.
        if not _has_useful_index_origin_tool(state["messages"]):
            be = data.get("bounds_evidence")
            if isinstance(be, dict) and be.get("evidence_completeness") == "fully traced":
                be["evidence_completeness"] = "not traced"
            if data.get("confidence") == "high":
                data["confidence"] = "medium"
                _log("[report] capped confidence high→medium (no useful index slice)")

        # Parameter path without call-site follow-up: never allow medium/high "true".
        if _parameter_path_open(state["messages"]):
            be = data.get("bounds_evidence")
            if isinstance(be, dict) and be.get("evidence_completeness") == "fully traced":
                be["evidence_completeness"] = "partially traced"
            if data.get("classification") == "true" and data.get("confidence") in {
                "medium",
                "high",
            }:
                data["confidence"] = "low"
                _log(
                    "[report] capped true confidence → low "
                    "(parameter index without call-site follow-up)"
                )

        # "not traced" must not carry medium/high confidence.
        be = data.get("bounds_evidence")
        if isinstance(be, dict) and be.get("evidence_completeness") == "not traced":
            if data.get("confidence") in {"medium", "high"}:
                data["confidence"] = "low"
                _log("[report] capped confidence → low (evidence not traced)")

        # Incomplete origin is not a false-positive finding.
        be = data.get("bounds_evidence")
        completeness = be.get("evidence_completeness") if isinstance(be, dict) else None
        if (
            data.get("classification") == "false"
            and completeness == "not traced"
        ):
            data["classification"] = "review"
            _log("[report] false→review (origin not traced; incomplete ≠ false)")

        if not data.get("classification"):
            _log("[report] classification missing; requesting repair pass")
            repair = report_llm.invoke(
                [
                    SystemMessage(
                        content=(
                            "Return ONLY JSON with keys classification "
                            '("true", "false", or "review"), comment (short string), '
                            "and confidence (low|medium|high) based on this "
                            "draft report and tool evidence. No other keys. "
                            "Use review when origin is incomplete or the leftover "
                            "Astrée interval is unexplained. Do not treat Astrée "
                            "abstract [lo,hi] upper bounds as concrete index values."
                        )
                    ),
                    HumanMessage(
                        content=(
                            f"Draft report:\n{json.dumps(data)[:3500]}\n\n"
                            f"Tool evidence excerpt:\n{pack[:3500]}"
                        )
                    ),
                ]
            )
            repair_text = repair.content if hasattr(repair, "content") else str(repair)
            if isinstance(repair_text, list):
                repair_text = "\n".join(
                    b.get("text", "") if isinstance(b, dict) else str(b)
                    for b in repair_text
                )
            try:
                patch = _normalize_report_dict(extract_json(str(repair_text)))
                for k in ("classification", "comment", "confidence"):
                    if patch.get(k):
                        data[k] = patch[k]
            except Exception as exc:  # noqa: BLE001
                _log(f"[report] repair parse failed: {exc}")
            be = data.get("bounds_evidence")
            completeness = (
                be.get("evidence_completeness") if isinstance(be, dict) else None
            )
            if data.get("classification") == "false" and completeness == "not traced":
                data["classification"] = "review"
                _log("[report] false→review after repair (origin not traced)")

        data = _maybe_close_case_reprompt(
            report_llm, data, state["messages"], pack
        )

        report = AlarmInvestigationReport.model_validate(data)
        return {"report": report.model_dump()}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_node("nudge", nudge_node)
    graph.add_node("retry", retry_node)
    graph.add_node("report", report_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        after_agent,
        {"tools": "tools", "nudge": "nudge", "retry": "retry", "report": "report"},
    )
    graph.add_edge("nudge", "agent")
    graph.add_edge("retry", "agent")
    graph.add_conditional_edges(
        "tools", after_tools, {"agent": "agent", "report": "report"}
    )
    graph.add_edge("report", END)
    return graph.compile()


def json_schema_brief() -> str:
    return (
        '{"alarm_order_id":int,"alarm_type":str,"alarm_category":str,"location":str,'
        '"astree_message":str,'
        '"affected_symbols":[{"name":str,"kind":"variable|array|pointer|struct|field|other|unknown","evidence":str}],'
        '"function_name":str|null,"function_snippet":str,'
        '"manipulation_sequence":[{"order":int,"function":str,"access":str,"variable":str,"location":str,"note":str}],'
        '"bounds_evidence":{"symbol":str,"declared_size":int|null,'
        '"index_origin_summary":str,'
        '"evidence_completeness":"fully traced|partially traced|not traced"}|null,'
        '"summary":str,'
        '"classification":"true|false|review",'
        '"comment":str,'
        '"confidence":"low|medium|high",'
        '"tools_used":[str]}'
    )


def _normalize_report_dict(data: dict) -> dict:
    """Coerce common model variants into the Pydantic schema (no verdict injection)."""
    out = dict(data)
    raw_cls = str(out.get("classification") or "").strip().lower()
    if raw_cls in {"true", "t", "yes", "bug", "tp", "true_positive", "true-positive"}:
        out["classification"] = "true"
    elif raw_cls in {
        "review",
        "uncertain",
        "undecided",
        "unknown",
        "needs_review",
        "needs-review",
        "needs review",
        "flag",
    }:
        out["classification"] = "review"
    elif raw_cls in {
        "false",
        "f",
        "no",
        "fp",
        "false_positive",
        "false-positive",
        "not a bug",
    }:
        out["classification"] = "false"
    # Leave other values for Pydantic to reject / surface.

    conf = str(out.get("confidence") or "").strip().lower()
    if conf in {"low", "medium", "high"}:
        out["confidence"] = conf
    elif conf.endswith("%") or conf.replace(".", "", 1).isdigit():
        # Numeric confidence → map roughly; still the model's number, not CSV.
        try:
            n = float(conf.rstrip("%"))
            if n >= 75:
                out["confidence"] = "high"
            elif n >= 40:
                out["confidence"] = "medium"
            else:
                out["confidence"] = "low"
        except ValueError:
            out["confidence"] = "medium"

    if "comment" not in out or out.get("comment") is None:
        out["comment"] = ""
    if "astree_message" not in out or out.get("astree_message") is None:
        out["astree_message"] = ""
    return out


def extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"No JSON object in model output:\n{text[:800]}")
    return json.loads(text[start : end + 1])


def investigate_alarm(
    order_id: int,
    store: DataStore,
    model: Optional[str] = None,
) -> AlarmInvestigationReport:
    """Run investigation to completion and return the final report."""
    report: Optional[AlarmInvestigationReport] = None
    for event in stream_investigate(order_id, store, model=model):
        if event.get("type") == "report":
            report = AlarmInvestigationReport.model_validate(event["data"])
        elif event.get("type") == "error":
            raise RuntimeError(event.get("message") or "Investigation failed")
    if report is None:
        raise ValueError("Agent finished without producing a structured report.")
    return report


def stream_investigate(
    order_id: int,
    store: DataStore,
    model: Optional[str] = None,
):
    """Yield chat-friendly progress events while investigating an alarm.

    Event types:
      - status: {message}
      - tool_call: {name, args}
      - tool_result: {name, preview}
      - agent: {content}  (brief model text if any)
      - report: {data: AlarmInvestigationReport dict}
      - error: {message}
      - done: {}
    """
    yield {
        "type": "status",
        "message": f"Starting investigation for Order {order_id}…",
    }
    try:
        agent = build_agent(store, model=model)
        initial = {
            "order_id": order_id,
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"Investigate Astrée alarm Order id {order_id}. "
                        "Use whichever of the five tools you need."
                    )
                ),
            ],
            "report": None,
            "tool_rounds": 0,
            "origin_nudges": 0,
            "call_site_nudges": 0,
            "guard_nudges": 0,
            "next_hop_nudges": 0,
            "force_retries": 0,
        }
        yield {"type": "status", "message": "Agent is deciding which tools to call…"}
        yield {
            "type": "status",
            "message": (
                f"Calling {llm_backend_label()} now — the first response often "
                "takes 1–3 minutes; heartbeats will appear while waiting."
            ),
        }

        final_report = None
        for update in agent.stream(
            initial,
            config={"recursion_limit": 64},
            stream_mode="updates",
        ):
            if not isinstance(update, dict):
                continue
            for node_name, payload in update.items():
                if node_name == "nudge":
                    yield {
                        "type": "status",
                        "message": (
                            "Index operand not correctly traced yet — nudging "
                            "agent to slice index_operand_candidates (not the "
                            "array / sentinel)."
                        ),
                    }
                elif node_name == "retry":
                    yield {
                        "type": "status",
                        "message": (
                            "Process gate still open — forcing another tool "
                            "turn instead of writing the report."
                        ),
                    }
                elif node_name == "agent":
                    msgs = payload.get("messages") or []
                    for msg in msgs:
                        if not isinstance(msg, AIMessage):
                            continue
                        tool_calls = getattr(msg, "tool_calls", None) or []
                        for tc in tool_calls:
                            yield {
                                "type": "tool_call",
                                "name": tc.get("name") or "tool",
                                "args": tc.get("args") or {},
                                "id": tc.get("id"),
                            }
                        text = _message_text(msg)
                        if not tool_calls:
                            yield {"type": "no_tool_call"}
                        if text and not tool_calls:
                            yield {"type": "agent", "content": text}
                        elif text and tool_calls:
                            yield {
                                "type": "status",
                                "message": f"Model requested {len(tool_calls)} tool call(s).",
                            }
                elif node_name == "tools":
                    msgs = payload.get("messages") or []
                    for msg in msgs:
                        if not isinstance(msg, ToolMessage):
                            continue
                        preview = str(msg.content)
                        if len(preview) > 1200:
                            preview = preview[:1200] + "\n…[truncated]"
                        yield {
                            "type": "tool_result",
                            "name": getattr(msg, "name", None) or "tool",
                            "preview": preview,
                        }
                    yield {
                        "type": "status",
                        "message": "Tool results received — agent continuing…",
                    }
                elif node_name == "report":
                    data = payload.get("report")
                    if data:
                        data.setdefault("alarm_order_id", order_id)
                        final_report = data
                        yield {
                            "type": "status",
                            "message": "Building structured investigation report…",
                        }
                        yield {"type": "report", "data": data}

        if final_report is None:
            yield {
                "type": "error",
                "message": "Investigation finished without a structured report.",
            }
        else:
            yield {"type": "done", "order_id": order_id}
    except Exception as exc:  # noqa: BLE001
        _log(f"[agent] stream error: {exc}")
        yield {"type": "error", "message": str(exc)}


def _message_text(msg: AIMessage) -> str:
    content = getattr(msg, "content", "")
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts).strip()
    return str(content or "").strip()
