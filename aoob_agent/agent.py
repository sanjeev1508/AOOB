"""LangGraph ReAct agent over local Ollama or NVIDIA NIM for AOOB investigation."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Annotated, Any, Literal, Optional, TypedDict
from uuid import uuid4
from urllib.parse import urlparse
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener, urlopen

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.utils import trim_messages
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from aoob_agent.data_store import DataStore
from aoob_agent.report import AlarmInvestigationReport
from aoob_agent.tools import TOOLS, bind_store

DEFAULT_NVIDIA_MODEL = "meta/llama-3.1-70b-instruct"
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"

# Soft ceiling so multi-tool investigations (decl + index slice + guards + …) can finish.
_MAX_TOOL_MESSAGES = 12

# Index-origin retrieval tools — process gates only (never forces true/false).
_ORIGIN_TOOLS = frozenset({"get_trimmed_sequence", "get_caller_context"})

INDEX_ORIGIN_NUDGE = """You have not yet gathered enough index-origin evidence.

Required next step (retrieval only — not a verdict):
- Call get_function_snippet and read index_operands_at_alarm.
- Trace that operand symbol using get_trimmed_sequence(variable=<operand>). Do not
    treat tracing the indexed array symbol as index-origin completion.
- If get_trimmed_sequence returns origin_resolved=true, do not hop callers.
- If origin_resolved=false and scope is parameter, call get_caller_context(parameter=<operand>). 
- Do not finalize from Astrée interval text alone.
"""

CALLER_NUDGE = """The trace still requires caller-side evidence.

Required next step (retrieval only — not a verdict):
- Call get_caller_context (pass parameter name when tracing a parameter index).
- Then call get_trimmed_sequence on the same parameter/operand symbol.
- Continue until caller_context reports unresolved=true or origin_resolved becomes true.
"""

NEXT_HOP_NUDGE = """A caller hop target is open and has not been investigated yet.

Required next step (retrieval only — not a verdict):
- Call get_trimmed_sequence for the same traced operand symbol after the caller hop.
- If origin_resolved remains false, call get_caller_context again.
"""

SYSTEM_PROMPT = """You are an Astrée out-of-bounds (AOOB) investigation agent.

You must use retrieval tools only. Never invent sources, declarations, writes,
or caller relationships. There are no hardcoded verdict rules in Python: you
author classification/comment/confidence from tool facts.

Planner-turn discipline (STRICT):
- Stay on AOOB investigation only. Never provide generic software/module
    explanations, AUTOSAR overviews, architecture summaries, or tutorial text.
- If a tool is needed, emit tool calls only.
- If no additional tool is needed for this turn, emit exactly this JSON and
    nothing else: {"type":"no_tool_call"}
- Do not output markdown tables/lists unrelated to bounds analysis.

Available tools (retrieval-only):
1) get_variable_scope(symbol, alarm_order_id?)
    - Returns local/global/parameter/dynamic_heap/unknown scope facts.
2) get_declaration_info(symbol)
     - Returns declared type/kind/array_size/is_pointer/location/parse_note.
3) get_trimmed_sequence(alarm_order_id?, variable?)
    - Cross-function data-flow ordered sequence (shared with cf_viz panel).
    - Hard filter mode: keep write/mixed steps plus the queried alarm step.
    - Returns origin_resolved=true when the first step is already alarm-site
      write/mixed (self-contained origin at site).
4) get_function_snippet(alarm_order_id?, context_lines?)
    - Returns snippet for current trace function context and index_operands_at_alarm.
5) get_caller_context(parameter, alarm_order_id?, current_function?)
     - One-hop callers and optional argument expression text; unresolved=true when
         no caller edge is available.

Navigation strategy:
1. Call get_function_snippet and read index_operands_at_alarm.
2. Select the index operand symbol from that list and trace that operand, not
   the indexed array symbol.
3. Call get_variable_scope + get_declaration_info for the traced operand.
4. Call get_trimmed_sequence for that operand.
5. If origin_resolved=true, the index-origin gate is satisfied.
6. If origin_resolved=false and scope=parameter, call get_caller_context,
   then get_trimmed_sequence again on the same operand.
7. If origin_resolved=false and scope is local/global, do not caller-hop; keep
   tracing with available DF evidence or report limitation.
8. Classify based on evidence.

Critical epistemic guidance:
- Astrée [lo, hi] is abstract-domain information, not a concrete runtime value.
- Forbidden rationale: "array_size < Astrée hi => true bug".
- Index arithmetic rule: if in-code guards constrain the index into the legal
    safe range (for example if (idx < N)) or tracing resolves to a concrete value
    within capacity, classify optimistically as false (an analyzer false positive).
- For complex expressions (a+b, pointer offsets, struct members), reason with
    explicit bounds math and show index_expression, inferred range, and safe range.
- Missing allocation/capacity/guard context (especially dynamic pointers) must
    fallback to review with medium confidence and explicitly list missing evidence
    in bounds_evidence.index_origin_summary.
- Incomplete origin tracing should stay review/low confidence.

Trade-off to remember:
- This 5-tool set intentionally has no dedicated guard/constant/index-structure
    tools. Read guard/sentinel/shift evidence directly from function snippets
    instead of assuming a separate tool will return them.
"""

REPORT_PROMPT = """Using ONLY provided tool outputs, produce final report JSON.

No prose wrappers. No markdown. Return one JSON object matching schema.

Rules:
- Astrée alarm.message intervals are abstract over-approximations.
- Do not treat abstract hi as concrete index value.
- Do not claim true bug from array_size vs Astrée hi alone.
- If bounds context is missing (for example dynamic pointer allocation unknown),
  prefer classification=review with medium confidence and explicit missing evidence.
- tools_used must list only tools actually called.
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
    consecutive_no_tool_calls: int


def load_env(project_root: Path) -> None:
    load_dotenv(project_root / ".env", override=False)


def llm_choice_mode() -> Optional[str]:
    """AOOB_LLM_CHOOSED/AOOB_LLM_CHOOSE=local|network|nvidia selects preferred source."""
    raw = (
        os.getenv("AOOB_LLM_CHOOSED")
        or os.getenv("AOOB_LLM_CHOOSE")
        or os.getenv("CHOOSED")
        or ""
    ).strip().lower()
    return raw if raw in {"local", "network", "nvidia"} else None


def _local_ollama_model_name() -> str:
    return (
        os.getenv("OLLAMA_LOCAL")
        or os.getenv("OLLAMA_LOCAL_MODEL")
        or os.getenv("OLLAMA_LOCAL_PATH")
        or ""
    ).strip()


def _network_ollama_model_name() -> str:
    return (os.getenv("OLLAMA_NETWORK_MODEL") or "").strip()


def ollama_base_url(mode: Optional[str] = None) -> str:
    selected = mode or llm_choice_mode()
    if selected == "local":
        return (
            os.getenv("OLLAMA_LOCAL_HOST")
            or os.getenv("OLLAMA_LOCAL_BASE_URL")
            or os.getenv("OLLAMA_HOST")
            or os.getenv("OLLAMA_BASE_URL")
            or DEFAULT_OLLAMA_HOST
        ).rstrip("/")
    if selected == "network":
        return (
            os.getenv("OLLAMA_NETWORK_BASE_URL")
            or os.getenv("OLLAMA_HOST")
            or os.getenv("OLLAMA_BASE_URL")
            or DEFAULT_OLLAMA_HOST
        ).rstrip("/")
    return (
        os.getenv("OLLAMA_LOCAL_HOST")
        or os.getenv("OLLAMA_LOCAL_BASE_URL")
        or os.getenv("OLLAMA_HOST")
        or os.getenv("OLLAMA_BASE_URL")
        or os.getenv("OLLAMA_NETWORK_BASE_URL")
        or DEFAULT_OLLAMA_HOST
    ).rstrip("/")


def ollama_model_name() -> str:
    """Model tag from Ollama env vars (e.g. qwen2.5:7b)."""
    selected = llm_choice_mode()
    if selected == "local":
        return _local_ollama_model_name() or (os.getenv("OLLAMA_MODEL") or "").strip()
    if selected == "network":
        return _network_ollama_model_name() or (os.getenv("OLLAMA_MODEL") or "").strip()
    return (
        _local_ollama_model_name()
        or (os.getenv("OLLAMA_MODEL") or "").strip()
        or _network_ollama_model_name()
    )


def planner_model_name(use_ollama: Optional[bool] = None) -> str:
    """Model for tool-planning turns (can be lighter/faster than report model)."""
    if use_ollama is None:
        use_ollama = using_ollama()
    if use_ollama:
        selected = llm_choice_mode()
        if selected == "local":
            return (
                os.getenv("AOOB_TOOL_LOCAL_MODEL")
                or os.getenv("OLLAMA_TOOL_LOCAL_MODEL")
                or os.getenv("AOOB_TOOL_MODEL")
                or os.getenv("OLLAMA_TOOL_MODEL")
                or _local_ollama_model_name()
                or (os.getenv("OLLAMA_MODEL") or "").strip()
            ).strip()
        if selected == "network":
            return (
                os.getenv("AOOB_TOOL_NETWORK_MODEL")
                or os.getenv("OLLAMA_TOOL_NETWORK_MODEL")
                or os.getenv("AOOB_TOOL_MODEL")
                or os.getenv("OLLAMA_TOOL_MODEL")
                or _network_ollama_model_name()
                or (os.getenv("OLLAMA_MODEL") or "").strip()
            ).strip()
        return (
            os.getenv("AOOB_TOOL_MODEL")
            or os.getenv("OLLAMA_TOOL_MODEL")
            or _local_ollama_model_name()
            or (os.getenv("OLLAMA_MODEL") or "").strip()
            or _network_ollama_model_name()
        ).strip()
    return (
        os.getenv("AOOB_TOOL_NVIDIA_MODEL")
        or os.getenv("NVIDIA_TOOL_MODEL")
        or os.getenv("NVIDIA_MODEL")
        or DEFAULT_NVIDIA_MODEL
    ).strip()


def ollama_configured() -> bool:
    return bool(ollama_model_name())


def nvidia_configured() -> bool:
    return bool((os.getenv("NVIDIA_API_KEY") or "").strip())


def ollama_runtime_fallback_enabled() -> bool:
    raw = (os.getenv("AOOB_OLLAMA_RUNTIME_FALLBACK") or "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _ollama_health_timeout() -> float:
    raw = (os.getenv("OLLAMA_HEALTH_TIMEOUT") or "1.5").strip()
    try:
        return float(raw)
    except ValueError:
        return 1.5


def _ollama_available(timeout: Optional[float] = None) -> bool:
    if not ollama_configured():
        return False
    base = ollama_base_url()
    request = Request(f"{base}/api/tags", headers={"Accept": "application/json"})
    host = (urlparse(base).hostname or "").strip().lower()
    bypass_proxy = host in {"127.0.0.1", "localhost", "::1"}
    try:
        if bypass_proxy:
            opener = build_opener(ProxyHandler({}))
            response_ctx = opener.open(
                request,
                timeout=timeout or _ollama_health_timeout(),
            )
        else:
            response_ctx = urlopen(request, timeout=timeout or _ollama_health_timeout())
        with response_ctx as response:
            status = getattr(response, "status", 200)
            return 200 <= status < 300
    except (OSError, URLError, ValueError):
        return False


def _is_ollama_runtime_error(exc: Exception) -> bool:
    text = str(exc).lower()
    needles = (
        "timed out",
        "timeout",
        "connection refused",
        "failed to connect",
        "connection aborted",
        "connection reset",
        "max retries exceeded",
        "temporarily unavailable",
        "server disconnected",
        "network is unreachable",
    )
    return any(n in text for n in needles)


def llm_backend_override() -> Optional[str]:
    """AOOB_LLM_BACKEND=nvidia|ollama forces a backend even if both are configured."""
    raw = (os.getenv("AOOB_LLM_BACKEND") or "").strip().lower()
    return raw if raw in {"nvidia", "ollama"} else None


def using_ollama() -> bool:
    override = llm_backend_override()
    selected = llm_choice_mode()
    if override == "nvidia":
        return False
    if selected == "nvidia" and override != "ollama":
        return False
    if override == "ollama":
        if not ollama_configured():
            raise RuntimeError(
                "AOOB_LLM_BACKEND=ollama but no Ollama model env is set. "
                "Use OLLAMA_LOCAL_MODEL/OLLAMA_LOCAL_PATH, OLLAMA_MODEL, or OLLAMA_NETWORK_MODEL."
            )
        if not _ollama_available():
            raise RuntimeError(
                f"AOOB_LLM_BACKEND=ollama but Ollama is unreachable at {ollama_base_url()}."
            )
        return True
    return ollama_configured() and _ollama_available()


def resolve_model_name(
    model: Optional[str] = None,
    use_ollama: Optional[bool] = None,
    role: Literal["planner", "report"] = "report",
) -> str:
    if model:
        return model
    if use_ollama is None:
        use_ollama = using_ollama()
    if role == "planner":
        return planner_model_name(use_ollama=use_ollama)
    if use_ollama:
        return ollama_model_name()
    return os.getenv("NVIDIA_MODEL", DEFAULT_NVIDIA_MODEL) or DEFAULT_NVIDIA_MODEL


def llm_backend_label(role: Literal["planner", "report"] = "report") -> str:
    if using_ollama():
        return f"Ollama ({resolve_model_name(role=role)})"
    return f"NVIDIA ({resolve_model_name(role=role)})"


def planner_backend_label() -> str:
    return llm_backend_label(role="planner")


def report_backend_label() -> str:
    return llm_backend_label(role="report")


def build_llm(
    model: Optional[str] = None,
    backend: Optional[str] = None,
    role: Literal["planner", "report"] = "report",
) -> Any:
    if backend == "ollama":
        use_ollama = True
    elif backend == "nvidia":
        use_ollama = False
    else:
        use_ollama = using_ollama()
    name = resolve_model_name(model, use_ollama=use_ollama, role=role)
    selected = llm_choice_mode()
    timeout = float(os.getenv("OLLAMA_TIMEOUT" if use_ollama else "NVIDIA_TIMEOUT", "300"))
    if (
        not use_ollama
        and selected != "nvidia"
        and ollama_configured()
        and nvidia_configured()
        and not llm_backend_override()
    ):
        _log(f"[llm] ollama unreachable at {ollama_base_url()}; falling back to nvidia")
    if use_ollama:
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:
            raise RuntimeError(
                "An Ollama model is configured but langchain-ollama is not installed. "
                "Run: py -3.13 -m pip install langchain-ollama"
            ) from exc
        base_url = ollama_base_url()
        num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "16384"))
        _log(
            f"[llm] role={role} backend=ollama model={name} base_url={base_url} "
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
            "NVIDIA_API_KEY missing and no reachable Ollama backend was found. "
            "Set NVIDIA_API_KEY or configure a reachable Ollama model/base URL in .env."
        )
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    _log(f"[llm] role={role} backend=nvidia model={name} timeout={timeout}s")
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


def _latest_trimmed_sequence(messages: list) -> Optional[dict]:
    for m in reversed(messages):
        if not isinstance(m, ToolMessage):
            continue
        if getattr(m, "name", None) != "get_trimmed_sequence":
            continue
        data = _parse_tool_json(str(m.content))
        if data:
            return data
    return None


def _coerce_xml_scalar(text: str) -> Any:
    val = (text or "").strip()
    if re.fullmatch(r"-?\d+", val):
        try:
            return int(val)
        except ValueError:
            return val
    if re.fullmatch(r"-?\d+\.\d+", val):
        try:
            return float(val)
        except ValueError:
            return val
    low = val.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low in {"none", "null"}:
        return None
    return val


def parse_llm_tool_response(llm_output: str) -> dict[str, Any]:
    """Parse tool calls from model text with JSON first, then XML fallback."""
    cleaned_output = (llm_output or "").strip()

    # 1) Standard JSON attempt (with optional markdown-fence stripping).
    try:
        json_candidate = cleaned_output
        if json_candidate.startswith("```"):
            lines = json_candidate.splitlines()
            if len(lines) >= 2 and (lines[0].startswith("```json") or lines[0].startswith("```")):
                json_candidate = "\n".join(lines[1:-1]).strip()
        payload = json.loads(json_candidate)
    except Exception:  # noqa: BLE001
        payload = None

    if isinstance(payload, dict):
        if payload.get("type") == "tool" and payload.get("name"):
            return {
                "type": "tool",
                "name": str(payload.get("name")),
                "args": payload.get("args") if isinstance(payload.get("args"), dict) else {},
            }
        if payload.get("name") and isinstance(payload.get("arguments"), dict):
            return {
                "type": "tool",
                "name": str(payload.get("name")),
                "args": dict(payload.get("arguments") or {}),
            }
        if payload.get("name") and isinstance(payload.get("args"), dict):
            return {
                "type": "tool",
                "name": str(payload.get("name")),
                "args": dict(payload.get("args") or {}),
            }
        tool_calls = payload.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            tc0 = tool_calls[0] if isinstance(tool_calls[0], dict) else None
            if tc0 and tc0.get("name"):
                args = tc0.get("args") or tc0.get("arguments") or {}
                return {
                    "type": "tool",
                    "name": str(tc0.get("name")),
                    "args": args if isinstance(args, dict) else {},
                }

    # 2) XML fallback for <tool_call> ... </tool_call>
    xml_match = re.search(
        r"<tool_call>\s*<name>(.*?)</name>\s*<arguments>(.*?)</arguments>\s*</tool_call>",
        llm_output or "",
        re.DOTALL | re.IGNORECASE,
    )
    if xml_match:
        tool_name = (xml_match.group(1) or "").strip()
        args_content = (xml_match.group(2) or "").strip()
        try:
            tool_args = json.loads(args_content)
            if isinstance(tool_args, dict):
                return {"type": "tool", "name": tool_name, "args": tool_args}
        except Exception:  # noqa: BLE001
            pass

        arg_pairs = re.findall(r"<([A-Za-z_][\w\-]*)>(.*?)</\1>", args_content, re.DOTALL)
        if arg_pairs:
            tool_args = {k.strip(): _coerce_xml_scalar(v) for k, v in arg_pairs}
            return {"type": "tool", "name": tool_name, "args": tool_args}

    return {"type": "no_tool_call"}


def _recover_tool_calls_from_text(text: str) -> list[dict[str, Any]]:
    """Recover tool calls from malformed LLM content (JSON/XML/noisy markdown)."""
    if not text.strip():
        return []

    known = {t.name for t in TOOLS}
    recovered: list[dict[str, Any]] = []

    # First pass: single-call robust parser requested for noisy outputs.
    parsed = parse_llm_tool_response(text)
    if parsed.get("type") == "tool" and parsed.get("name") in known:
        recovered.append(
            {
                "name": str(parsed.get("name")),
                "args": parsed.get("args") if isinstance(parsed.get("args"), dict) else {},
                "id": f"fallback-{uuid4()}",
                "type": "tool_call",
            }
        )

    # Second pass: allow multiple XML tool_call blocks if present.
    xml_blocks = re.finditer(
        r"<tool_call>\s*<name>(.*?)</name>\s*<arguments>(.*?)</arguments>\s*</tool_call>",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    for m in xml_blocks:
        name = (m.group(1) or "").strip()
        if name not in known:
            continue
        args_body = (m.group(2) or "").strip()
        args: dict[str, Any] = {}
        try:
            obj = json.loads(args_body)
            if isinstance(obj, dict):
                args = obj
        except Exception:  # noqa: BLE001
            pairs = re.findall(r"<([A-Za-z_][\w\-]*)>(.*?)</\1>", args_body, re.DOTALL)
            if pairs:
                args = {k.strip(): _coerce_xml_scalar(v) for k, v in pairs}
        recovered.append(
            {
                "name": name,
                "args": args,
                "id": f"fallback-{uuid4()}",
                "type": "tool_call",
            }
        )

    # De-dup by (name,args) while preserving order.
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for tc in recovered:
        sig = json.dumps({"name": tc.get("name"), "args": tc.get("args")}, sort_keys=True)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(tc)
    return out


def _latest_trimmed_sequence_with_index(messages: list) -> tuple[Optional[dict], int]:
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if not isinstance(m, ToolMessage):
            continue
        if getattr(m, "name", None) != "get_trimmed_sequence":
            continue
        data = _parse_tool_json(str(m.content))
        if data:
            return data, i
    return None, -1


def _stable_progress_payload(name: str, data: dict) -> Optional[str]:
    """Canonical payload snapshot for no-progress loop detection."""
    if name == "get_trimmed_sequence":
        snap = {
            "variable": data.get("variable"),
            "origin_resolved": data.get("origin_resolved"),
            "total_steps": data.get("total_steps"),
            "sequence": data.get("sequence") or [],
            "scope_mode": data.get("scope_mode"),
            "trace_function": data.get("trace_function"),
        }
        return json.dumps(snap, sort_keys=True, separators=(",", ":"))
    if name == "get_caller_context":
        snap = {
            "current_function": data.get("current_function"),
            "callers": data.get("callers") or [],
            "unresolved": data.get("unresolved"),
        }
        return json.dumps(snap, sort_keys=True, separators=(",", ":"))
    return None


def _origin_progress_stalled(messages: list) -> bool:
    """True when origin-tracing tool hops repeat without new information."""
    relevant: list[tuple[str, str]] = []
    for m in messages:
        if not isinstance(m, ToolMessage):
            continue
        name = getattr(m, "name", None)
        if name not in _ORIGIN_TOOLS:
            continue
        data = _parse_tool_json(str(m.content))
        if not data:
            continue
        stable = _stable_progress_payload(str(name), data)
        if not stable:
            continue
        relevant.append((str(name), stable))

    if len(relevant) < 2:
        return False

    # Same tool called twice with identical result.
    if relevant[-1] == relevant[-2]:
        return True

    # Alternating seq/caller cycle repeating the same pair.
    if len(relevant) >= 4:
        a, b, c, d = relevant[-4], relevant[-3], relevant[-2], relevant[-1]
        if a[0] != b[0] and c[0] != d[0] and a[0] == c[0] and b[0] == d[0]:
            if a[1] == c[1] and b[1] == d[1]:
                return True
    return False


def _tool_payload_signature(msg: ToolMessage) -> str:
    name = str(getattr(msg, "name", "") or "")
    payload = _parse_tool_json(str(msg.content))
    if payload is None:
        payload_txt = str(msg.content).strip()
    else:
        payload_txt = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return f"{name}|{payload_txt}"


def _any_tool_loop_stalled(messages: list) -> bool:
    """Detect repeated identical tool-result patterns across all tools."""
    tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
    if len(tool_msgs) < 2:
        return False

    sigs = [_tool_payload_signature(m) for m in tool_msgs]

    # Same tool result repeated back-to-back.
    if len(sigs) >= 2 and sigs[-1] == sigs[-2]:
        return True

    # Two-step cycle repeated (A,B,A,B).
    if len(sigs) >= 4:
        if sigs[-4] == sigs[-2] and sigs[-3] == sigs[-1]:
            return True

    # Three-step cycle repeated (A,B,C,A,B,C).
    if len(sigs) >= 6:
        if sigs[-6:-3] == sigs[-3:]:
            return True

    return False


_OFFTOPIC_HINTS = (
    "autosar",
    "overview",
    "summary table",
    "embedded software module",
    "architecture",
    "electronic control unit",
)


def _is_offtopic_no_tool_text(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    # Legit planner no-op marker.
    if '{"type":"no_tool_call"}' in t.replace(" ", ""):
        return False
    # Obvious tool call/XML snippets are not off-topic.
    if "<tool_call>" in t or '"tool_calls"' in t or '"name"' in t and '"arguments"' in t:
        return False
    return any(h in t for h in _OFFTOPIC_HINTS)


def _latest_index_operand(messages: list) -> Optional[str]:
    for m in reversed(messages):
        if not isinstance(m, ToolMessage):
            continue
        if getattr(m, "name", None) != "get_function_snippet":
            continue
        data = _parse_tool_json(str(m.content))
        if not data:
            continue
        ops = data.get("index_operands_at_alarm") or []
        if not isinstance(ops, list) or not ops:
            continue
        first = ops[0] if isinstance(ops[0], dict) else None
        if not first:
            continue
        operand = str(first.get("operand_text") or "").strip()
        if operand:
            return operand
    return None


def _latest_index_operand_symbol(messages: list) -> Optional[str]:
    for m in reversed(messages):
        if not isinstance(m, ToolMessage):
            continue
        if getattr(m, "name", None) != "get_function_snippet":
            continue
        data = _parse_tool_json(str(m.content))
        if not data:
            continue
        ops = data.get("index_operands_at_alarm") or []
        if not isinstance(ops, list):
            continue
        for op in ops:
            if not isinstance(op, dict):
                continue
            resolved = str(op.get("resolved_operand_symbol") or "").strip()
            if resolved:
                return resolved
            operand = str(op.get("operand_text") or "").strip()
            if operand:
                return operand
    return None


def _latest_scope_for_symbol(messages: list, symbol: str) -> Optional[str]:
    tail = _ident_tail(symbol)
    if not tail:
        return None
    for m in reversed(messages):
        if not isinstance(m, ToolMessage):
            continue
        if getattr(m, "name", None) != "get_variable_scope":
            continue
        data = _parse_tool_json(str(m.content))
        if not data:
            continue
        sym = _ident_tail(str(data.get("symbol") or ""))
        if sym == tail:
            scope = str(data.get("scope") or "").strip().lower()
            if scope:
                return scope
    return None


def _has_useful_index_origin_tool(messages: list) -> bool:
    data = _latest_trimmed_sequence(messages)
    if not data:
        return False
    if not bool(data.get("origin_resolved")):
        return False
    operand = _latest_index_operand(messages)
    if not operand:
        return True
    traced = str(data.get("variable") or "")
    if not traced:
        return False
    return _ident_tail(traced) == _ident_tail(operand)


def _needs_caller_followup(messages: list) -> bool:
    data, seq_idx = _latest_trimmed_sequence_with_index(messages)
    if not data:
        return False
    if bool(data.get("origin_resolved")):
        return False

    traced = str(data.get("variable") or "")
    if not traced:
        return False

    scope = _latest_scope_for_symbol(messages, traced)
    if scope != "parameter":
        return False

    if seq_idx < 0:
        return False
    for m in messages[seq_idx + 1 :]:
        if isinstance(m, ToolMessage) and getattr(m, "name", None) == "get_caller_context":
            return False
    return True


def _pending_hop_target(messages: list) -> Optional[str]:
    target: Optional[str] = None
    for m in messages:
        if not isinstance(m, ToolMessage):
            continue
        name = getattr(m, "name", None)
        data = _parse_tool_json(str(m.content))
        if not data:
            continue
        if name == "get_caller_context":
            if data.get("unresolved") is True:
                target = None
                continue
            callers = data.get("callers") or []
            if callers and isinstance(callers[0], dict):
                target = str(callers[0].get("function") or "").strip() or None
        elif name == "get_trimmed_sequence" and target:
            # Any follow-up trimmed sequence after a caller hop satisfies this gate.
            target = None
    return target


def _needs_call_site_followup(messages: list) -> bool:
    return _needs_caller_followup(messages)


def _saw_parameter_note(messages: list) -> bool:
    for m in messages:
        if not isinstance(m, ToolMessage):
            continue
        if getattr(m, "name", None) != "get_variable_scope":
            continue
        data = _parse_tool_json(str(m.content))
        if data and data.get("scope") == "parameter":
            return True
    return False


def _is_oob_alarm(store: DataStore, order_id: int) -> bool:
    alarm = store.get_alarm(order_id)
    if alarm is None:
        return False
    cat = (alarm.category or "").lower()
    return "out-of-bound" in cat or "out of bound" in cat or "array_out_of_bounds" in cat


def _build_origin_nudge(messages: list) -> str:
    lines = [INDEX_ORIGIN_NUDGE]
    seq = _latest_trimmed_sequence(messages) or {}
    operand = _latest_index_operand(messages)
    if operand:
        lines.append(f"Index operand candidate from snippet: {operand!r}")
    if seq:
        lines.append(
            f"Latest variable={seq.get('variable')!r}, "
            f"origin_resolved={seq.get('origin_resolved')}, total_steps={seq.get('total_steps')}"
        )
    return "\n".join(lines)


def _tool_call_templates(
    *,
    tool_name: str,
    order_id: int,
    variable: str = "",
) -> str:
    if tool_name == "get_caller_context":
        args = {
            "parameter": variable,
            "alarm_order_id": order_id,
        }
    elif tool_name == "get_trimmed_sequence":
        args = {
            "alarm_order_id": order_id,
            "variable": variable,
        }
    elif tool_name == "get_variable_scope":
        args = {
            "symbol": variable,
            "alarm_order_id": order_id,
        }
    elif tool_name == "get_declaration_info":
        args = {
            "symbol": variable,
        }
    else:
        args = {
            "alarm_order_id": order_id,
        }

    json_call = json.dumps({"name": tool_name, "arguments": args}, indent=2)
    xml_parts = [
        "<tool_call>",
        f"  <name>{tool_name}</name>",
        "  <arguments>",
    ]
    for k, v in args.items():
        xml_parts.append(f"    <{k}>{v}</{k}>")
    xml_parts += [
        "  </arguments>",
        "</tool_call>",
    ]
    return (
        "Call exactly this tool next.\n"
        "JSON tool-call schema:\n"
        f"{json_call}\n\n"
        "XML tool-call schema:\n"
        + "\n".join(xml_parts)
    )


def _build_dynamic_nudge(messages: list, order_id: int) -> str:
    operand = _latest_index_operand_symbol(messages) or _latest_index_operand(messages) or ""
    seq = _latest_trimmed_sequence(messages) or {}
    traced = str(seq.get("variable") or "").strip() or operand

    if "get_function_snippet" not in _tool_names_used(messages):
        return (
            "You returned reasoning but no tool call. Retrieve the alarm-site index expression first.\n"
            + _tool_call_templates(
                tool_name="get_function_snippet",
                order_id=order_id,
            )
        )

    if not _has_useful_index_origin_tool(messages):
        target = traced or operand
        return (
            "You must trace the index operand now; do not continue with prose-only reasoning.\n"
            f"Target symbol: {target}\n"
            + _tool_call_templates(
                tool_name="get_trimmed_sequence",
                order_id=order_id,
                variable=target,
            )
        )

    if _needs_call_site_followup(messages):
        target = traced or operand
        return (
            "Caller-side parameter origin is still open.\n"
            f"Target parameter: {target}\n"
            + _tool_call_templates(
                tool_name="get_caller_context",
                order_id=order_id,
                variable=target,
            )
        )

    if _guards_missing(messages):
        return (
            "Collect guard/sentinel evidence from source before finalizing.\n"
            + _tool_call_templates(
                tool_name="get_function_snippet",
                order_id=order_id,
            )
        )

    target = traced or operand
    return (
        "Ground the verdict with one more trace step on the same index symbol.\n"
        f"Target symbol: {target}\n"
        + _tool_call_templates(
            tool_name="get_trimmed_sequence",
            order_id=order_id,
            variable=target,
        )
    )


def _build_call_site_nudge(messages: list) -> str:
    return CALLER_NUDGE


def _build_guard_nudge(messages: list) -> str:
    return (
        "Before finishing, call get_function_snippet in the current trace context "
        "and cite any in-snippet guard/sentinel/shift evidence directly."
    )


def _parameter_path_open(messages: list) -> bool:
    return _needs_call_site_followup(messages)


def _guards_missing(messages: list) -> bool:
    return "get_function_snippet" not in _tool_names_used(messages)


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
_TRANSFORM_RE = re.compile(r">>|<<|(?<!&)&(?!&)|(?<!\|)\|(?!\|)|\^")

_SITE_TOOL_NAMES = (
    "get_function_snippet",
    "get_trimmed_sequence",
    "get_variable_scope",
    "get_declaration_info",
)
_HOP_TOOL_NAMES = (
    "get_caller_context",
    "get_trimmed_sequence",
    "get_function_snippet",
)
_ORIGIN_CORE_NAMES = (
    "get_trimmed_sequence",
    "get_variable_scope",
    "get_declaration_info",
)
_CALLSITE_TOOL_NAMES = (
    "get_caller_context",
    "get_trimmed_sequence",
    "get_function_snippet",
)
_GUARD_TOOL_NAMES = (
    "get_function_snippet",
    "get_declaration_info",
    "get_variable_scope",
)
_FOLLOW_TOOL_NAMES = (
    "get_function_snippet",
    "get_trimmed_sequence",
    "get_caller_context",
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
    """Identifiers in plain C-like text (no semantic inference)."""
    found: list[str] = []
    seen: set[str] = set()
    for m in _IDENT_TOKEN_RE.finditer(text or ""):
        name = _ident_norm(m.group(1))
        if not name or name.lower() in _SKIP_IDENT_TOKENS or name in seen:
            continue
        seen.add(name)
        found.append(name)
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


def _payload_writes(data: dict) -> list:
    writes = data.get("sequence") or data.get("writes") or []
    if not isinstance(writes, list):
        return []
    return [w for w in writes if isinstance(w, dict) and w.get("access") == "write"]


def _looks_like_aggregate(name: str) -> bool:
    tail = _ident_tail(name)
    low = tail.lower()
    return low.endswith(
        ("_ast", "_pst", "_table", "_buf", "_buffer", "_array", "_queue", "_lut")
    )


def _named_next_from_payload(data: dict) -> list[str]:
    """Identifiers explicitly surfaced by caller context argument expressions."""
    named: list[str] = []
    seen: set[str] = set()

    def _add(raw: str) -> None:
        name = _ident_norm(raw)
        if not name or name.lower() in _SKIP_IDENT_TOKENS:
            return
        if _looks_like_aggregate(name):
            return
        key = _ident_tail(name)
        if key in seen:
            return
        seen.add(key)
        named.append(name)

    for c in data.get("callers") or []:
        if not isinstance(c, dict):
            continue
        for ident in c.get("argument_identifiers") or []:
            _add(str(ident))
    return named


def symbol_explicitly_unresolvable(messages: list, symbol: str) -> bool:
    """True when caller context explicitly reports unresolved."""
    if not symbol:
        return False
    for m in messages:
        if not isinstance(m, ToolMessage):
            continue
        data = _parse_tool_json(str(m.content))
        if not data:
            continue
        if getattr(m, "name", None) == "get_caller_context" and data.get("unresolved") is True:
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
        if getattr(m, "name", None) == "get_trimmed_sequence" and _payload_writes(data):
            return True
    return False


def pending_named_hop(messages: list) -> Optional[str]:
    return _pending_hop_target(messages)


def tools_returned_closing_evidence(messages: list) -> bool:
    """Bounds/snippet/writes evidence present in tool payloads."""
    for m in messages:
        if not isinstance(m, ToolMessage):
            continue
        data = _parse_tool_json(str(m.content))
        if not data:
            continue
        name = getattr(m, "name", None)
        if name == "get_declaration_info":
            size = data.get("array_size")
            if data.get("found") and size not in (None, 0, "0"):
                return True
            nested = data.get("nested_array_fields") or []
            if any(
                isinstance(f, dict) and f.get("array_size") not in (None, 0, "0")
                for f in nested
            ):
                return True
        if name == "get_function_snippet" and data.get("snippet"):
            return True
        if name == "get_trimmed_sequence" and _payload_writes(data):
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
    elif "get_trimmed_sequence" not in used:
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
            "Suggested: get_trimmed_sequence(variable=<same operand>) after hop, "
            "then get_caller_context again only if origin_resolved stays false."
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
        if _origin_progress_stalled(messages):
            return False
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
        if getattr(msg, "name", None) == "get_trimmed_sequence":
            _add(msg)
            break

    sticky = _ORIGIN_TOOLS | {
        "get_function_snippet",
        "get_declaration_info",
        "get_variable_scope",
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
    stalled = _origin_progress_stalled(messages)
    completeness = (be.get("evidence_completeness") or "").strip()
    if not has_origin:
        be["evidence_completeness"] = "not traced"
    elif stalled and completeness in {"", "fully traced"}:
        be["evidence_completeness"] = "partially traced"
    elif needs_cs:
        if completeness in {"", "fully traced"}:
            be["evidence_completeness"] = "partially traced"
    elif not completeness:
        be["evidence_completeness"] = "partially traced"
    be.setdefault("symbol", "")
    if "declared_size" not in be:
        be["declared_size"] = None
    if "target_capacity" not in be:
        be["target_capacity"] = be.get("declared_size")
    be.setdefault("index_expression", "")
    be.setdefault("index_inferred_range", "")
    be.setdefault("index_safe_range", "")
    guards = be.get("guards_found")
    if not isinstance(guards, list):
        be["guards_found"] = []
    be.setdefault("index_origin_summary", "")
    data["bounds_evidence"] = be
    # Structured observability for completeness assignment behavior.
    _log(
        "[report.completeness] "
        + json.dumps(
            {
                "order_id": order_id,
                "has_useful_index_origin": has_origin,
                "parameter_path_open": needs_cs,
                "saw_parameter_note": _saw_parameter_note(messages),
                "has_call_site_followup": not _needs_call_site_followup(messages),
                "origin_progress_stalled": stalled,
                "input_evidence_completeness": completeness,
                "computed_evidence_completeness": be.get("evidence_completeness"),
                "parameter_origin_fully_traced_structurally_reachable": bool(
                    has_origin
                    and _saw_parameter_note(messages)
                    and not needs_cs
                ),
            },
            sort_keys=True,
        )
    )
    return data


def _latest_non_tool_report_json(messages: list) -> Optional[dict]:
    """Try to reuse JSON from the latest non-tool AI message before re-prompting."""
    required_report_keys = {
        "classification",
        "comment",
        "confidence",
        "summary",
    }
    for msg in reversed(messages):
        if not isinstance(msg, AIMessage):
            continue
        if getattr(msg, "tool_calls", None):
            continue
        content = getattr(msg, "content", "")
        if isinstance(content, list):
            text = "\n".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            ).strip()
        else:
            text = str(content or "").strip()
        if not text or "{" not in text:
            continue
        try:
            data = _normalize_report_dict(extract_json(text))
            if str(data.get("type") or "").strip().lower() == "no_tool_call":
                continue
            if not any(k in data for k in required_report_keys):
                continue
            return data
        except Exception:  # noqa: BLE001
            continue
    return None


def _build_sequence_context(store: DataStore, order_id: int) -> str:
    """Return a compact write-node → alarm-read roadmap to seed the initial prompt."""
    alarm = store.get_alarm(order_id)
    if alarm is None:
        return ""
    parsed = alarm.parsed_location
    if not parsed:
        return ""
    alarm_line = parsed["line"]

    # Collect DF records within ±2 lines of the alarm.
    site_records: list = []
    for ln in range(alarm_line - 2, alarm_line + 3):
        site_records.extend(store.data_flow_by_line.get(ln, []))
    if not site_records:
        return ""

    # Pick primary variable by frequency at the site.
    var_freq: dict[str, int] = {}
    for r in site_records:
        bare = (r.variable or "").split("@", 1)[0].strip('"').strip()
        if bare:
            var_freq[bare] = var_freq.get(bare, 0) + 1
    if not var_freq:
        return ""
    primary = max(var_freq, key=var_freq.get)

    # All write events for the primary variable, deduped by (function, line).
    all_events = store.data_flow_by_variable.get(primary, [])
    seen_keys: set = set()
    writes: list = []
    for r in sorted(all_events, key=lambda r: r.line or 0):
        if (r.access or "").lower() != "write":
            continue
        key = (r.function, r.line)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        writes.append(r)

    lines = [
        "--- Sequence context (use as navigation map) ---",
        f"Alarm site (READ node — start here): {alarm.location}",
        f"Category: {alarm.category or 'Out-of-bound array access'}",
        f"Primary variable: {primary}",
    ]
    if writes:
        lines.append(f"Write nodes for '{primary}' ({min(len(writes), 8)} shown):")
        for i, w in enumerate(writes[:8], 1):
            loc = w.location or f"line {w.line}"
            lines.append(f"  {i}. {w.function}  @  {loc}")
    else:
        lines.append(
            f"No write nodes indexed for '{primary}' "
            "— index is likely a function parameter (trace via call sites)."
        )
    lines += [
        "",
        "Follow navigation strategy: get_function_snippet for index_operands_at_alarm "
        "→ get_variable_scope/get_declaration_info for that operand → "
        "get_trimmed_sequence on operand → if origin_resolved=false and scope=parameter "
        "then get_caller_context and repeat get_trimmed_sequence.",
        "--- End sequence context ---",
    ]
    return "\n".join(lines)


def _degraded_report_payload(order_id: int, messages: list, reason: str) -> dict:
    """Guaranteed schema-shaped fallback when model output is empty/unparseable."""
    return {
        "alarm_order_id": order_id,
        "affected_symbols": [],
        "function_name": None,
        "function_snippet": "",
        "manipulation_sequence": [],
        "bounds_evidence": {
            "symbol": "",
            "declared_size": None,
            "index_expression": "",
            "index_inferred_range": "",
            "index_safe_range": "",
            "target_capacity": None,
            "guards_found": [],
            "index_origin_summary": "",
            "evidence_completeness": "not traced",
        },
        "summary": "Investigation report degraded due to unparseable model output.",
        "classification": "review",
        "comment": f"Report synthesis fallback: {reason}",
        "confidence": "low",
        "tools_used": sorted(_tool_names_used(messages)),
    }


def _maybe_close_case_reprompt(
    llm: Any,
    data: dict,
    messages: list,
    pack: str,
    *,
    order_id: int,
) -> dict:
    """One extra model turn when the draft hedges past evidence already in tools.

    Does not write true/false in Python. If the model still returns review, keep it.
    """
    reprompt_enabled = close_case_reprompt_enabled()
    be = data.get("bounds_evidence")
    completeness = be.get("evidence_completeness") if isinstance(be, dict) else None
    draft_cls = data.get("classification")
    closing_evidence_present = tools_returned_closing_evidence(messages)
    reprompt_trigger_fired = bool(
        reprompt_enabled
        and draft_cls == "review"
        and completeness == "fully traced"
        and closing_evidence_present
    )
    _log(
        "[report.reprompt_gate] "
        + json.dumps(
            {
                "order_id": order_id,
                "draft_classification": draft_cls,
                "computed_evidence_completeness": completeness,
                "closing_evidence_present": closing_evidence_present,
                "reprompt_enabled": reprompt_enabled,
                "reprompt_trigger_fired": reprompt_trigger_fired,
            },
            sort_keys=True,
        )
    )
    if not reprompt_enabled:
        return data
    if draft_cls != "review":
        return data
    if completeness != "fully traced":
        return data
    if not closing_evidence_present:
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
    # Single-model mode: use the same backend/model for tool planning and report.
    # This avoids planner/report drift and keeps behavior deterministic per run.
    if model:
        llm = build_llm(model=model, role="report")
    else:
        llm = build_llm(role="report")
    report_llm = llm
    current_backend = "ollama" if using_ollama() else "nvidia"

    def _try_runtime_fallback(exc: Exception, stage: str) -> bool:
        nonlocal llm, report_llm, current_backend
        if current_backend != "ollama":
            return False
        if llm_backend_override() == "ollama":
            return False
        if not nvidia_configured() or not ollama_runtime_fallback_enabled():
            return False
        if not _is_ollama_runtime_error(exc):
            return False
        _log(f"[llm] ollama runtime failure during {stage}: {exc}")
        _log("[llm] switching runtime backend to nvidia for this investigation")
        llm = build_llm(model=model, backend="nvidia", role="report")
        report_llm = llm
        current_backend = "nvidia"
        return True

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
        response: Optional[AIMessage] = None
        try:
            response = runner.invoke(trimmed)
        except Exception as exc:  # noqa: BLE001
            if _try_runtime_fallback(exc, "agent tool-planning"):
                runner = _bind_llm_tools(llm, visible, force=force)
                response = runner.invoke(trimmed)
            if force:
                _log(f"[agent] tool_choice=any failed ({exc}); retry unbound")
                try:
                    response = _bind_llm_tools(llm, visible, force=False).invoke(trimmed)
                except Exception as exc2:  # noqa: BLE001
                    if _try_runtime_fallback(exc2, "agent unbound retry"):
                        response = _bind_llm_tools(llm, visible, force=False).invoke(trimmed)
                    else:
                        raise
            else:
                raise
        if response is None:
            raise RuntimeError("Agent planning did not produce a response.")

        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            response_text = _message_text(response)
            fallback_calls = _recover_tool_calls_from_text(response_text)
            if fallback_calls:
                _log(f"[agent] recovered {len(fallback_calls)} tool call(s) from XML/JSON fallback parser")
                response = AIMessage(
                    content=response.content,
                    tool_calls=fallback_calls,
                    additional_kwargs=getattr(response, "additional_kwargs", {}) or {},
                    id=getattr(response, "id", None),
                )
                tool_calls = fallback_calls
            elif _is_offtopic_no_tool_text(response_text):
                _log("[agent] off-topic no-tool response detected; enforcing strict no_tool_call marker")
                response = AIMessage(content='{"type":"no_tool_call"}')

        _log(f"[agent] tool_calls={len(tool_calls)}")
        no_tool_streak = 0 if tool_calls else int(state.get("consecutive_no_tool_calls") or 0) + 1
        return {
            "messages": [response],
            "consecutive_no_tool_calls": no_tool_streak,
        }

    def tools_node(state: AgentState) -> dict:
        result = ToolNode(TOOLS).invoke(state)
        rounds = int(state.get("tool_rounds") or 0) + 1
        _log(f"[tools] completed round={rounds}")
        return {
            **result,
            "tool_rounds": rounds,
            "consecutive_no_tool_calls": 0,
        }

    def nudge_node(state: AgentState) -> dict:
        # Prefer origin → named next-hop → call-site follow-up → guards.
        dynamic = _build_dynamic_nudge(state["messages"], state["order_id"])
        if not _has_useful_index_origin_tool(state["messages"]):
            n = int(state.get("origin_nudges") or 0) + 1
            text = dynamic
            _log(f"[nudge] useful index-origin slice required (nudge={n})")
            return {
                "origin_nudges": n,
                "messages": [HumanMessage(content=text)],
            }
        pending = pending_named_hop(state["messages"]) if next_hop_gate_enabled() else None
        if pending:
            n = int(state.get("next_hop_nudges") or 0) + 1
            text = dynamic + "\n\n" + _build_next_hop_nudge(state["messages"])
            _log(f"[nudge] next-hop identifier {pending!r} required (nudge={n})")
            return {
                "next_hop_nudges": n,
                "messages": [HumanMessage(content=text)],
            }
        if _needs_call_site_followup(state["messages"]):
            n = int(state.get("call_site_nudges") or 0) + 1
            text = dynamic + "\n\n" + _build_call_site_nudge(state["messages"])
            _log(f"[nudge] call-site argument follow-up required (nudge={n})")
            return {
                "call_site_nudges": n,
                "messages": [HumanMessage(content=text)],
            }
        n = int(state.get("guard_nudges") or 0) + 1
        text = dynamic + "\n\n" + _build_guard_nudge(state["messages"])
        _log(f"[nudge] function snippet evidence required (nudge={n})")
        return {
            "guard_nudges": n,
            "messages": [HumanMessage(content=text)],
        }

    def after_nudge(state: AgentState) -> Literal["agent", "report"]:
        streak = int(state.get("consecutive_no_tool_calls") or 0)
        if streak >= 2:
            _log(
                "[route] consecutive no-tool-call guard tripped after nudge "
                f"(streak={streak}) -> report"
            )
            return "report"
        return "agent"

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
        no_tool_streak = int(state.get("consecutive_no_tool_calls") or 0)

        if _any_tool_loop_stalled(msgs):
            _log("[route] repeated tool-result pattern detected -> report")
            return "report"

        if no_tool_streak >= 2:
            _log(
                "[route] consecutive no-tool-call guard tripped in agent router "
                f"(streak={no_tool_streak}) -> report"
            )
            return "report"

        if oob and _origin_progress_stalled(msgs):
            _log("[route] origin-tracing stalled with repeated identical tool outputs → report")
            return "report"

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
                _log("[route] missing get_function_snippet → nudge")
                return "nudge"
            if force_n < 2:
                _log("[route] guards still missing after nudge → retry/force")
                return "retry"
        return "report"

    def after_tools(state: AgentState) -> Literal["agent", "report"]:
        if _count_tool_messages(state["messages"]) >= _MAX_TOOL_MESSAGES:
            _log(f"[route] tool-message ceiling {_MAX_TOOL_MESSAGES} → report")
            return "report"
        if _any_tool_loop_stalled(state["messages"]):
            _log("[route] repeated tool-result pattern detected after tools -> report")
            return "report"
        return "agent"

    def report_node(state: AgentState) -> dict:
        _log("[report] building structured JSON report ...")
        pack = _pack_report_evidence(state["messages"])
        used = sorted(_tool_names_used(state["messages"]))
        data: Optional[dict] = _latest_non_tool_report_json(state["messages"])
        if data is not None:
            _log("[report] reusing JSON from latest no-tool AI message")
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

        if data is None:
            try:
                structured = report_llm.with_structured_output(AlarmInvestigationReport)
                result = structured.invoke([system, human])
                if isinstance(result, AlarmInvestigationReport):
                    data = result.model_dump()
                elif isinstance(result, dict):
                    data = result
            except Exception as exc:  # noqa: BLE001
                if _try_runtime_fallback(exc, "report structured pass"):
                    try:
                        structured = report_llm.with_structured_output(AlarmInvestigationReport)
                        result = structured.invoke([system, human])
                        if isinstance(result, AlarmInvestigationReport):
                            data = result.model_dump()
                        elif isinstance(result, dict):
                            data = result
                    except Exception as exc2:  # noqa: BLE001
                        _log(
                            "[report] with_structured_output failed after fallback "
                            f"({exc2}); JSON extract fallback"
                        )
                else:
                    _log(f"[report] with_structured_output failed ({exc}); JSON extract fallback")

        if data is None:
            try:
                raw = report_llm.invoke([system, human])
            except Exception as exc:  # noqa: BLE001
                if _try_runtime_fallback(exc, "report JSON pass"):
                    raw = report_llm.invoke([system, human])
                else:
                    raise
            content = raw.content if hasattr(raw, "content") else str(raw)
            if isinstance(content, list):
                content = "\n".join(
                    b.get("text", "") if isinstance(b, dict) else str(b)
                    for b in content
                )
            try:
                data = extract_json(str(content))
            except Exception as exc:  # noqa: BLE001
                _log(f"[report] JSON extract failed ({exc}); using degraded report")
                data = _degraded_report_payload(
                    state["order_id"],
                    state["messages"],
                    "final report output was empty or not JSON",
                )

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

        if _origin_progress_stalled(state["messages"]):
            be = data.get("bounds_evidence")
            if isinstance(be, dict):
                if be.get("evidence_completeness") == "fully traced":
                    be["evidence_completeness"] = "partially traced"
                elif not be.get("evidence_completeness"):
                    be["evidence_completeness"] = "not traced"
            if data.get("confidence") in {"high", "medium"}:
                data["confidence"] = "low"
            _log("[report] downgraded confidence/completeness due to stalled origin tracing")

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

        # "not traced" must not carry high confidence.
        be = data.get("bounds_evidence")
        if isinstance(be, dict) and be.get("evidence_completeness") == "not traced":
            if data.get("confidence") == "high":
                data["confidence"] = "medium"
                _log("[report] capped confidence high→medium (evidence not traced)")

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
            repair_prompt = [
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
            try:
                repair = report_llm.invoke(repair_prompt)
            except Exception as exc:  # noqa: BLE001
                if _try_runtime_fallback(exc, "report repair pass"):
                    repair = report_llm.invoke(repair_prompt)
                else:
                    raise
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
            report_llm,
            data,
            state["messages"],
            pack,
            order_id=state["order_id"],
        )

        try:
            report = AlarmInvestigationReport.model_validate(data)
            return {"report": report.model_dump()}
        except Exception as exc:  # noqa: BLE001
            _log(f"[report] model validation failed ({exc}); using degraded report")
            fallback = _degraded_report_payload(
                state["order_id"],
                state["messages"],
                "report validation failed",
            )
            fallback = _fill_report_metadata(
                _normalize_report_dict(fallback),
                store,
                state["order_id"],
                state["messages"],
            )
            report = AlarmInvestigationReport.model_validate(fallback)
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
    graph.add_conditional_edges(
        "nudge", after_nudge, {"agent": "agent", "report": "report"}
    )
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
        '"index_expression":str,"index_inferred_range":str,"index_safe_range":str,'
        '"target_capacity":int|str|null,"guards_found":[str],'
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
    if "summary" not in out or out.get("summary") is None:
        out["summary"] = ""
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
        seq_context = _build_sequence_context(store, order_id)
        initial_human = (
            f"Investigate Astrée alarm Order id {order_id}.\n\n"
            + (seq_context + "\n\n" if seq_context else "")
            + "Follow the navigation strategy in the system prompt. Use tools only."
        )
        initial = {
            "order_id": order_id,
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=initial_human),
            ],
            "report": None,
            "tool_rounds": 0,
            "origin_nudges": 0,
            "call_site_nudges": 0,
            "guard_nudges": 0,
            "next_hop_nudges": 0,
            "force_retries": 0,
            "consecutive_no_tool_calls": 0,
        }
        yield {"type": "status", "message": "Agent is deciding which tools to call…"}
        yield {
            "type": "status",
            "message": (
                f"Calling {report_backend_label()} now for tool planning and report "
                "generation (single-model mode) — the first response often "
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
                            "Index origin is still incomplete — nudging agent "
                            "to run trimmed sequence and caller-context steps."
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
                        normalized_no_tool = text.replace(" ", "").replace("\n", "")
                        if text and not tool_calls and normalized_no_tool not in {
                            '{"type":"no_tool_call"}',
                            '```json{"type":"no_tool_call"}```',
                            '```{"type":"no_tool_call"}```',
                        }:
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
