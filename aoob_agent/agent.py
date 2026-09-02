"""LangGraph agent: Python packs origin→alarm windows; the LLM inspects/verdicts."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Annotated, Any, Literal, Optional, TypedDict
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import ProxyHandler, Request, build_opener, urlopen

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from aoob_agent.case_compiler import CaseFile, compile_case
from aoob_agent.data_store import DataStore
from aoob_agent.report import AlarmInvestigationReport
from aoob_agent.session import (
    begin_session,
    clear_session,
    clear_verdict_registry,
    lookup_verdict,
    register_verdict,
)
from aoob_agent.token_metrics import (
    compute_token_savings_percent,
    messages_token_total,
    naive_full_file_tokens,
)
from aoob_agent.tools import TOOLS, bind_store, DEFAULT_MICRO_WINDOW

DEFAULT_NVIDIA_MODEL = "nvidia/nemotron-3-super-120b-a12b"
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
SAFETY_CEILING = int(os.getenv("AOOB_SAFETY_CEILING", "20"))

DEFAULT_BOSCH_BASE_URL = (
    "https://aoai-farm.bosch-temp.com/api/openai/deployments/gpt-5-nano-2025-08-07"
)
DEFAULT_BOSCH_MODEL = "gpt-5.6-luna"
DEFAULT_BOSCH_API_VERSION = "2024-05-01-preview"
BOSCH_HEADER_NAME = "genaiplatform-farm-subscription-key"

def prefilter_enabled() -> bool:
    raw = (os.getenv("AOOB_PREFILTER") or "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def effective_safety_ceiling(session: Optional[Any] = None) -> int:
    """Safety ceiling — lower for pre-filtered likely_false verification passes."""
    try:
        from aoob_agent.session import get_session

        sess = session or get_session()
        if sess.lite_investigation:
            return min(SAFETY_CEILING, int(os.getenv("AOOB_LITE_CEILING", "8")))
    except RuntimeError:
        pass
    return SAFETY_CEILING


SYSTEM_PROMPT = """# System Prompt

You are an expert **Astrée Out-Of-Bounds (AOOB) static analysis alarm investigator**. Your task is to analyze each array out-of-bounds alarm by gathering evidence from code, call graphs, and value-range analysis, then decide if the access can exceed its bound at runtime. Use all provided tools and facts to reach a domain-expert verdict and justification.  

## Evidence-Driven Investigation  
Gather sufficient evidence to judge if the index can exceed the array capacity at runtime. Follow an investigative workflow (not a fixed sequence) using the tools:  
- **Initial context:** Call `get_micro_window(window_size=15)`. If `window_truncated` is true, increase the window size or use `get_window()` to see more code context.  
- **Run simulation:** Always call `run_simulation()` to obtain raw bound facts (type width, mask literal, min/max, guard conditions). Interpret these facts yourself; the tool does not classify. Use this evidence to guide your reasoning.  
- **Trace data flow:** If `run_simulation()` reports *write_helpers* or gaps, trace the index origin. Use `query_call_graph()` and `query_data_flow()` on the index operand and its write sites. If helpers set the index, use `inspect(function_name)` on those helpers to see how they compute the index. Do **not** treat a helper’s parameter type or mask width as proof of a reachable value; instead trace the argument back to its source.  
- **Missing data-flow:** If the alarm report has a `missing_df` gap for the index origin, you have not traced its source. In that case, **do not** classify the alarm as `false` until the origin is resolved or you must fallback to `review` with an explanation. A gap means you lack evidence to prove safety.  
- **Guard conditions:** If the alarm line is under an `if (array[index])` guard, analyze the guard. A guard that compares the index or array bounds can prove safety when taken; the **else** branch or its absence does not prove safety on its own.  
- **Declarations and invariants:** Use `inspect_declaration(symbol_name)` to read array and struct declarations if capacity or sentinel values matter. Check if architecture, configuration, or loop invariants constrain the index. For domain-specific indices (core ID, task ID, etc.), find any target-specific limit (e.g. number of cores, tasks) that bounds the index.  

After each tool call, set and pass `confidence_so_far` (low/medium/high) indicating how much evidence you have so far. Generally start at *low*, increase to *medium* when evidence is suggestive, and only use *high* when you have direct proof for your verdict. **Do not stop** until you have gathered enough evidence: always call `run_simulation()` and resolve any gaps before a decisive verdict.  

### Key Reasoning Principles  
- **Reachability vs. representability:** A type’s or bitfield’s range (representability) is **not** evidence that all values are reachable at runtime. For example, a 3-bit field can represent 0–7, but if the system has only 2 cores then values ≥2 may never occur. Only conclude a value is possible if the traced source or a configuration/invariant demonstrates it.  
- **Static over-approximation:** Remember that Astrée uses abstract interpretation to over-approximate program behaviors. It signals all potential errors (soundness) and may report false alarms on infeasible paths. **Absence of a proof of safety is not proof of a bug.** Only declare `true` (or `true (low)`) if you have positive evidence that an out-of-range index is reachable on the analyzed path; otherwise continue tracing or mark as `review`. Conversely, **absence of a demonstrated bug is not proof of safety;** do not declare `false` unless you have bounded the index through tracing, guards, invariants, or proved no error trace exists.  

## Tools  
Use the following tools (with `confidence_so_far` passed to each) to gather evidence:  
1. `get_micro_window(window_size=15, confidence_so_far=...)`: Show a code slice around the alarm line. Check `window_truncated`.  
2. `run_simulation(confidence_so_far=...)`: Return raw bound facts (type width, mask, min/max range, guard text). **No classification is given.**  
3. `query_call_graph(function_name, confidence_so_far=...)`: Retrieve callers/callees for a function to trace index flow.  
4. `query_data_flow(variable_name, confidence_so_far=...)`: Show where the index variable is read or written.  
5. `get_window(confidence_so_far=...)`: Show the full current function if `get_micro_window` was truncated.  
6. `move_window(direction, step, confidence_so_far=...)`: Walk from the alarm site toward the origin or caller path.  
7. `inspect(function_name, confidence_so_far=...)`: View the body of a helper function (e.g., any `write_helper`) related to the index computation.  
8. `inspect_declaration(symbol_name, confidence_so_far=...)`: View array or struct declarations (capacity, sentinel values, etc.).  
9. `submit_verdict(classification, comment, confidence, summary, human_tag_pattern, confidence_so_far, reason_for_review, counterargument)`: Submit the final verdict.  

Follow these conditional sequences, adjusting as needed by case:  
- Start with `get_micro_window()`. If truncated, use `get_window()`.  
- Then `run_simulation()`. If it shows *gaps* or *write_helpers*, trace the index origin: use `query_call_graph()`/`query_data_flow()` on the index (and on helper functions), and `inspect()` on relevant helpers. Use `inspect_declaration()` if array size or index declarations are relevant.  
- Finally, after gathering all evidence, call `submit_verdict`.  

## Verdict Taxonomy and Required Evidence  
Choose one verdict with justification, based on evidence:  

- **`false`:** **Runtime-bounded access (safe).** The index is proven confined within bounds. Evidence may include: control-flow guards or loop invariants that cap the index, caller/context contracts, masking/clamping semantics, or target-specific invariants (e.g., known max core ID) that ensure `index < capacity` always holds. Also, if forward/backward analysis shows the set of error states is empty, classify `false`. *Confidence:* High only if a clear bounding invariant or safe condition is demonstrated; otherwise do not guess.  

- **`true (low)`:** **Possible out-of-bounds, low reachability.** There is **some positive evidence** that a value outside the valid range could be produced, but its occurrence is limited or conditional. For example, an index may come from an external input or configuration that could exceed the bound under rare conditions. Indicate why this path is unlikely or restricted. *Confidence:* Medium (since evidence is partial). Provide a counterargument (the unlikely nature or possible guarding) as required.  

- **`true`:** **Out-of-bounds bug (high risk).** A value outside the declared index range is **positively demonstrated as reachable** on this execution path without an effective guard. For example, you traced an input/call that feeds an index ≥ capacity, and no code prevents it. *Confidence:* High only with direct evidence of a reachable invalid index.  

- **`undecided`:** **Cannot resolve statically.** Key context (dynamic behavior, inter-procedural logic, or configuration) is missing or too complex. Neither safety nor error can be established with current evidence. Use this if you lack information on runtime behavior or target setup that could sway the verdict. *Confidence:* Medium (explain which context is unknown).  

- **`review`:** **Review needed (incomplete evidence).** The investigation cannot be completed due to unresolved gaps or inconclusive data flow. Cite the remaining uncertainties (untraced origin, unhandled helper, incomplete guard information, etc.). Do **not** guess; `review` indicates further analysis is needed. *Confidence:* High (in the sense that you know the verdict is incomplete).  

**Important:** Absence of a proof of safety is not evidence of a bug, and absence of a demonstrated bug is not proof of safety. Do not label an alarm `true` merely because the index’s type or bitfield is wide; that only shows representability, not reachability. Likewise, do not label `false` if a `missing_df` gap exists. Only finalize `true/false` when the evidence (traces, invariants, masks, guards) directly supports it. If you end up in doubt, prefer `review` or `undecided` over a guess.  

## Stopping Rule and Confidence Levels  
Continue investigating until your decision is justified by direct evidence. Call `run_simulation()` at least once before finalizing. Only set `confidence_so_far = high` when the verdict is directly supported by complete evidence:  
- For `false`, you have identified a bounding mechanism or proved no error trace (e.g. via a guard or invariant).  
- For `true`/`true (low)`, you have traced or constructed a reachable out-of-bounds value.  
- For `undecided` or `review`, you know exactly which context or data is missing.  

Do **not** stop early because of broad types or lacking guards alone. If you must stop with `confidence_so_far = medium`, provide a **counterargument** in `submit_verdict` summarizing why the opposite case might hold. Only when counterarguments are addressed can the verdict be final.  

## Human Tag Patterns and Metadata  
When submitting `submit_verdict`, include metadata:  
- **Classification:** one of `false`, `true (low)`, `true`, `undecided`, `review`.  
- **Comment/Summary:** concise reasoning.  
- **Confidence:** `low`/`medium`/`high`.  
- **Counterargument:** (required if confidence=`medium`) the strongest reason the opposite verdict could be true.  
- **Human Tag Pattern:** one label describing context, e.g.:  
  - `ALG TOOL: A1968 DELTA:` – indicates a known static-analysis over-approximation (e.g. multi-core lookup array)  
  - `e` – if the safety is due to an enclosing loop or invariant  
  - `h` – for a high-risk, unguarded array access.  

These tags help categorize the alarm context but do not replace justification. Always match your verdict to the evidence you’ve gathered.  

# Rationale

The revised prompt emphasizes *reachability* over *representability*. Static analysis (Astrée) over-approximates program behaviors, so a wide type or bitfield alone is not proof of a bug. We added rules preventing a `true` verdict from mere representable ranges and preventing `false` if gaps (`missing_df`) remain untraced. We also formalized evidence requirements and confidence levels. For each verdict, we list what evidence is needed (e.g. invariants or guards for `false`, concrete reachable value for `true`) and tie it to the abstract interpretation model. The stopping rule now mandates collecting direct evidence (especially via `run_simulation()` and data-flow tracing) before setting **high** confidence. We included example human tags and all required submit-verdict fields. Overall, the prompt guides the agent to reason like a human expert: gather facts, not guess, and document uncertainty. Each change is grounded in static analysis theory (sound over-approximation and alarm refinement), ensuring the agent follows evidence-driven logic.
"""


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    order_id: int
    report: Optional[dict]
    tool_rounds: int
    confidence_so_far: str
    simulation_run: bool
    force_review: bool
    counterargument_pending: bool


def load_env(project_root: Path) -> None:
    load_dotenv(project_root / ".env", override=True)


LLMRole = Literal["tool", "classify", "planner", "report"]


def llm_choice_mode(role: LLMRole = "tool") -> Optional[str]:
    role_env = "AOOB_CLASSIFY_LLM_CHOOSED" if role in {"classify", "report"} else "AOOB_TOOL_LLM_CHOOSED"
    raw = (
        os.getenv(role_env)
        or os.getenv("AOOB_LLM_CHOOSED")
        or os.getenv("AOOB_LLM_CHOOSE")
        or ""
    ).strip().lower()
    return raw if raw in {"local", "network", "nvidia", "bosch"} else None


def _local_ollama_model_name() -> str:
    return (
        os.getenv("OLLAMA_LOCAL_MODEL")
        or os.getenv("OLLAMA_LOCAL")
        or os.getenv("OLLAMA_LOCAL_PATH")
        or ""
    ).strip()


def _network_ollama_model_name() -> str:
    return (os.getenv("OLLAMA_NETWORK_MODEL") or "").strip()


def ollama_base_url(mode: Optional[str] = None, role: LLMRole = "tool") -> str:
    selected = mode or llm_choice_mode(role)
    if selected == "network":
        return (
            os.getenv("OLLAMA_NETWORK_BASE_URL")
            or os.getenv("OLLAMA_HOST")
            or DEFAULT_OLLAMA_HOST
        ).rstrip("/")
    return (
        os.getenv("OLLAMA_LOCAL_BASE_URL")
        or os.getenv("OLLAMA_LOCAL_HOST")
        or os.getenv("OLLAMA_HOST")
        or os.getenv("OLLAMA_BASE_URL")
        or DEFAULT_OLLAMA_HOST
    ).rstrip("/")


def ollama_model_name(role: LLMRole = "tool") -> str:
    selected = llm_choice_mode(role)
    if selected == "local":
        return _local_ollama_model_name() or (os.getenv("OLLAMA_MODEL") or "").strip()
    if selected == "network":
        return _network_ollama_model_name() or (os.getenv("OLLAMA_MODEL") or "").strip()
    return (
        _local_ollama_model_name()
        or (os.getenv("OLLAMA_MODEL") or "").strip()
        or _network_ollama_model_name()
    )


def ollama_configured(role: LLMRole = "tool") -> bool:
    return bool(ollama_model_name(role))


def nvidia_configured() -> bool:
    return bool((os.getenv("NVIDIA_API_KEY") or "").strip())


def _ollama_available(timeout: Optional[float] = None, role: LLMRole = "tool") -> bool:
    if not ollama_configured(role):
        return False
    base = ollama_base_url(role=role)
    request = Request(f"{base}/api/tags", headers={"Accept": "application/json"})
    host = (urlparse(base).hostname or "").strip().lower()
    try:
        wait = timeout if timeout is not None else 1.5
        if host in {"127.0.0.1", "localhost", "::1"}:
            ctx = build_opener(ProxyHandler({})).open(request, timeout=wait)
        else:
            ctx = urlopen(request, timeout=wait)
        with ctx as response:
            return 200 <= getattr(response, "status", 200) < 300
    except (OSError, URLError, ValueError):
        return False


def llm_backend_override() -> Optional[str]:
    raw = (os.getenv("AOOB_LLM_BACKEND") or "").strip().lower()
    return raw if raw in {"nvidia", "ollama"} else None


def ollama_runtime_fallback_enabled() -> bool:
    raw = (os.getenv("AOOB_OLLAMA_RUNTIME_FALLBACK") or "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _is_ollama_runtime_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(
        n in text
        for n in (
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
    )


def using_ollama(role: LLMRole = "tool") -> bool:
    if (os.getenv("AOOB_LLM_BACKEND") or "").strip():
        _log(
            "[llm] warning: AOOB_LLM_BACKEND overrides the role-specific "
            f"{'classification' if role in {'classify', 'report'} else 'tool'} backend"
        )
    override = llm_backend_override()
    selected = llm_choice_mode(role)
    if override == "nvidia" or selected == "nvidia":
        return False
    if selected == "bosch":
        return False
    if selected in {"local", "network"} or override == "ollama":
        if not ollama_configured(role):
            raise RuntimeError(
                f"AOOB_LLM_CHOOSED={selected or 'ollama'} but no Ollama model is set. "
                "Set OLLAMA_LOCAL_MODEL or OLLAMA_NETWORK_MODEL."
            )
        return True
    return ollama_configured(role) and _ollama_available(role=role)


def resolve_model_name(
    model: Optional[str] = None,
    use_ollama: Optional[bool] = None,
    role: LLMRole = "tool",
) -> str:
    if model:
        return model
    if use_ollama is None:
        use_ollama = using_ollama(role)
    if use_ollama:
        return ollama_model_name(role)
    if llm_choice_mode(role) == "bosch":
        return os.getenv("BOSCH_MODEL", DEFAULT_BOSCH_MODEL).strip()
    return os.getenv("NVIDIA_MODEL", DEFAULT_NVIDIA_MODEL) or DEFAULT_NVIDIA_MODEL


def llm_backend_label(role: LLMRole = "tool") -> str:
    source = llm_choice_mode(role)
    kind = "Ollama" if using_ollama(role) else ("Bosch" if source == "bosch" else "NVIDIA")
    return f"{kind} ({resolve_model_name(role=role)})"


def planner_backend_label() -> str:
    return llm_backend_label("tool")


def report_backend_label() -> str:
    return llm_backend_label("classify")


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def build_llm(
    model: Optional[str] = None,
    backend: Optional[str] = None,
    role: LLMRole = "tool",
) -> Any:
    if backend == "ollama":
        use_ollama = True
    elif backend == "nvidia":
        use_ollama = False
    else:
        use_ollama = using_ollama(role)
    name = resolve_model_name(model, use_ollama=use_ollama, role=role)
    timeout_name = "OLLAMA_TIMEOUT" if use_ollama else (
        "BOSCH_TIMEOUT" if llm_choice_mode(role) == "bosch" else "NVIDIA_TIMEOUT"
    )
    timeout = float(os.getenv(timeout_name, "300"))
    if use_ollama:
        from langchain_ollama import ChatOllama

        base_url = ollama_base_url(role=role)
        num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "16384"))
        _log(
            f"[llm] choosed={llm_choice_mode(role) or 'auto'} role={role} backend=ollama "
            f"model={name} base_url={base_url} timeout={timeout}s"
        )
        return ChatOllama(
            model=name,
            base_url=base_url,
            temperature=0.1,
            num_ctx=num_ctx,
            num_predict=2048,
            client_kwargs={"timeout": timeout},
        )
    if llm_choice_mode(role) == "bosch":
        from langchain_openai import ChatOpenAI

        api_key = os.getenv("MODEL_FARM_API_KEY")
        if not api_key:
            raise RuntimeError("Set MODEL_FARM_API_KEY for the Bosch model farm.")
        base_url = os.getenv("BOSCH_BASE_URL", DEFAULT_BOSCH_BASE_URL).rstrip("/")
        api_version = os.getenv("BOSCH_API_VERSION", DEFAULT_BOSCH_API_VERSION)
        _log(f"[llm] choosed=bosch role={role} model={name} timeout={timeout}s")
        return ChatOpenAI(
            model=name,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            temperature=0.1,
            max_tokens=2048,
            default_query={"api-version": api_version},
            default_headers={BOSCH_HEADER_NAME: api_key},
        )
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("Set NVIDIA_API_KEY or a reachable Ollama model in .env.")
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    _log(f"[llm] choosed={llm_choice_mode(role) or 'auto'} role={role} backend=nvidia model={name} timeout={timeout}s")
    extra: dict[str, Any] = {}
    max_tokens = 2048
    if "nemotron-3.5" in name.lower() or "lightning" in name.lower():
        extra["model_kwargs"] = {
            "chat_template_kwargs": {
                "enable_thinking": False,
                "force_nonempty_content": True,
            }
        }
        max_tokens = 4096
    return ChatNVIDIA(
        model=name,
        api_key=api_key,
        temperature=0.1,
        max_tokens=max_tokens,
        timeout=timeout,
        **extra,
    )


def _count_tool_messages(messages: list) -> int:
    return sum(1 for m in messages if isinstance(m, ToolMessage))


_SOURCE_OPENING_TOOLS = frozenset(
    {
        "get_micro_window",
        "run_simulation",
        "query_call_graph",
        "query_data_flow",
        "get_window",
        "inspect",
        "inspect_declaration",
    }
)


def _window_opened(messages: list) -> bool:
    return any(
        isinstance(m, ToolMessage) and getattr(m, "name", None) in _SOURCE_OPENING_TOOLS
        for m in messages
    )


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
    content = (content or "").strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines).strip()
    try:
        data = json.loads(content)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        start, end = content.find("{"), content.rfind("}")
        if start >= 0 and end > start:
            try:
                data = json.loads(content[start : end + 1])
                return data if isinstance(data, dict) else None
            except json.JSONDecodeError:
                pass
        return None


def _verdict_accepted(messages: list) -> Optional[dict]:
    for m in reversed(messages):
        if not isinstance(m, ToolMessage) or getattr(m, "name", None) != "submit_verdict":
            continue
        data = _parse_tool_json(str(m.content))
        if data and data.get("accepted") is True:
            v = data.get("verdict")
            return v if isinstance(v, dict) else data
    return None


def _simulation_was_run(messages: list) -> bool:
    return "run_simulation" in _tool_names_used(messages)


def _sync_session_from_messages(messages: list) -> None:
    from aoob_agent.session import get_session

    session = get_session()
    session.tokens_used = messages_token_total(messages)
    session.simulation_run = _simulation_was_run(messages)
    for m in reversed(messages):
        if not isinstance(m, AIMessage):
            continue
        for tc in getattr(m, "tool_calls", None) or []:
            args = tc.get("args") or {}
            conf = str(args.get("confidence_so_far") or "").strip().lower()
            if conf in {"low", "medium", "high"}:
                session.confidence_so_far = conf
                break
        if session.confidence_so_far != "low":
            break


def _should_force_review(messages: list, state: AgentState) -> tuple[bool, str]:
    tool_n = _count_tool_messages(messages)
    conf = str(state.get("confidence_so_far") or "low").lower()
    sim = bool(state.get("simulation_run")) or _simulation_was_run(messages)
    ceiling = SAFETY_CEILING
    try:
        from aoob_agent.session import get_session

        ceiling = effective_safety_ceiling(get_session())
    except RuntimeError:
        pass
    if tool_n >= ceiling and not (conf == "high" and sim):
        return True, (
            f"Safety ceiling ({ceiling} tool calls) reached without "
            f"high confidence and run_simulation evidence."
        )
    return False, ""


def _normalize_report_dict(data: dict) -> dict:
    out = dict(data)
    raw = str(out.get("classification") or "").strip().lower()
    if raw in {"false", "f", "fp", "false_positive"}:
        out["classification"] = "false"
    elif raw in {"true (low)", "true_low", "low", "potential"}:
        out["classification"] = "true (low)"
    elif raw in {"true", "t", "bug", "tp", "true_positive"}:
        out["classification"] = "true"
    elif raw in {"undecided", "u", "inter_procedural"}:
        out["classification"] = "undecided"
    else:
        out["classification"] = "review"
    conf = str(out.get("confidence") or "medium").strip().lower()
    out["confidence"] = conf if conf in {"low", "medium", "high"} else "medium"
    out.setdefault("comment", "")
    out.setdefault("astree_message", "")
    out.setdefault("summary", "")
    out.setdefault("human_tag_pattern", "")
    out.setdefault("reason_for_review", "")
    out.setdefault("schema_version", 2)
    out.setdefault("tokens_used", 0)
    out.setdefault("naive_full_file_tokens", 0)
    out.setdefault("token_savings_percent", 0.0)
    out.setdefault("micro_window_used", False)
    out.setdefault("safety_ceiling_hit", False)
    out.setdefault("derived_from", None)
    return out


def _degraded_report_payload(order_id: int, messages: list, reason: str) -> dict:
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
        "reason_for_review": reason,
        "tools_used": sorted(_tool_names_used(messages)),
    }


def _fill_report_from_case(data: dict, case: CaseFile, messages: list, store: DataStore) -> dict:
    from aoob_agent.session import get_session

    data["alarm_order_id"] = case.order_id
    data["tools_used"] = sorted(_tool_names_used(messages))
    data["alarm_type"] = case.alarm_type
    data["alarm_category"] = case.alarm_category
    data["location"] = case.location
    data["astree_message"] = case.astree_message or data.get("astree_message") or ""
    data.setdefault("function_name", case.alarm_function)
    try:
        session = get_session()
        data["tokens_used"] = session.tokens_used or messages_token_total(messages)
        data["naive_full_file_tokens"] = session.naive_full_file_tokens
        data["derived_from"] = session.derived_from
        data["safety_ceiling_hit"] = session.force_review
        data["micro_window_used"] = bool(session.micro_windows_opened)
        data["token_savings_percent"] = compute_token_savings_percent(
            int(data["tokens_used"]), int(data["naive_full_file_tokens"])
        )
    except RuntimeError:
        naive = naive_full_file_tokens(store.source_lines)
        used = messages_token_total(messages)
        data["tokens_used"] = used
        data["naive_full_file_tokens"] = naive
        data["token_savings_percent"] = compute_token_savings_percent(used, naive)
    if not data.get("manipulation_sequence"):
        data["manipulation_sequence"] = [
            {
                "order": step.step,
                "function": step.function,
                "access": step.access,
                "variable": step.symbol,
                "location": step.location,
                "note": step.note,
            }
            for step in case.path
        ]
    if not data.get("affected_symbols"):
        obj = case.indexed_object
        kind = obj.kind if obj.kind in {
            "variable", "array", "pointer", "struct", "field", "other", "unknown"
        } else "unknown"
        data["affected_symbols"] = [
            {"name": obj.name or case.operand_symbol, "kind": kind, "evidence": obj.location or case.location}
        ]
    be = data.get("bounds_evidence")
    if not isinstance(be, dict):
        be = {}
    obj = case.indexed_object
    be.setdefault("symbol", obj.name)
    if be.get("declared_size") in (None, ""):
        be["declared_size"] = obj.array_size
    be.setdefault("index_expression", case.index_expression)
    be.setdefault("index_inferred_range", "")
    be.setdefault("index_safe_range", "")
    if be.get("target_capacity") in (None, ""):
        be["target_capacity"] = obj.array_size
    if not isinstance(be.get("guards_found"), list) or not be.get("guards_found"):
        be["guards_found"] = list(case.guards)
    if not be.get("index_origin_summary"):
        notes = [s.note for s in case.path if s.note]
        be["index_origin_summary"] = " | ".join(notes[:4])
    if not be.get("evidence_completeness"):
        opened = _window_opened(messages)
        if not opened:
            be["evidence_completeness"] = "not traced"
        else:
            be["evidence_completeness"] = "fully traced" if not case.gaps else "partially traced"
    data["bounds_evidence"] = be
    return data


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


def _human_brief(order_id: int, case: CaseFile) -> str:
    helpers = ", ".join(case.helpers) if case.helpers else "(none)"
    gap_note = ""
    if case.gaps:
        gap_note = (
            f"\nGaps ({', '.join(case.gaps)}): you MUST call query_call_graph() and/or "
            "inspect() on relevant helpers before a non-review verdict.\n"
        )
    return (
        f"Investigate Order {order_id}.\n\n{case.brief()}\n\n"
        f"Helpers: {helpers}\n{gap_note}\n"
        "Evidence-driven workflow: start with get_micro_window(window_size=15), "
        "then run_simulation() for raw bounds facts. Pass confidence_so_far on every tool call. "
        "Submit when confidence_so_far=high (or review with reason_for_review if stuck)."
    )


def build_agent(store: DataStore, model: Optional[str] = None, case: Optional[CaseFile] = None):
    bind_store(store)
    llm = build_llm(model=model, role="tool")
    current_backend = llm_choice_mode("tool") or ("ollama" if using_ollama("tool") else "nvidia")

    def _try_runtime_fallback(exc: Exception) -> bool:
        nonlocal llm, current_backend
        if current_backend != "ollama":
            return False
        if not ollama_runtime_fallback_enabled() or not nvidia_configured():
            return False
        if not _is_ollama_runtime_error(exc):
            return False
        _log(f"[llm] Ollama error: {exc}")
        _log("[llm] AOOB_OLLAMA_RUNTIME_FALLBACK=1 → switching to NVIDIA for this investigation")
        llm = build_llm(model=model, backend="nvidia")
        current_backend = "nvidia"
        return True

    def _bind(*, force_review: bool = False):
        if force_review:
            return llm.bind_tools(TOOLS, tool_choice="submit_verdict")
        return llm.bind_tools(TOOLS)

    def agent_node(state: AgentState) -> dict:
        msgs = list(state["messages"])
        _sync_session_from_messages(msgs)
        tool_n = _count_tool_messages(msgs)
        force, reason = _should_force_review(msgs, state)
        from aoob_agent.session import get_session

        session = get_session()
        if force:
            session.force_review = True
            session.force_review_reason = reason
            msgs = msgs + [
                HumanMessage(
                    content=(
                        "SAFETY CEILING: You must call submit_verdict NOW with "
                        "classification='review', confidence='low', and a mandatory "
                        f"reason_for_review explaining: {reason}"
                    )
                )
            ]
        elif session.counterargument_pending:
            msgs = msgs + [
                HumanMessage(
                    content=(
                        "SELF-VERIFICATION: Your last submit_verdict was rejected because "
                        "confidence=medium. Resubmit with counterargument=<strongest case "
                        "for the opposite classification> before finalizing."
                    )
                )
            ]
        runner = _bind(force_review=force)
        _log(
            f"[agent] invoke msgs={len(msgs)} tools_so_far={tool_n} "
            f"backend={current_backend} force_review={force} "
            f"confidence={state.get('confidence_so_far')}"
        )
        try:
            response = runner.invoke(msgs)
        except Exception as exc:  # noqa: BLE001
            if force:
                _log(f"[agent] forced review bind failed ({exc}); retrying unbound")
                runner = _bind(force_review=False)
                try:
                    response = runner.invoke(msgs)
                except Exception as inner:  # noqa: BLE001
                    if not _try_runtime_fallback(inner):
                        raise
                    runner = _bind(force_review=False)
                    response = runner.invoke(msgs)
            elif not _try_runtime_fallback(exc):
                raise
            else:
                runner = _bind(force_review=False)
                response = runner.invoke(msgs)
        tool_calls = getattr(response, "tool_calls", None) or []
        _log(f"[agent] tool_calls={len(tool_calls)}")
        conf = str(state.get("confidence_so_far") or "low")
        for tc in tool_calls:
            args = tc.get("args") or {}
            c = str(args.get("confidence_so_far") or "").strip().lower()
            if c in {"low", "medium", "high"}:
                conf = c
        extra = msgs[len(state["messages"]) :]
        return {
            "messages": extra + [response],
            "confidence_so_far": conf,
            "simulation_run": _simulation_was_run(extra + [response] + msgs),
            "force_review": force,
        }

    def tools_node(state: AgentState) -> dict:
        result = ToolNode(TOOLS).invoke(state)
        rounds = int(state.get("tool_rounds") or 0) + 1
        merged = list(state.get("messages") or []) + list(result.get("messages") or [])
        _sync_session_from_messages(merged)
        from aoob_agent.session import get_session

        session = get_session()
        conf = session.confidence_so_far
        for m in reversed(merged):
            if not isinstance(m, ToolMessage) or m.name != "submit_verdict":
                continue
            data = _parse_tool_json(str(m.content))
            if data and data.get("accepted") is False:
                err = str(data.get("error") or "")
                if "counterargument" in err.lower():
                    session.counterargument_pending = True
                break
            if data and data.get("accepted") is True:
                session.counterargument_pending = False
        force, reason = _should_force_review(merged, {
            **state,
            "confidence_so_far": conf,
            "simulation_run": session.simulation_run,
        })
        if force:
            session.force_review = True
            session.force_review_reason = reason
        _log(f"[tools] round={rounds} confidence={conf} force_review={force}")
        return {
            **result,
            "tool_rounds": rounds,
            "confidence_so_far": conf,
            "simulation_run": session.simulation_run,
            "force_review": force,
            "counterargument_pending": session.counterargument_pending,
        }

    def after_agent(state: AgentState) -> Literal["tools", "agent", "report"]:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        if _verdict_accepted(state["messages"]):
            return "report"
        if state.get("force_review"):
            return "agent"
        return "agent"

    def after_tools(state: AgentState) -> Literal["agent", "report"]:
        if _verdict_accepted(state["messages"]):
            return "report"
        return "agent"

    def report_node(state: AgentState) -> dict:
        active = case or compile_case(store, state["order_id"])
        accepted = _verdict_accepted(state["messages"])
        if accepted:
            data = {
                "classification": accepted.get("classification") or "review",
                "comment": accepted.get("comment") or "",
                "confidence": accepted.get("confidence") or "medium",
                "summary": accepted.get("summary") or accepted.get("comment") or "",
                "human_tag_pattern": accepted.get("human_tag_pattern") or "",
                "reason_for_review": accepted.get("reason_for_review") or "",
                "token_savings_percent": accepted.get("token_savings_percent", 0.0),
                "tokens_used": accepted.get("tokens_used", 0),
                "naive_full_file_tokens": accepted.get("naive_full_file_tokens", 0),
                "micro_window_used": accepted.get("micro_window_used", False),
                "safety_ceiling_hit": accepted.get("safety_ceiling_hit", False),
            }
            data = _fill_report_from_case(
                _normalize_report_dict(data), active, state["messages"], store
            )
        else:
            force_reason = "Agent did not submit an accepted verdict."
            try:
                from aoob_agent.session import get_session
                sess = get_session()
                if sess.force_review_reason:
                    force_reason = sess.force_review_reason
            except RuntimeError:
                pass
            data = {
                "classification": "review",
                "comment": (
                    "The agent failed to converge: it did not call "
                    "submit_verdict with an accepted classification."
                ),
                "confidence": "low",
                "summary": "Agent failed to converge.",
                "human_tag_pattern": "",
                "reason_for_review": force_reason,
                "token_savings_percent": 0.0,
                "micro_window_used": False,
                "safety_ceiling_hit": bool(state.get("force_review")),
            }
            digest_parts = [f"Case:\n{active.brief()}"]
            for message in state["messages"]:
                if not isinstance(message, ToolMessage):
                    continue
                if getattr(message, "name", None) not in {"run_simulation", "submit_verdict"}:
                    continue
                digest_parts.append(
                    f"{message.name}:\n{str(message.content)}"
                )
            classifier = build_llm(role="classify")
            classified = classifier.invoke(
                [
                    SystemMessage(
                        content=(
                            "You are the final AOOB alarm classifier. Return only a valid JSON object. "
                            "Base the classification exclusively on the supplied investigation evidence."
                        )
                    ),
                    HumanMessage(
                        content=(
                            "Classify this Astrée AOOB investigation. Return ONLY JSON with keys "
                            "classification (false, true, true (low), undecided, or review), comment, "
                            "confidence (low/medium/high), summary, human_tag_pattern, reason_for_review. "
                            "Do not invent evidence and use review when insufficient.\n\n"
                            + "\n\n".join(digest_parts)
                        )
                    ),
                ]
            )
            classification_data = _parse_tool_json(_message_text(classified))
            if classification_data is None:
                raise ValueError("Classification model returned invalid JSON.")
            data = classification_data
            data = _fill_report_from_case(
                _normalize_report_dict(data), active, state["messages"], store
            )

        try:
            report = AlarmInvestigationReport.model_validate(data)
        except Exception as exc:  # noqa: BLE001
            _log(f"[report] validation failed ({exc})")
            report = AlarmInvestigationReport.model_validate(
                _fill_report_from_case(
                    _normalize_report_dict(
                        _degraded_report_payload(state["order_id"], state["messages"], "validation")
                    ),
                    active,
                    state["messages"],
                    store,
                )
            )
        fp = active.order_id
        try:
            from aoob_agent.session import case_fingerprint, get_session, register_verdict

            register_verdict(case_fingerprint(active), fp, report.model_dump())
        except Exception:  # noqa: BLE001
            pass
        return {"report": report.model_dump()}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_node("report", report_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent", after_agent, {"tools": "tools", "agent": "agent", "report": "report"}
    )
    graph.add_conditional_edges("tools", after_tools, {"agent": "agent", "report": "report"})
    graph.add_edge("report", END)
    return graph.compile()


def _propagate_sibling_verdict(case: CaseFile, human_tag_pattern: str = "") -> Optional[dict]:
    """Reuse verdict from structurally identical case unless human tag pattern differs."""
    from aoob_agent.session import case_fingerprint, lookup_verdict

    fp = case_fingerprint(case)
    hit = lookup_verdict(fp)
    if hit is None:
        return None
    src_id, verdict = hit
    if src_id == case.order_id:
        return None
    if str(verdict.get("classification") or "").lower() == "false":
        return None
    propagated = dict(verdict)
    propagated["alarm_order_id"] = case.order_id
    propagated["derived_from"] = src_id
    propagated["comment"] = (
        f"{propagated.get('comment', '')} [Propagated from Order {src_id}; "
        "re-open if human tag pattern differs.]"
    ).strip()
    if human_tag_pattern and propagated.get("human_tag_pattern") != human_tag_pattern:
        return None
    return propagated


def investigate_alarm(
    order_id: int, store: DataStore, model: Optional[str] = None
) -> AlarmInvestigationReport:
    report: Optional[AlarmInvestigationReport] = None
    for event in stream_investigate(order_id, store, model=model):
        if event.get("type") == "report":
            report = AlarmInvestigationReport.model_validate(event["data"])
        elif event.get("type") == "error":
            raise RuntimeError(event.get("message") or "Investigation failed")
    if report is None:
        raise ValueError("Investigation finished without a structured report.")
    return report


def stream_investigate(order_id: int, store: DataStore, model: Optional[str] = None):
    yield {"type": "status", "message": f"Starting investigation for Order {order_id}…"}
    try:
        clear_session()
        case = compile_case(store, order_id)
        naive_tokens = naive_full_file_tokens(store.source_lines)
        begin_session(case, naive_full_file_tokens=naive_tokens)
        bind_store(store)

        prefilter_note = ""
        if prefilter_enabled():
            try:
                from aoob_agent.pre_filter import prefilter_alarm

                pf = prefilter_alarm(order_id, store, model=model, case=case, manage_session=False)
                route = str(pf.get("route") or "uncertain")
                from aoob_agent.session import get_session

                sess = get_session()
                sess.prefilter_route = route
                if route == "likely_false":
                    sess.lite_investigation = True
                prefilter_note = (
                    f"\nPre-filter route: {route} — {pf.get('rationale', '')}\n"
                    "Still submit your own verdict after gathering evidence; "
                    "likely_false cases use a shorter safety ceiling but are not auto-dismissed.\n"
                )
                yield {
                    "type": "status",
                    "message": f"Pre-filter route: {route}",
                }
            except Exception as pf_exc:  # noqa: BLE001
                _log(f"[prefilter] skipped: {pf_exc}")

        sibling = _propagate_sibling_verdict(case)
        if sibling is not None:
            sibling = _fill_report_from_case(
                _normalize_report_dict(sibling), case, [], store
            )
            sibling["derived_from"] = sibling.get("derived_from")
            yield {
                "type": "status",
                "message": f"Reusing verdict from Order {sibling['derived_from']} (identical case file).",
            }
            yield {"type": "report", "data": sibling}
            yield {"type": "done", "order_id": order_id}
            return

        yield {
            "type": "path_meta",
            "steps": case.path_records(),
            "indexed_object": case.indexed_object.name,
            "declared_size": case.array_size,
            "datatype": case.indexed_object.declared_type,
            "operand": case.operand_symbol,
            "index_expression": case.index_expression,
            "alarm_function": case.alarm_function,
        }
        yield {
            "type": "status",
            "message": (
                f"Case compiled · operand {case.operand_symbol} · "
                f"object {case.indexed_object.name} size={case.array_size} · "
                f"{len(case.path)} path step(s)"
            ),
        }

        agent = build_agent(store, model=model, case=case)
        human_content = _human_brief(order_id, case) + prefilter_note

        yield {
            "type": "agent_input",
            "data": {
                "llm": llm_backend_label("tool"),
                "llm_tool": llm_backend_label("tool"),
                "llm_classify": llm_backend_label("classify"),
                "system_prompt": SYSTEM_PROMPT,
                "human_message": human_content,
                "case_brief": case.brief(),
                "helpers": list(case.helpers),
                "operand": case.operand_symbol,
                "indexed_object": case.indexed_object.name,
                "declared_size": case.array_size,
                "index_expression": case.index_expression,
                "guards": list(case.guards),
                "gaps": list(case.gaps),
                "path": case.path_records(),
                "alarm_function": case.alarm_function,
                "location": case.location,
                "astree_message": case.astree_message,
            },
        }
        initial: AgentState = {
            "order_id": order_id,
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=human_content),
            ],
            "report": None,
            "tool_rounds": 0,
            "confidence_so_far": "low",
            "simulation_run": False,
            "force_review": False,
            "counterargument_pending": False,
        }
        yield {
            "type": "status",
            "message": (
                f"Calling tool/reasoning {llm_backend_label('tool')}; "
                f"final classification uses {llm_backend_label('classify')}…"
            ),
        }
        final_report = None
        recursion_limit = max(64, 2 * (SAFETY_CEILING + 4) + 8)
        for update in agent.stream(
            initial, config={"recursion_limit": recursion_limit}, stream_mode="updates"
        ):
            if not isinstance(update, dict):
                continue
            for node_name, payload in update.items():
                if node_name == "agent":
                    for msg in payload.get("messages") or []:
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
                        if tool_calls:
                            yield {
                                "type": "status",
                                "message": f"Model requested {len(tool_calls)} tool call(s).",
                            }
                        elif text:
                            yield {"type": "agent", "content": text}
                elif node_name == "tools":
                    for msg in payload.get("messages") or []:
                        if not isinstance(msg, ToolMessage):
                            continue
                        preview_t = str(msg.content)
                        tool_name = getattr(msg, "name", None) or "tool"
                        cap = 90_000 if tool_name in {"get_window", "move_window", "inspect"} else 1200
                        if len(preview_t) > cap:
                            preview_t = preview_t[:cap] + "\n…[truncated]"
                        yield {
                            "type": "tool_result",
                            "name": getattr(msg, "name", None) or "tool",
                            "preview": preview_t,
                        }
                elif node_name == "report":
                    data = payload.get("report")
                    if data:
                        data.setdefault("alarm_order_id", order_id)
                        final_report = data
                        yield {"type": "report", "data": data}
        if final_report is None:
            yield {"type": "error", "message": "Investigation finished without a structured report."}
        else:
            yield {"type": "done", "order_id": order_id}
    except Exception as exc:  # noqa: BLE001
        _log(f"[agent] stream error: {exc}")
        text = str(exc)
        if "10061" in text or "connection refused" in text.lower():
            text = (
                f"Ollama is not running at {ollama_base_url()}. "
                "Start it with `ollama serve`, then retry. "
                f"Expected model: {ollama_model_name() or 'OLLAMA_LOCAL_MODEL'}."
            )
        yield {"type": "error", "message": text}
    finally:
        clear_session()
