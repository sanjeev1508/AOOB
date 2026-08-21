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
from aoob_agent.session import begin_session, clear_session
from aoob_agent.tools import TOOLS, bind_store

DEFAULT_NVIDIA_MODEL = "meta/llama-3.1-70b-instruct"
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"

SYSTEM_PROMPT = """You are an Astrée out-of-bounds (AOOB) investigator.

Python already compiled the case file, including per-step path metadata
(scope, sequence, access, declared size, datatype, line). Do not invent
declarations, callers, writes, or guards.

Tools:
- get_window() — complete current function body. Call this to read source.
- move_window(direction="next"|"prev", step=0) — navigate the path only
  (no source). Call get_window afterwards if you need that step's code.
- inspect(function_name) — complete helper function body, only helpers listed
  on the case file.
- submit_verdict(classification, comment, confidence) — true | false | review.
  This is the terminal call. You MUST call it to finish.
  If declared size is unknown, classification must be review, not true/false.

Rules:
- Use Python-extracted guards and origin inits. Nearby if() is not a bound unless listed.
- Astrée [lo, hi] is abstract. Forbidden: array_size < Astrée hi ⇒ true.
- if/for guards or a listed constant origin init that keeps the index inside capacity ⇒ false.
- Index can exceed capacity and no extracted guard/init ⇒ true (medium unless origin is traced).
- Missing size/origin/allocation ⇒ review.
"""

REPORT_PROMPT = """JSON report only, no markdown.
Copy Python guards into bounds_evidence.guards_found. Do not invent guards.
size_unknown / object_undeclared ⇒ classification review.
tools_used = tools actually called.
"""


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    order_id: int
    report: Optional[dict]
    tool_rounds: int
    idle_turns: int
    verdict_nudge_sent: int


def load_env(project_root: Path) -> None:
    load_dotenv(project_root / ".env", override=False)


def llm_choice_mode() -> Optional[str]:
    raw = (os.getenv("AOOB_LLM_CHOOSED") or os.getenv("AOOB_LLM_CHOOSE") or "").strip().lower()
    return raw if raw in {"local", "network", "nvidia"} else None


def _local_ollama_model_name() -> str:
    return (
        os.getenv("OLLAMA_LOCAL_MODEL")
        or os.getenv("OLLAMA_LOCAL")
        or os.getenv("OLLAMA_LOCAL_PATH")
        or ""
    ).strip()


def _network_ollama_model_name() -> str:
    return (os.getenv("OLLAMA_NETWORK_MODEL") or "").strip()


def ollama_base_url(mode: Optional[str] = None) -> str:
    selected = mode or llm_choice_mode()
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


def ollama_model_name() -> str:
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


def ollama_configured() -> bool:
    return bool(ollama_model_name())


def nvidia_configured() -> bool:
    return bool((os.getenv("NVIDIA_API_KEY") or "").strip())


def _ollama_available(timeout: Optional[float] = None) -> bool:
    if not ollama_configured():
        return False
    base = ollama_base_url()
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


def using_ollama() -> bool:
    """Honor AOOB_LLM_CHOOSED. Do not silently pick NVIDIA because a health ping failed."""
    override = llm_backend_override()
    selected = llm_choice_mode()
    if override == "nvidia" or selected == "nvidia":
        return False
    if selected in {"local", "network"} or override == "ollama":
        if not ollama_configured():
            raise RuntimeError(
                f"AOOB_LLM_CHOOSED={selected or 'ollama'} but no Ollama model is set. "
                "Set OLLAMA_LOCAL_MODEL or OLLAMA_NETWORK_MODEL."
            )
        return True
    return ollama_configured() and _ollama_available()


def resolve_model_name(
    model: Optional[str] = None,
    use_ollama: Optional[bool] = None,
    role: Literal["planner", "report"] = "report",
) -> str:
    del role
    if model:
        return model
    if use_ollama is None:
        use_ollama = using_ollama()
    if use_ollama:
        return ollama_model_name()
    return os.getenv("NVIDIA_MODEL", DEFAULT_NVIDIA_MODEL) or DEFAULT_NVIDIA_MODEL


def llm_backend_label(role: Literal["planner", "report"] = "report") -> str:
    kind = "Ollama" if using_ollama() else "NVIDIA"
    return f"{kind} ({resolve_model_name(role=role)})"


def planner_backend_label() -> str:
    return llm_backend_label()


def report_backend_label() -> str:
    return llm_backend_label()


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


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
    timeout = float(os.getenv("OLLAMA_TIMEOUT" if use_ollama else "NVIDIA_TIMEOUT", "300"))
    if use_ollama:
        from langchain_ollama import ChatOllama

        base_url = ollama_base_url()
        num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "16384"))
        _log(
            f"[llm] choosed={llm_choice_mode() or 'auto'} backend=ollama "
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
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("Set NVIDIA_API_KEY or a reachable Ollama model in .env.")
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    _log(f"[llm] choosed={llm_choice_mode() or 'auto'} backend=nvidia model={name} timeout={timeout}s")
    return ChatNVIDIA(
        model=name, api_key=api_key, temperature=0.1, max_tokens=2048, timeout=timeout
    )


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


def _verdict_reject_count(messages: list) -> int:
    n = 0
    for m in messages:
        if not isinstance(m, ToolMessage) or getattr(m, "name", None) != "submit_verdict":
            continue
        data = _parse_tool_json(str(m.content))
        if data and data.get("accepted") is False:
            n += 1
    return n


def _max_tool_budget(path_len: int) -> int:
    return max(4, path_len + 2)


def _normalize_report_dict(data: dict) -> dict:
    out = dict(data)
    raw = str(out.get("classification") or "").strip().lower()
    if raw in {"true", "t", "bug", "tp", "true_positive"}:
        out["classification"] = "true"
    elif raw in {"false", "f", "fp", "false_positive"}:
        out["classification"] = "false"
    else:
        out["classification"] = "review"
    conf = str(out.get("confidence") or "medium").strip().lower()
    out["confidence"] = conf if conf in {"low", "medium", "high"} else "medium"
    out.setdefault("comment", "")
    out.setdefault("astree_message", "")
    out.setdefault("summary", "")
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
        "tools_used": sorted(_tool_names_used(messages)),
    }


def _fill_report_from_case(data: dict, case: CaseFile, messages: list) -> dict:
    data["alarm_order_id"] = case.order_id
    data["tools_used"] = sorted(_tool_names_used(messages))
    data["alarm_type"] = case.alarm_type
    data["alarm_category"] = case.alarm_category
    data["location"] = case.location
    data["astree_message"] = case.astree_message or data.get("astree_message") or ""
    data.setdefault("function_name", case.alarm_function)
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
    opened = _window_opened(messages)
    if case.array_size is None and data.get("classification") in {"true", "false"}:
        data["classification"] = "review"
        data["confidence"] = "low"
        be["evidence_completeness"] = "not traced"
    if not opened:
        be["evidence_completeness"] = "not traced"
        if data.get("classification") == "false":
            data["classification"] = "review"
        if data.get("confidence") == "high":
            data["confidence"] = "medium"
    elif not be.get("evidence_completeness"):
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


def build_agent(store: DataStore, model: Optional[str] = None, case: Optional[CaseFile] = None):
    bind_store(store)
    llm = build_llm(model=model)
    current_backend = "ollama" if using_ollama() else "nvidia"
    path_len = len(case.path) if case is not None else 1

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

    def _bind():
        return llm.bind_tools(TOOLS)

    def agent_node(state: AgentState) -> dict:
        msgs = list(state["messages"])
        tool_n = _count_tool_messages(msgs)
        idle = int(state.get("idle_turns") or 0)
        nudge_sent = int(state.get("verdict_nudge_sent") or 0)
        extra: list = []
        if idle >= 2 and nudge_sent == 0 and not _verdict_accepted(msgs):
            extra = [
                HumanMessage(
                    content=(
                        "call submit_verdict now with your best classification "
                        "given the evidence so far"
                    )
                )
            ]
            msgs = msgs + extra
            nudge_sent = 1
        runner = _bind()
        _log(f"[agent] invoke msgs={len(msgs)} tools_so_far={tool_n} backend={current_backend}")
        try:
            response = runner.invoke(msgs)
        except Exception as exc:  # noqa: BLE001
            if not _try_runtime_fallback(exc):
                raise
            runner = _bind()
            response = runner.invoke(msgs)
        tool_calls = getattr(response, "tool_calls", None) or []
        _log(f"[agent] tool_calls={len(tool_calls)}")
        next_idle = 0 if tool_calls else idle + 1
        return {
            "messages": extra + [response],
            "idle_turns": next_idle,
            "verdict_nudge_sent": nudge_sent,
        }

    def tools_node(state: AgentState) -> dict:
        result = ToolNode(TOOLS).invoke(state)
        rounds = int(state.get("tool_rounds") or 0) + 1
        _log(f"[tools] round={rounds}")
        return {**result, "tool_rounds": rounds, "idle_turns": 0}

    def after_agent(state: AgentState) -> Literal["tools", "agent", "report"]:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        idle = int(state.get("idle_turns") or 0)
        nudge_sent = int(state.get("verdict_nudge_sent") or 0)
        if _verdict_accepted(state["messages"]):
            return "report"
        # Two idle turns → one submit_verdict nudge; still idle → stop.
        if idle >= 2 and nudge_sent == 0:
            return "agent"
        if idle >= 3 and nudge_sent >= 1:
            return "report"
        if idle < 2:
            return "agent"
        return "report"

    def after_tools(state: AgentState) -> Literal["agent", "report"]:
        if _verdict_accepted(state["messages"]):
            return "report"
        if _verdict_reject_count(state["messages"]) >= 2:
            return "report"
        if _count_tool_messages(state["messages"]) >= _max_tool_budget(path_len):
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
            }
        else:
            data = {
                "classification": "review",
                "comment": (
                    "The agent failed to converge: it did not call "
                    "submit_verdict with an accepted classification."
                ),
                "confidence": "low",
                "summary": "Agent failed to converge.",
            }
        data = _fill_report_from_case(_normalize_report_dict(data), active, state["messages"])
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
                )
            )
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
        begin_session(case)
        bind_store(store)
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
        helpers = ", ".join(case.helpers) if case.helpers else "(none)"
        human_content = (
            f"Investigate Order {order_id}.\n\n{case.brief()}\n\n"
            f"Helpers: {helpers}\n\n"
            "Call get_window to read a function, inspect listed helpers if needed, "
            "then submit_verdict. Do not skip submit_verdict."
        )
        yield {
            "type": "agent_input",
            "data": {
                "llm": llm_backend_label(),
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
        initial = {
            "order_id": order_id,
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=human_content),
            ],
            "report": None,
            "tool_rounds": 0,
            "idle_turns": 0,
            "verdict_nudge_sent": 0,
        }
        yield {"type": "status", "message": f"Calling {llm_backend_label()}…"}
        final_report = None
        for update in agent.stream(initial, config={"recursion_limit": 24}, stream_mode="updates"):
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
                        # get_window/move_window/inspect open one whole
                        # function (data_store caps a function span at 1200
                        # lines) — the browser preview should size to that,
                        # not a flat cut that can land mid-snippet. This is
                        # a display-only cap; the LLM's own message history
                        # keeps the untruncated ToolMessage either way.
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
