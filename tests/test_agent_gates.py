#!/usr/bin/env python3
"""Unit checks for agent helper gates under the 5-tool set."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_core.messages import ToolMessage

from aoob_agent.agent import (  # noqa: E402
    _has_useful_index_origin_tool,
    _needs_call_site_followup,
    _origin_progress_stalled,
    pending_named_hop,
    resolve_model_name,
    select_visible_tool_names,
    using_ollama,
)


def _tool(name: str, payload: dict) -> ToolMessage:
    return ToolMessage(content=json.dumps(payload), tool_call_id="t1", name=name)


def test_has_useful_index_origin_tool_true_with_write() -> None:
    msgs = [
        _tool(
            "get_function_snippet",
            {
                "index_operands_at_alarm": [
                    {"indexed_object": "arr", "operand_text": "idx"}
                ]
            },
        ),
        _tool(
            "get_trimmed_sequence",
            {
                "variable": "idx",
                "origin_resolved": True,
                "sequence": [{"access": "write", "is_alarm_site": True, "line": 20}],
            },
        )
    ]
    assert _has_useful_index_origin_tool(msgs) is True


def test_needs_callsite_followup_when_no_writes() -> None:
    msgs = [
        _tool(
            "get_variable_scope",
            {"symbol": "idx", "scope": "parameter"},
        ),
        _tool(
            "get_trimmed_sequence",
            {
                "variable": "idx",
                "origin_resolved": False,
                "sequence": [{"access": "read", "is_alarm_site": True, "line": 20}],
            },
        )
    ]
    assert _needs_call_site_followup(msgs) is True


def test_followup_clears_after_caller_context() -> None:
    msgs = [
        _tool(
            "get_variable_scope",
            {"symbol": "idx", "scope": "parameter"},
        ),
        _tool(
            "get_trimmed_sequence",
            {
                "variable": "idx",
                "origin_resolved": False,
                "sequence": [{"access": "read", "is_alarm_site": True}],
            },
        ),
        _tool(
            "get_caller_context",
            {"current_function": "Foo", "callers": [{"function": "Bar"}], "unresolved": False},
        ),
    ]
    assert _needs_call_site_followup(msgs) is False


def test_pending_named_hop_from_caller_context() -> None:
    msgs = [
        _tool(
            "get_caller_context",
            {"current_function": "Foo", "callers": [{"function": "Bar"}], "unresolved": False},
        )
    ]
    assert pending_named_hop(msgs) == "Bar"


def test_pending_named_hop_clears_after_target_sequence() -> None:
    msgs = [
        _tool(
            "get_caller_context",
            {"current_function": "Foo", "callers": [{"function": "Bar"}], "unresolved": False},
        ),
        _tool(
            "get_trimmed_sequence",
            {"variable": "idx", "origin_resolved": False, "sequence": [{"access": "read", "is_alarm_site": True}]},
        ),
    ]
    assert pending_named_hop(msgs) is None


def test_no_callsite_followup_when_origin_resolved_true() -> None:
    msgs = [
        _tool("get_variable_scope", {"symbol": "idx", "scope": "parameter"}),
        _tool(
            "get_trimmed_sequence",
            {
                "variable": "idx",
                "origin_resolved": True,
                "sequence": [{"access": "write", "is_alarm_site": True}],
            },
        ),
    ]
    assert _needs_call_site_followup(msgs) is False


def test_origin_progress_stalled_on_identical_trimmed_sequence() -> None:
    payload = {
        "variable": "idx",
        "origin_resolved": False,
        "total_steps": 1,
        "sequence": [{"access": "read", "is_alarm_site": True, "function": "Foo"}],
        "scope_mode": "function",
        "trace_function": "Foo",
    }
    msgs = [
        _tool("get_trimmed_sequence", payload),
        _tool("get_trimmed_sequence", payload),
    ]
    assert _origin_progress_stalled(msgs) is True


def test_origin_progress_stalled_on_repeated_seq_caller_cycle() -> None:
    seq = {
        "variable": "idx",
        "origin_resolved": False,
        "total_steps": 1,
        "sequence": [{"access": "read", "is_alarm_site": True, "function": "Foo"}],
    }
    caller = {
        "current_function": "Foo",
        "callers": [{"function": "Bar", "line": 10}],
        "unresolved": False,
    }
    msgs = [
        _tool("get_trimmed_sequence", seq),
        _tool("get_caller_context", caller),
        _tool("get_trimmed_sequence", seq),
        _tool("get_caller_context", caller),
    ]
    assert _origin_progress_stalled(msgs) is True


def test_origin_progress_not_stalled_when_results_change() -> None:
    msgs = [
        _tool(
            "get_trimmed_sequence",
            {
                "variable": "idx",
                "origin_resolved": False,
                "total_steps": 1,
                "sequence": [{"access": "read", "is_alarm_site": True, "function": "Foo"}],
            },
        ),
        _tool(
            "get_trimmed_sequence",
            {
                "variable": "idx",
                "origin_resolved": False,
                "total_steps": 2,
                "sequence": [
                    {"access": "read", "is_alarm_site": True, "function": "Foo"},
                    {"access": "write", "is_alarm_site": False, "function": "Foo"},
                ],
            },
        ),
    ]
    assert _origin_progress_stalled(msgs) is False


def test_visible_tools_capped_and_seeded() -> None:
    os.environ["AOOB_MAX_VISIBLE_TOOLS"] = "3"
    try:
        names = select_visible_tool_names([])
        assert len(names) == 3
        assert "get_trimmed_sequence" in names
    finally:
        os.environ.pop("AOOB_MAX_VISIBLE_TOOLS", None)


def test_ollama_network_env_selected_when_available() -> None:
    keys = [
        "AOOB_LLM_BACKEND",
        "AOOB_LLM_CHOOSED",
        "NVIDIA_API_KEY",
        "NVIDIA_MODEL",
        "OLLAMA_LOCAL_MODEL",
        "OLLAMA_LOCAL_BASE_URL",
        "OLLAMA_MODEL",
        "OLLAMA_NETWORK_MODEL",
        "OLLAMA_NETWORK_BASE_URL",
    ]
    old = {key: os.environ.get(key) for key in keys}
    try:
        os.environ.pop("AOOB_LLM_BACKEND", None)
        os.environ["AOOB_LLM_CHOOSED"] = "network"
        os.environ["NVIDIA_API_KEY"] = "test-key"
        os.environ["NVIDIA_MODEL"] = "nvidia/test"
        os.environ.pop("OLLAMA_LOCAL_MODEL", None)
        os.environ.pop("OLLAMA_LOCAL_BASE_URL", None)
        os.environ.pop("OLLAMA_MODEL", None)
        os.environ["OLLAMA_NETWORK_MODEL"] = "gemma4:31b"
        os.environ["OLLAMA_NETWORK_BASE_URL"] = "http://127.0.0.1:11434"
        with patch("aoob_agent.agent._ollama_available", return_value=True):
            assert using_ollama() is True
            assert resolve_model_name() == "gemma4:31b"
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def main() -> int:
    tests = [
        test_has_useful_index_origin_tool_true_with_write,
        test_needs_callsite_followup_when_no_writes,
        test_followup_clears_after_caller_context,
        test_pending_named_hop_from_caller_context,
        test_pending_named_hop_clears_after_target_sequence,
        test_no_callsite_followup_when_origin_resolved_true,
        test_origin_progress_stalled_on_identical_trimmed_sequence,
        test_origin_progress_stalled_on_repeated_seq_caller_cycle,
        test_origin_progress_not_stalled_when_results_change,
        test_visible_tools_capped_and_seeded,
        test_ollama_network_env_selected_when_available,
    ]
    for fn in tests:
        fn()
        print(f"ok {fn.__name__}")
    print("all agent-gate checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
