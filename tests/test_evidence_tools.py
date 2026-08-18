#!/usr/bin/env python3
"""Acceptance checks for the 5-tool retrieval set."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aoob_agent.data_store import DataStore
from aoob_agent.tools import (  # noqa: E402
    TOOLS,
    _extract_index_operands,
    bind_store,
    get_caller_context,
    get_declaration_info,
    get_function_snippet,
    get_trimmed_sequence,
    get_variable_scope,
)


def _j(tool_call: str) -> dict:
    return json.loads(tool_call)


def test_tools_exactly_five() -> None:
    names = [t.name for t in TOOLS]
    assert names == [
        "get_variable_scope",
        "get_declaration_info",
        "get_trimmed_sequence",
        "get_function_snippet",
        "get_caller_context",
    ]


def test_variable_scope_local_global_unknown() -> None:
    local = _j(get_variable_scope.invoke({"symbol": "checkword", "alarm_order_id": 39}))
    assert local["scope"] in {"local", "parameter"}

    global_sym = _j(get_variable_scope.invoke({"symbol": "BswM_Cfg_LinSmCurrentStateInfo_ast", "alarm_order_id": 2475}))
    assert global_sym["scope"] in {"global", "local", "parameter"}

    unknown = _j(get_variable_scope.invoke({"symbol": "__no_such_symbol_zz__", "alarm_order_id": 39}))
    assert unknown["scope"] == "unknown"
    assert unknown.get("parse_note")


def test_declaration_info_shapes() -> None:
    arr = _j(get_declaration_info.invoke({"symbol": "MOCMEM_VBLSTRUCTEARLYCLEAREDRAM_MO_INST_LOOKUP"}))
    assert arr["found"] is True
    assert arr["kind"] in {"array", "field", "struct", "variable", "pointer", "other"}

    ptr_or_var = _j(get_declaration_info.invoke({"symbol": "CanTp_SubState"}))
    assert ptr_or_var["found"] is True

    missing = _j(get_declaration_info.invoke({"symbol": "__missing_decl_name__"}))
    assert missing["found"] is False
    assert missing.get("parse_note")


def test_trimmed_sequence_drops_pre_alarm_reads_and_keeps_alarm_read() -> None:
    seq = _j(
        get_trimmed_sequence.invoke(
            {
                "alarm_order_id": 2475,
                "variable": "BswM_Cfg_LinSmCurrentStateInfo_ast",
            }
        )
    )
    rows = seq.get("sequence") or []
    assert rows, seq
    assert len(rows) == 3, seq
    assert rows[0]["function"] == "BswM_Prv_GetLinSmIdicationInfo_en"
    assert rows[1]["function"] == "BswM_LinSM_CurrentState"
    assert rows[2]["function"] == "BswM_Prv_CopyModeInitValues"
    assert rows[0]["is_alarm_site"] is True
    assert seq.get("origin_resolved") is True
    assert seq.get("scope_mode") == "global"


def test_trimmed_sequence_zero_writes_parameter_case() -> None:
    _j(get_variable_scope.invoke({"symbol": "Index", "alarm_order_id": 5891}))
    seq = _j(get_trimmed_sequence.invoke({"alarm_order_id": 5891, "variable": "Index"}))
    rows = seq.get("sequence") or []
    assert rows == []
    assert seq.get("origin_resolved") is False
    assert seq.get("scope_mode") in {"function", "global"}


def test_function_snippet_reports_index_operands_from_alarm_line() -> None:
    snip = _j(get_function_snippet.invoke({"alarm_order_id": 2475, "context_lines": 40}))
    ops = snip.get("index_operands_at_alarm") or []
    assert ops, snip
    pairs = {(o.get("indexed_object"), o.get("operand_text")) for o in ops}
    assert ("BswM_Cfg_LinSmCurrentStateInfo_ast", "idxMRPChnl_u16") in pairs
    assert ("dataLinSmIndState_aen", "dataReqMode_u16") in pairs
    for op in ops:
        assert "operand_symbol_candidates" in op
        assert "resolved_operand_symbol" in op


def test_index_operand_parser_keeps_raw_nested_and_computed_text() -> None:
    nested = "ret = arr[a][b];"
    computed = "ret = arr[f(x + 1u)];"
    nested_out = _extract_index_operands(nested)
    computed_out = _extract_index_operands(computed)
    assert ("arr", "a") in nested_out
    assert ("arr[a]", "b") in nested_out
    assert ("arr", "f(x + 1u)") in computed_out


def test_function_snippet_follows_context_after_caller_hop() -> None:
    _j(get_trimmed_sequence.invoke({"alarm_order_id": 5891, "variable": "Index"}))
    hop = _j(get_caller_context.invoke({"alarm_order_id": 5891, "parameter": "Index"}))
    snip = _j(get_function_snippet.invoke({}))
    if hop.get("callers"):
        assert snip.get("function_name") in {hop["current_function"], hop["callers"][0]["function"]}
    else:
        assert snip.get("function_name") == hop.get("current_function")


def test_caller_context_shapes() -> None:
    _j(get_trimmed_sequence.invoke({"alarm_order_id": 39, "variable": "checkword"}))
    c1 = _j(get_caller_context.invoke({"alarm_order_id": 39, "parameter": "checkword"}))
    assert "callers" in c1
    assert "unresolved" in c1


def main() -> int:
    store = DataStore.load(ROOT / "data")
    bind_store(store)
    tests = [
        test_tools_exactly_five,
        test_variable_scope_local_global_unknown,
        test_declaration_info_shapes,
        test_trimmed_sequence_drops_pre_alarm_reads_and_keeps_alarm_read,
        test_trimmed_sequence_zero_writes_parameter_case,
        test_function_snippet_reports_index_operands_from_alarm_line,
        test_index_operand_parser_keeps_raw_nested_and_computed_text,
        test_function_snippet_follows_context_after_caller_hop,
        test_caller_context_shapes,
    ]
    for fn in tests:
        fn()
        print(f"ok {fn.__name__}")
    print("all evidence-tool checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
