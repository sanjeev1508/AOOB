#!/usr/bin/env python3
"""Unit checks for next-hop / schema-cap / close-case helpers (no LLM)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_core.messages import ToolMessage

from aoob_agent.agent import (
    _idents_from_c_text,
    pending_named_hop,
    select_visible_tool_names,
    symbol_explicitly_unresolvable,
    tools_returned_closing_evidence,
)


def _tool(name: str, payload: dict) -> ToolMessage:
    return ToolMessage(content=json.dumps(payload), tool_call_id="t1", name=name)


def test_idents_from_assignment() -> None:
    names = _idents_from_c_text("idxMRPChnl_u16 = headIndex_u8;")
    assert "headIndex_u8" in names, names


def test_pending_hop_from_write_rhs() -> None:
    msgs = [
        _tool(
            "get_backward_slice",
            {
                "variable": "idxMRPChnl_u16",
                "writes_found": [
                    {
                        "assigned_expression_text": "idxMRPChnl_u16 = headIndex_u8;",
                    }
                ],
                "unresolved_paths": [],
                "lookup": {"queried_name": "idxMRPChnl_u16", "df_keys_matched": ["idxMRPChnl_u16"]},
            },
        )
    ]
    assert pending_named_hop(msgs) == "headIndex_u8"


def test_hop_released_when_df_keys_empty() -> None:
    msgs = [
        _tool(
            "get_backward_slice",
            {
                "variable": "idxMRPChnl_u16",
                "writes_found": [
                    {"assigned_expression_text": "idxMRPChnl_u16 = UntraceableMacro;"}
                ],
                "lookup": {"queried_name": "idxMRPChnl_u16", "df_keys_matched": ["idxMRPChnl_u16"]},
            },
        ),
        _tool(
            "get_backward_slice",
            {
                "variable": "UntraceableMacro",
                "writes_found": [],
                "unresolved_paths": [],
                "lookup": {
                    "queried_name": "UntraceableMacro",
                    "df_keys_matched": [],
                },
                "parse_note": "No data-flow rows for this name.",
            },
        ),
    ]
    assert symbol_explicitly_unresolvable(msgs, "UntraceableMacro") is True
    assert pending_named_hop(msgs) is None


def test_cf_boundary_does_not_release() -> None:
    msgs = [
        _tool(
            "get_backward_slice",
            {
                "variable": "idxMRPChnl_u16",
                "writes_found": [
                    {"assigned_expression_text": "x = headIndex_u8;"}
                ],
                "lookup": {"df_keys_matched": ["idxMRPChnl_u16"]},
            },
        ),
        _tool(
            "get_backward_slice",
            {
                "variable": "headIndex_u8",
                "writes_found": [],
                "unresolved_paths": [
                    {
                        "reason": "reached function with no visible caller in control-flow graph",
                        "function": "Foo",
                    }
                ],
                "lookup": {
                    "queried_name": "headIndex_u8",
                    "df_keys_matched": ["headIndex_u8"],
                },
            },
        ),
    ]
    assert symbol_explicitly_unresolvable(msgs, "headIndex_u8") is False
    assert pending_named_hop(msgs) == "headIndex_u8"


def test_hop_cleared_after_successful_slice() -> None:
    msgs = [
        _tool(
            "get_backward_slice",
            {
                "variable": "idxMRPChnl_u16",
                "writes_found": [
                    {"assigned_expression_text": "idxMRPChnl_u16 = headIndex_u8;"}
                ],
                "lookup": {"df_keys_matched": ["idxMRPChnl_u16"]},
            },
        ),
        _tool(
            "get_backward_slice",
            {
                "variable": "headIndex_u8",
                "writes_found": [
                    {"assigned_expression_text": "headIndex_u8 = 0u;"}
                ],
                "lookup": {"df_keys_matched": ["headIndex_u8"]},
            },
        ),
    ]
    assert pending_named_hop(msgs) is None


def test_closing_evidence_bound_and_shift() -> None:
    bound = [
        _tool(
            "get_declaration_bounds",
            {"symbol": "Table", "found": True, "array_size": 2},
        )
    ]
    assert tools_returned_closing_evidence(bound) is True
    shift = [
        _tool(
            "get_index_expression_structure",
            {
                "found": True,
                "operand": "idDFC.id",
                "operations": [{"op": "shift_right", "amount": 4}],
            },
        )
    ]
    assert tools_returned_closing_evidence(shift) is True
    empty = [
        _tool("get_declaration_bounds", {"symbol": "x", "found": False, "array_size": None})
    ]
    assert tools_returned_closing_evidence(empty) is False


def test_visible_tools_capped_at_site_stage() -> None:
    os.environ["AOOB_MAX_VISIBLE_TOOLS"] = "3"
    os.environ["AOOB_NEXT_HOP_GATE"] = "0"
    try:
        names = select_visible_tool_names([])
        assert len(names) == 3, names
        assert names[0] == "get_affected_symbols"
        after_site = [
            _tool(
                "get_affected_symbols",
                {
                    "index_operand_candidates": ["checkword"],
                    "indexed_objects": ["Table"],
                },
            )
        ]
        names2 = select_visible_tool_names(after_site)
        assert len(names2) == 3, names2
        assert "get_backward_slice" in names2
    finally:
        os.environ.pop("AOOB_MAX_VISIBLE_TOOLS", None)
        os.environ.pop("AOOB_NEXT_HOP_GATE", None)


def test_hop_ignores_aggregate_tokens_in_complex_rhs() -> None:
    msgs = [
        _tool(
            "get_backward_slice",
            {
                "variable": "headIndex_u8",
                "writes_found": [
                    {
                        "assigned_expression_text": (
                            "headIndex_u8 = BswM_Prv_IntrptQueue_ast[idx].field;"
                        )
                    }
                ],
                "lookup": {"df_keys_matched": ["headIndex_u8"]},
            },
        )
    ]
    pending = pending_named_hop(msgs)
    assert pending != "BswM_Prv_IntrptQueue_ast"
    assert pending != "BswM_Prv_IntrptQueue_ast___36"


def test_bare_rhs_is_the_only_write_hop() -> None:
    msgs = [
        _tool(
            "get_backward_slice",
            {
                "variable": "idxMRPChnl_u16",
                "writes_found": [
                    {
                        "assigned_expression_text": (
                            "idxMRPChnl_u16 = (uint16) headIndex_u8;"
                        )
                    }
                ],
                "lookup": {"df_keys_matched": ["idxMRPChnl_u16"]},
            },
        )
    ]
    assert pending_named_hop(msgs) == "headIndex_u8"


def test_full_schema_when_uncapped() -> None:
    os.environ.pop("AOOB_MAX_VISIBLE_TOOLS", None)
    names = select_visible_tool_names([])
    assert len(names) == 10, names


def main() -> int:
    tests = [
        test_idents_from_assignment,
        test_pending_hop_from_write_rhs,
        test_hop_released_when_df_keys_empty,
        test_cf_boundary_does_not_release,
        test_hop_cleared_after_successful_slice,
        test_closing_evidence_bound_and_shift,
        test_visible_tools_capped_at_site_stage,
        test_full_schema_when_uncapped,
        test_hop_ignores_aggregate_tokens_in_complex_rhs,
        test_bare_rhs_is_the_only_write_hop,
    ]
    for fn in tests:
        fn()
        print(f"ok {fn.__name__}")
    print("all agent-gate checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
