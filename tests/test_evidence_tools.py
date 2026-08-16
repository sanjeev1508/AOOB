#!/usr/bin/env python3
"""Acceptance checks for declaration bounds + backward-slice tools (retrieval only)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aoob_agent.data_store import DataStore
from aoob_agent.tools import (
    bind_store,
    get_affected_symbols,
    get_all_writes_to_symbol,
    get_backward_slice,
    get_condition_guards,
    get_declaration_bounds,
    get_index_expression_structure,
    parse_index_expression_structure,
    resolve_symbolic_constant,
)


def main() -> int:
    store = DataStore.load(ROOT / "data")
    bind_store(store)
    print(f"declarations indexed: {len(store.declarations)}")

    # --- Alarm payload: Astrée message from CSV; never human classification/comment ---
    for oid in (40, 2474):
        sym = json.loads(get_affected_symbols.invoke({"alarm_order_id": oid}))
        alarm = sym.get("alarm") or {}
        print(f"\n[alarm {oid}]")
        print("message:", repr((alarm.get("message") or "")[:120]))
        assert "classification" not in alarm, "human classification must not reach tools"
        assert "comment" not in alarm, "comment/meta must not reach tools"
        assert "message" in alarm, "message key must be present"
        assert alarm.get("message"), (
            f"Order {oid}: Full_alarms.csv Message should carry Astrée diagnostic text"
        )
        assert "true" != alarm["message"].strip().lower()
        assert "false" != alarm["message"].strip().lower()

    # --- literal-sized array used by alarm 39 ---
    lit = json.loads(
        get_declaration_bounds.invoke(
            {"symbol_name": "MOCMEM_VBLSTRUCTEARLYCLEAREDRAM_MO_INST_LOOKUP"}
        )
    )
    print("\n[literal declaration]")
    print(json.dumps(lit, indent=2)[:800])
    assert lit.get("found") is True
    assert lit.get("array_size") == 2, lit
    assert lit.get("parse_note") in (None, "")

    # --- typedef + `1u` size + `{` on next line (Order 2475 array) ---
    lin = json.loads(
        get_declaration_bounds.invoke(
            {"symbol_name": "BswM_Cfg_LinSmCurrentStateInfo_ast"}
        )
    )
    print("\n[LinSmCurrentStateInfo_ast]")
    print(json.dumps(lin, indent=2)[:800])
    assert lin.get("found") is True, lin
    assert lin.get("array_size") == 1, lin

    # --- not-found must carry parse_note ---
    missing = json.loads(
        get_declaration_bounds.invoke({"symbol_name": "__no_such_symbol_zz__"})
    )
    print("\n[missing declaration]")
    print(json.dumps(missing, indent=2)[:600])
    assert missing.get("found") is False
    assert missing.get("parse_note"), "not-found must include parse_note"

    # --- non-literal / ambiguous size must keep parse_note, not invent a number ---
    ambiguous = None
    for name, rec in store.declarations.items():
        if rec.parse_note and rec.array_size is None and "[" in (rec.declared_type or ""):
            ambiguous = name
            break
    assert ambiguous, "Expected at least one declaration with parse_note / non-literal size"
    amb = json.loads(get_declaration_bounds.invoke({"symbol_name": ambiguous}))
    print(f"\n[ambiguous declaration: {ambiguous}]")
    print(json.dumps(amb, indent=2)[:800])
    assert amb.get("array_size") is None
    assert amb.get("parse_note"), "parse_note required when size is not a literal"

    # --- enclosing function for multi-line header (Order 2475 site) ---
    fn, a, b = store.find_enclosing_function(82154)
    print(f"\n[enclosing @82154] {fn} {a}-{b}")
    assert fn == "BswM_Prv_GetLinSmIdicationInfo_en", fn
    assert a <= 82154 <= b

    # --- backward slice for checkword at alarm 39 location ---
    loc = "ALL_bc_with_context.c:331731.56-85"
    slice_payload = json.loads(
        get_backward_slice.invoke(
            {
                "variable_name": "checkword",
                "from_location": loc,
                "max_depth": 3,
            }
        )
    )
    print("\n[backward slice checkword @ alarm 39]")
    print(
        json.dumps(
            {
                "writes": len(slice_payload.get("writes_found") or []),
                "unresolved": slice_payload.get("unresolved_paths"),
                "truncated": slice_payload.get("truncated"),
                "start_function": slice_payload.get("start_function"),
                "sample_write": (slice_payload.get("writes_found") or [None])[0],
                "lookup": slice_payload.get("lookup"),
            },
            indent=2,
        )
    )
    assert "writes_found" in slice_payload
    assert slice_payload["writes_found"], (
        "Expected at least the local initializer / prior write for checkword"
    )
    assert "unresolved_paths" in slice_payload
    assert "truncated" in slice_payload
    assert slice_payload.get("lookup", {}).get("df_keys_matched") is not None

    # --- idxMRPChnl_u16: name matching must surface DF keys even if prior writes empty ---
    idx_slice = json.loads(
        get_backward_slice.invoke(
            {
                "variable_name": "idxMRPChnl_u16",
                "from_location": "ALL_bc_with_context.c:82154.91-106",
                "max_depth": 4,
            }
        )
    )
    print("\n[backward slice idxMRPChnl_u16 @ alarm 2475]")
    print(
        json.dumps(
            {
                "start_function": idx_slice.get("start_function"),
                "writes": len(idx_slice.get("writes_found") or []),
                "lookup": idx_slice.get("lookup"),
                "functions_visited": idx_slice.get("functions_visited"),
            },
            indent=2,
        )
    )
    lookup = idx_slice.get("lookup") or {}
    assert lookup.get("df_keys_matched"), "bare/qualified DF keys should match"
    assert lookup.get("df_write_count", 0) >= 1, (
        "DF has writes for this name — empty prior writes must not look like a key miss"
    )

    # Force a shallow depth to observe truncated / unresolved structural flags.
    shallow = json.loads(
        get_backward_slice.invoke(
            {
                "variable_name": "checkword",
                "from_location": loc,
                "max_depth": 0,
            }
        )
    )
    print("\n[backward slice max_depth=0]")
    print(
        json.dumps(
            {
                "truncated": shallow.get("truncated"),
                "unresolved_paths": shallow.get("unresolved_paths"),
                "writes": len(shallow.get("writes_found") or []),
            },
            indent=2,
        )
    )
    assert shallow.get("truncated") or shallow.get("unresolved_paths"), (
        "Expected truncated and/or unresolved_paths at depth 0 — "
        "traversal must not invent an origin past graph limits"
    )

    # --- index expression structure (Order 1129 pattern) ---
    idx = parse_index_expression_structure(
        "((DFC_stMem[((idDFC.id) >> 4u)]) &= (uint16)(~((uint)(1uL << ((idDFC.id) & 0xFu)))))"
    )
    print("\n[index expression structure]")
    print(json.dumps(idx, indent=2)[:800])
    assert idx.get("found")
    assert "right_shift" in {op.get("op") for op in idx.get("operations") or []}
    assert idx.get("operations")[0].get("amount") == 4
    assert "idDFC" in (idx.get("operand") or "")

    tool_idx = json.loads(
        get_index_expression_structure.invoke(
            {"expression_text": "(idDFC.id) >> 4u"}
        )
    )
    assert tool_idx.get("operand") == "idDFC.id"
    assert tool_idx["operations"][0]["op"] == "right_shift"

    # --- symbolic constant (Order 1926 sentinel) ---
    const = json.loads(
        resolve_symbolic_constant.invoke(
            {"name": "NvM_Prv_idJob_Invalid_e___E1861698"}
        )
    )
    print("\n[resolve symbolic constant]")
    print(json.dumps(const, indent=2)[:800])
    assert const.get("found"), const
    assert const.get("kind") == "enum_member"
    assert const.get("value") == 16, const  # Idle=0 … Invalid=16, Count=17

    # --- condition guards at Order 1926 access ---
    guards = json.loads(
        get_condition_guards.invoke(
            {
                "variable_name": "idJob_en",
                "from_location": "ALL_bc_with_context.c:196958.54-62",
            }
        )
    )
    print("\n[condition guards]")
    print(json.dumps(guards, indent=2)[:1200])
    assert any(
        "idJob_en" in (g.get("condition_text") or "")
        for g in guards.get("guards_found") or []
    ), guards

    # --- flat writes cross-check ---
    writes = json.loads(
        get_all_writes_to_symbol.invoke({"symbol_name": "checkword"})
    )
    print("\n[all writes checkword]")
    print(
        json.dumps(
            {
                "write_count": writes.get("write_count"),
                "truncated": writes.get("truncated"),
                "sample": (writes.get("writes") or [])[:3],
            },
            indent=2,
        )
    )
    assert writes.get("write_count", 0) >= 1

    # --- SlacCurProfileValidation field tracing (batch_eval FN cluster) ---
    a6413 = store.get_alarm(6413)
    assert a6413 is not None
    for name in (
        "SlacCurProfileValidation",
        "EthTrcv_30_Ar7000_Slac_MgmtStruct.SlacCurProfileValidation",
    ):
        sl = json.loads(
            get_backward_slice.invoke(
                {
                    "variable_name": name,
                    "from_location": a6413.location,
                    "max_depth": 8,
                }
            )
        )
        print(f"\n[slice {name}] writes={len(sl.get('writes_found') or [])}")
        assert len(sl.get("writes_found") or []) >= 1, sl.get("lookup")
        sample = (sl.get("writes_found") or [])[0]
        assert "SlacCurProfileValidation" in (
            sample.get("assigned_expression_text") or ""
        )

    # --- CanTp_SubState must not report size 0 from call-site false match ---
    cantp = json.loads(
        get_declaration_bounds.invoke({"symbol_name": "CanTp_SubState"})
    )
    print("\n[CanTp_SubState declaration]")
    print(json.dumps(cantp, indent=2)[:600])
    assert cantp.get("found")
    assert cantp.get("array_size") == 10, cantp
    assert cantp.get("array_size") != 0

    # --- condition guards include for/while kinds; tool is registered ---
    from aoob_agent.tools import TOOLS, get_call_site_arguments

    assert "get_condition_guards" in {t.name for t in TOOLS}
    assert "get_call_site_arguments" in {t.name for t in TOOLS}

    # --- Order 5891: attribute-decorated inline + Index must not cross-contaminate ---
    a5891 = store.get_alarm(5891)
    assert a5891 is not None
    fn, start, end = store.find_enclosing_function(
        store.parse_location_line(a5891.location) or 0
    )
    assert fn and "TcpIp_GetIpV6SocketDynIdxOfIpV6EthBufData" in fn, fn
    assert fn != "__attribute__", fn
    sl5891 = json.loads(
        get_backward_slice.invoke(
            {
                "variable_name": "Index",
                "from_location": a5891.location,
                "max_depth": 8,
            }
        )
    )
    assert sl5891.get("start_function") == fn, sl5891.get("start_function")
    for w in sl5891.get("writes_found") or []:
        assert w.get("function") in {None, fn} or "CanIf_Prv_GetFreeTxBufferIndex" not in str(
            w.get("function")
        ), w
    # Generic Index must not pull unrelated global for-loops.
    assert not any(
        "CanIf_Prv_GetFreeTxBufferIndex" in str(w.get("function"))
        for w in (sl5891.get("writes_found") or [])
    ), sl5891.get("writes_found")

    # Parameter path embeds call-site argument navigation (no order hardcoding).
    assert sl5891.get("parameter_note"), sl5891
    csa = sl5891.get("call_site_arguments") or {}
    assert csa.get("parameter_type_width", {}).get("abstract_max") == 255, csa
    assert "ethBufDataIdx" in (csa.get("next_slice_candidates") or []), csa
    assert any(
        (s.get("argument_expression") or "").find("ethBufDataIdx") >= 0
        for s in (csa.get("call_sites") or [])
    ), csa.get("call_sites")

    csa2 = json.loads(
        get_call_site_arguments.invoke(
            {
                "callee_function": fn,
                "parameter_name": "Index",
                "from_location": a5891.location,
            }
        )
    )
    assert csa2.get("next_slice_candidates"), csa2

    # Union object + .raw[2] — DF CSV uses bare TcpIp_IpV6EthBufData; bound is .raw.
    obj5891 = json.loads(
        get_declaration_bounds.invoke({"symbol_name": "TcpIp_IpV6EthBufData"})
    )
    assert obj5891.get("found") is True, obj5891
    assert obj5891.get("kind") == "object", obj5891
    assert obj5891.get("array_size") is None, obj5891
    nested = obj5891.get("nested_array_fields") or []
    assert any(f.get("array_size") == 2 and "raw" in (f.get("name") or "") for f in nested), nested
    raw5891 = json.loads(
        get_declaration_bounds.invoke({"symbol_name": "TcpIp_IpV6EthBufData.raw"})
    )
    assert raw5891.get("found") is True and raw5891.get("array_size") == 2, raw5891

    print("\nOK — retrieval tools behave structurally (no classification).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
