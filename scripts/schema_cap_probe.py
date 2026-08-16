#!/usr/bin/env python3
"""Isolate schema-count vs history length on cliff cases.

Binds exactly AOOB_MAX_VISIBLE_TOOLS schemas (default 3), dynamically swapped,
with next-hop / close-case re-prompt OFF so the only loop change is schema count.
Compare first tool_calls=0 against the known 4–6 cliff from the full 10-tool list.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aoob_agent.agent import llm_backend_label, load_env, stream_investigate
from aoob_agent.data_store import DataStore

# Prior balanced NVIDIA run first-zero clustered at tools_so_far 4–6.
CLIFF_CASES = (2475, 39)
PRIOR_CLIFF = "4-6"


def main() -> int:
    load_env(ROOT)
    os.environ["AOOB_LLM_BACKEND"] = os.environ.get("AOOB_LLM_BACKEND") or "nvidia"
    os.environ["AOOB_MAX_VISIBLE_TOOLS"] = os.environ.get("AOOB_MAX_VISIBLE_TOOLS") or "3"
    os.environ["AOOB_NEXT_HOP_GATE"] = "0"
    os.environ["AOOB_CLOSE_CASE_REPROMPT"] = "0"

    cap = os.environ["AOOB_MAX_VISIBLE_TOOLS"]
    out = ROOT / "reports" / "schema_cap_probe.json"
    print(
        f"backend={llm_backend_label()} cap={cap} next_hop=off "
        f"reprompt=off cases={list(CLIFF_CASES)}",
        flush=True,
    )
    store = DataStore.load(ROOT / "data")
    results = []
    for oid in CLIFF_CASES:
        print(f"\n=== PROBE Order {oid} ===", flush=True)
        t0 = time.time()
        tools_so_far = 0
        zeros: list[dict] = []
        classification = None
        error = None
        try:
            for event in stream_investigate(oid, store):
                et = event.get("type")
                if et == "tool_result":
                    tools_so_far += 1
                elif et == "no_tool_call":
                    zeros.append(
                        {
                            "tools_so_far": tools_so_far,
                            "elapsed_s": round(time.time() - t0, 1),
                        }
                    )
                    print(
                        f"[probe] {'first_zero' if len(zeros) == 1 else 'later_zero'} "
                        f"tools_so_far={tools_so_far}",
                        flush=True,
                    )
                elif et == "report":
                    classification = (event.get("data") or {}).get("classification")
                elif et == "error":
                    error = event.get("message")
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
        row = {
            "order_id": oid,
            "cap": int(cap),
            "prior_cliff_tools_so_far": PRIOR_CLIFF,
            "first_zero_at_tools_so_far": zeros[0]["tools_so_far"] if zeros else None,
            "zero_events": zeros,
            "tools_completed": tools_so_far,
            "classification": classification,
            "elapsed_s": round(time.time() - t0, 1),
            "error": error,
            "cliff_moved": (
                None
                if not zeros
                else zeros[0]["tools_so_far"] > 6 or zeros[0]["tools_so_far"] < 4
            ),
        }
        results.append(row)
        print(
            f"DONE {oid} first_zero={row['first_zero_at_tools_so_far']} "
            f"prior={PRIOR_CLIFF} moved={row['cliff_moved']} "
            f"cls={classification} {row['elapsed_s']}s",
            flush=True,
        )

    payload = {
        "note": (
            "If first_zero stays in 4–6, schema count is not the driver "
            "(history trim / turn depth more likely). Staging would not move the cliff. "
            "If first_zero moves later, staged/capped schemas are worth enabling."
        ),
        "results": results,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    print(f"Wrote {out}", flush=True)
    return 0 if not any(r.get("error") for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
