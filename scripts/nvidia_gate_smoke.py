#!/usr/bin/env python3
"""NVIDIA smoke of next-hop gate + close-case re-prompt (full 10-tool schema)."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aoob_agent.agent import investigate_alarm, llm_backend_label, load_env
from aoob_agent.data_store import DataStore

CASES = (2475, 39)


def main() -> int:
    load_env(ROOT)
    os.environ["AOOB_LLM_BACKEND"] = "nvidia"
    os.environ.pop("AOOB_MAX_VISIBLE_TOOLS", None)
    os.environ["AOOB_NEXT_HOP_GATE"] = "1"
    os.environ["AOOB_CLOSE_CASE_REPROMPT"] = "1"

    print(f"backend={llm_backend_label()} next_hop=on reprompt=on cap=all", flush=True)
    store = DataStore.load(ROOT / "data")
    results = []
    for oid in CASES:
        print(f"\n=== GATE SMOKE Order {oid} ===", flush=True)
        t0 = time.time()
        row: dict = {"order_id": oid}
        try:
            report = investigate_alarm(oid, store)
            dump = report.model_dump()
            be = dump.get("bounds_evidence") or {}
            row.update(
                {
                    "classification": dump.get("classification"),
                    "confidence": dump.get("confidence"),
                    "evidence_completeness": be.get("evidence_completeness"),
                    "index_origin_summary": be.get("index_origin_summary"),
                    "comment": dump.get("comment"),
                    "tools_used": dump.get("tools_used"),
                    "elapsed_s": round(time.time() - t0, 1),
                }
            )
        except Exception as exc:  # noqa: BLE001
            row["error"] = str(exc)
            row["elapsed_s"] = round(time.time() - t0, 1)
        results.append(row)
        print(
            f"DONE {oid} cls={row.get('classification')} "
            f"complete={row.get('evidence_completeness')} {row.get('elapsed_s')}s",
            flush=True,
        )

    out = ROOT / "reports" / "nvidia_gate_smoke.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    n_resolved = sum(1 for r in results if r.get("classification") in {"true", "false"})
    payload = {
        "resolution_rate": n_resolved / len(results) if results else None,
        "n_resolved": n_resolved,
        "results": results,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    print(f"Wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
