#!/usr/bin/env python3
"""Validate recall floor: all human-true alarms must not be classified false."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from aoob_agent.agent import stream_investigate
from aoob_agent.session import clear_verdict_registry
from aoob_agent.data_store import DataStore

GT_PATH = ROOT / "data" / "Full_alarms_with_result.csv"
OUT = ROOT / "reports" / "recall_floor_validation.json"


def _load_ground_truth() -> dict[int, str]:
    text = GT_PATH.read_text(encoding="utf-8-sig")
    lines = text.splitlines()
    start = 1 if lines and lines[0].lower().startswith("sep=") else 0
    reader = csv.DictReader(lines[start:], delimiter=";")
    out: dict[int, str] = {}
    for row in reader:
        raw = (row.get("Order") or "").replace(",", "").strip()
        try:
            oid = int(raw)
        except ValueError:
            continue
        label = (row.get("Classification") or "").strip()
        if label:
            out[oid] = label
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Recall floor on human-true alarms")
    parser.add_argument("--orders", type=int, nargs="*", default=None)
    args = parser.parse_args(argv)

    gt = _load_ground_truth()
    true_orders = sorted(
        oid for oid, lab in gt.items() if lab.lower().startswith("true")
    )
    if args.orders:
        true_orders = [o for o in args.orders if o in gt]

    store = DataStore.load(ROOT / "data")
    clear_verdict_registry()
    results: list[dict] = []
    violations: list[int] = []

    for oid in true_orders:
        human = gt[oid]
        print(f"RUN Order {oid} human={human}", flush=True)
        t0 = time.time()
        dump = None
        err = None
        for event in stream_investigate(oid, store):
            if event.get("type") == "report":
                dump = event.get("data")
            elif event.get("type") == "error":
                err = event.get("message")
        agent_cls = dump.get("classification") if dump else None
        recall_safe = agent_cls != "false"
        if not recall_safe:
            violations.append(oid)
        results.append(
            {
                "order_id": oid,
                "human": human,
                "agent_classification": agent_cls,
                "recall_safe": recall_safe,
                "error": err,
                "elapsed_s": round(time.time() - t0, 1),
                "report": dump,
            }
        )
        print(
            f"  -> agent={agent_cls} recall_safe={recall_safe}",
            flush=True,
        )

    summary = {
        "n_true_alarms": len(true_orders),
        "n_violations": len(violations),
        "violations": violations,
        "passed": len(violations) == 0,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"summary": summary, "results": results}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {OUT}")
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
