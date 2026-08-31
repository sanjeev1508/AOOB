#!/usr/bin/env python3
"""CLI: investigate an Astrée alarm Order id with the LangGraph AOOB agent."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aoob_agent.agent import investigate_alarm, load_env
from aoob_agent.data_store import DataStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="AOOB alarm investigation agent (LangGraph + Ollama or NVIDIA NIM)"
    )
    parser.add_argument(
        "order_id",
        type=str,
        help="Alarm Order id from Full_alarms.csv (commas allowed, e.g. 1,112)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data",
        help="Directory containing the four input files",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model id (Ollama tag if OLLAMA_LOCAL_PATH is set, else NVIDIA model id)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the JSON report",
    )
    args = parser.parse_args(argv)
    order_id = DataStore._parse_order_id(args.order_id)
    if order_id is None:
        print(f"Error: invalid order id {args.order_id!r}", file=sys.stderr)
        return 1

    load_env(ROOT)
    print(f"Loading data from {args.data_dir} ...", file=sys.stderr)
    store = DataStore.load(args.data_dir)
    alarm = store.get_alarm(order_id)
    if alarm is None:
        print(f"Error: alarm Order {order_id} not found.", file=sys.stderr)
        return 1
    print(
        f"Investigating Order {alarm.order}: {alarm.category} @ {alarm.location}",
        file=sys.stderr,
    )

    report = investigate_alarm(order_id, store, model=args.model)
    payload = report.model_dump()
    text = json.dumps(payload, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
