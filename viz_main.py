#!/usr/bin/env python3
"""CF visualization + agent chat UI: build the graph, then serve the streaming UI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cf_viz.export import build_and_export_full_graph


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build the complete Astrée control-flow graph, then optionally serve "
            "the chat + WebGL UI (Order id → streaming agent investigation + DF highlight)."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data",
        help="Directory with control flow.csv / data flow.csv / Full_alarms.csv",
    )
    parser.add_argument(
        "--process",
        default=None,
        help="Optional Astrée process filter",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=ROOT / "reports" / "cf_full_graph.json",
        help="Where to write the complete CF graph JSON",
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--build-full",
        action="store_true",
        help="Generate the complete CF graph JSON and exit",
    )
    mode.add_argument(
        "--serve",
        action="store_true",
        help="Serve chat + graph UI (builds JSON first if missing)",
    )

    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)

    # Default action: build-full if neither flag given? User wants build then UI.
    # If no mode: print help hint and build-full.
    if not args.build_full and not args.serve:
        args.build_full = True

    if args.build_full or (args.serve and not args.output.exists()):
        print(f"Building complete CF graph from {args.data_dir} ...", file=sys.stderr)
        payload = build_and_export_full_graph(
            args.data_dir,
            args.output,
            process=args.process,
        )
        stats = payload.get("stats", {})
        print(
            f"Wrote {args.output} "
            f"({stats.get('nodes')} nodes, {stats.get('edges')} edges)",
            file=sys.stderr,
        )
        if args.build_full and not args.serve:
            return 0

    if args.serve:
        import os

        os.environ["AOOB_DATA_DIR"] = str(args.data_dir)
        os.environ["AOOB_GRAPH_JSON"] = str(args.output)
        if args.process:
            os.environ["AOOB_PROCESS"] = args.process

        import uvicorn

        print(
            f"Serving UI at http://{args.host}:{args.port}/ "
            f"(graph={args.output})",
            file=sys.stderr,
        )
        uvicorn.run(
            "cf_viz.server:app",
            host=args.host,
            port=args.port,
            reload=False,
        )
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
