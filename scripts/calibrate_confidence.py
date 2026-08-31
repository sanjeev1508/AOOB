#!/usr/bin/env python3
"""Calibration harness: bucket agent-stated confidence vs human ground truth."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GT_PATH = ROOT / "data" / "Full_alarms_with_result.csv"
DEFAULT_BATCH = ROOT / "reports" / "batch_eval_100.json"


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


def _human_match(agent_cls: str, human: str) -> bool:
    h = human.lower()
    a = (agent_cls or "").lower()
    if h.startswith("true"):
        return a in {"true", "true (low)", "review"}
    if h == "false":
        return a == "false"
    return False


def _bucket(results: list[dict], gt: dict[int, str]) -> dict:
    by_conf: dict[str, list[dict]] = defaultdict(list)
    for row in results:
        conf = str(row.get("confidence") or "unknown").lower()
        by_conf[conf].append(row)

    calibration: dict[str, dict] = {}
    for conf, rows in sorted(by_conf.items()):
        scored = [r for r in rows if r.get("agent_classification") and r.get("order_id") in gt]
        matches = sum(
            1
            for r in scored
            if _human_match(str(r.get("agent_classification")), gt[int(r["order_id"])])
        )
        fn_on_true = sum(
            1
            for r in scored
            if gt[int(r["order_id"])].lower().startswith("true")
            and r.get("agent_classification") == "false"
        )
        calibration[conf] = {
            "n": len(scored),
            "match_rate": matches / len(scored) if scored else None,
            "false_on_human_true": fn_on_true,
        }

    tokens = [
        float((r.get("report") or {}).get("tokens_used") or 0)
        for r in results
        if (r.get("report") or {}).get("tokens_used")
    ]
    savings = [
        float((r.get("report") or {}).get("token_savings_percent") or 0)
        for r in results
        if (r.get("report") or {}).get("token_savings_percent") is not None
    ]

    def _stats(vals: list[float]) -> dict:
        if not vals:
            return {"mean": None, "median": None, "p95": None}
        s = sorted(vals)
        n = len(s)
        p95_idx = min(n - 1, int(0.95 * n))
        return {
            "mean": round(sum(s) / n, 2),
            "median": round(s[n // 2], 2),
            "p95": round(s[p95_idx], 2),
        }

    return {
        "confidence_buckets": calibration,
        "token_savings_percent": _stats(savings),
        "tokens_used": _stats(tokens),
        "n_results": len(results),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate agent confidence vs ground truth")
    parser.add_argument(
        "--batch",
        type=Path,
        default=DEFAULT_BATCH,
        help="Path to batch_eval JSON output",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "reports" / "confidence_calibration.json",
    )
    args = parser.parse_args(argv)

    if not args.batch.exists():
        print(f"Batch file not found: {args.batch}", file=sys.stderr)
        return 1

    data = json.loads(args.batch.read_text(encoding="utf-8"))
    results = data.get("results") or []
    gt = _load_ground_truth()
    report = _bucket(results, gt)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
