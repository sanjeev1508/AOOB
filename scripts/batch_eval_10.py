#!/usr/bin/env python3
"""Batch-run agent on a fixed 10-alarm sample and score vs ground truth."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aoob_agent.agent import investigate_alarm, llm_backend_label, load_env
from aoob_agent.data_store import DataStore

# 3 positives / 7 negatives — accuracy is confounded by the false base rate (70%).
SAMPLE_SKEWED = [
    (2475, "true"),
    (6413, "true (low)"),
    (6355, "true (low)"),
    (39, "false"),
    (41, "false"),
    (48, "false"),
    (91, "false"),
    (1129, "false"),
    (1926, "false"),
    (2858, "false"),
]

# 5 / 5 — always-false baseline is 50%; balanced_accuracy exposes coin-flip.
SAMPLE_BALANCED = [
    (2475, "true"),
    (6413, "true (low)"),
    (6355, "true (low)"),
    (6362, "true (low)"),
    (7256, "true (low)"),
    (39, "false"),
    (48, "false"),
    (1129, "false"),
    (1926, "false"),
    (2858, "false"),
]

SAMPLE = SAMPLE_SKEWED

# Comment used Astrée [lo,hi] vs declared size as if that were a runtime index.
_SHORTCUT_RE = re.compile(
    r"(exceeds the declared size|"
    r"less than (the )?(declared|array)|"
    r"abstract interval.{0,120}(exceed|not included|larger)|"
    r"\[0,\s*\d+\].{0,80}(declared|array size|size of)|"
    r"array size.{0,50}(less than|smaller than).{0,30}(hi|astr)|"
    r"which exceeds the (declared|array))",
    re.IGNORECASE | re.DOTALL,
)


def _interval_shortcut(comment: str) -> bool:
    return bool(_SHORTCUT_RE.search(comment or ""))


def _quality_flags(dump: dict, match: bool) -> dict:
    be = dump.get("bounds_evidence") or {}
    comment = dump.get("comment") or ""
    shortcut = _interval_shortcut(comment)
    cls = dump.get("classification")
    missing_be = not bool((be.get("index_origin_summary") or "").strip())
    missing_completeness = not bool(be.get("evidence_completeness"))
    return {
        "interval_shortcut": shortcut,
        "missing_bounds_evidence": missing_be,
        "missing_evidence_completeness": missing_completeness,
        # Right true/false label, but via the forbidden interval-as-index rationale.
        "silent_failure": bool(shortcut and match),
        # Same rationale used to justify review (new dumping-ground shape).
        "shortcut_on_review": bool(shortcut and cls == "review"),
        "forbidden_rationale": shortcut,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score the fixed 10-alarm sample vs human labels"
    )
    parser.add_argument(
        "--backend",
        choices=["nvidia", "ollama"],
        default=None,
        help="Force LLM backend (sets AOOB_LLM_BACKEND for this process)",
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="Use 5 true / 5 false sample (always-false baseline 50%)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="JSON output path (default reports/batch_eval_10_<backend>.json)",
    )
    args = parser.parse_args(argv)

    load_env(ROOT)
    if args.backend:
        os.environ["AOOB_LLM_BACKEND"] = args.backend

    sample = SAMPLE_BALANCED if args.balanced else SAMPLE_SKEWED
    backend = args.backend or ("ollama" if os.getenv("OLLAMA_LOCAL_PATH") else "nvidia")
    suffix = f"{backend}_balanced" if args.balanced else backend
    out = args.output or (ROOT / "reports" / f"batch_eval_10_{suffix}.json")

    print(f"backend={llm_backend_label()} sample={'5/5' if args.balanced else '3/7'} output={out}", flush=True)
    store = DataStore.load(ROOT / "data")
    results = []
    for oid, human in sample:
        if oid not in store.alarms:
            results.append(
                {
                    "order_id": oid,
                    "human": human,
                    "error": "missing from Full_alarms.csv",
                }
            )
            print(f"SKIP {oid} missing", flush=True)
            continue
        print(f"\n=== RUN Order {oid} (human={human}) ===", flush=True)
        t0 = time.time()
        try:
            report = investigate_alarm(oid, store)
            dump = report.model_dump()
            elapsed = round(time.time() - t0, 1)
            agent_cls = dump.get("classification")
            match = (
                agent_cls == "true"
                if human.startswith("true")
                else agent_cls == "false"
            )
            flags = _quality_flags(dump, match)
            row = {
                "order_id": oid,
                "human": human,
                "agent_classification": agent_cls,
                "match": match,
                "confidence": dump.get("confidence"),
                "comment": dump.get("comment"),
                "astree_message": dump.get("astree_message"),
                "location": dump.get("location"),
                "evidence_completeness": (dump.get("bounds_evidence") or {}).get(
                    "evidence_completeness"
                ),
                "index_origin_summary": (dump.get("bounds_evidence") or {}).get(
                    "index_origin_summary"
                ),
                "tools_used": dump.get("tools_used"),
                "elapsed_s": elapsed,
                **flags,
                "report": dump,
            }
            results.append(row)
            print(
                f"DONE {oid} agent={agent_cls} human={human} match={match} "
                f"conf={dump.get('confidence')} shortcut={flags['interval_shortcut']} "
                f"silent={flags['silent_failure']} "
                f"shortcut_on_review={flags['shortcut_on_review']} {elapsed}s",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            elapsed = round(time.time() - t0, 1)
            results.append(
                {
                    "order_id": oid,
                    "human": human,
                    "error": str(exc),
                    "elapsed_s": elapsed,
                    "match": False,
                }
            )
            print(f"FAIL {oid}: {exc} ({elapsed}s)", flush=True)

        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")

    scored = [r for r in results if "agent_classification" in r]
    n_true = sum(1 for r in scored if r["human"].startswith("true"))
    n_false = sum(1 for r in scored if r["human"] == "false")
    tp = sum(
        1
        for r in scored
        if r["human"].startswith("true") and r["agent_classification"] == "true"
    )
    tn = sum(
        1
        for r in scored
        if r["human"] == "false" and r["agent_classification"] == "false"
    )
    fp = sum(
        1
        for r in scored
        if r["human"] == "false" and r["agent_classification"] == "true"
    )
    fn = sum(
        1
        for r in scored
        if r["human"].startswith("true") and r["agent_classification"] == "false"
    )
    n_review = sum(1 for r in scored if r.get("agent_classification") == "review")
    n_review_on_true = sum(
        1
        for r in scored
        if r["human"].startswith("true") and r.get("agent_classification") == "review"
    )
    n_review_on_false = sum(
        1
        for r in scored
        if r["human"] == "false" and r.get("agent_classification") == "review"
    )
    tpr = tp / n_true if n_true else None
    tnr = tn / n_false if n_false else None
    n_shortcut = sum(1 for r in scored if r.get("interval_shortcut"))
    n_silent = sum(1 for r in scored if r.get("silent_failure"))
    n_shortcut_on_review = sum(1 for r in scored if r.get("shortcut_on_review"))
    n_missing_be = sum(1 for r in scored if r.get("missing_bounds_evidence"))
    n_pred_true = sum(1 for r in scored if r.get("agent_classification") == "true")
    n_pred_false = sum(1 for r in scored if r.get("agent_classification") == "false")
    n_resolved = n_pred_true + n_pred_false
    resolution_rate = (n_resolved / len(scored)) if scored else None
    resolved = [r for r in scored if r.get("agent_classification") in {"true", "false"}]
    human_match_among_resolved = (
        (sum(1 for r in resolved if r.get("match")) / len(resolved))
        if resolved
        else None
    )
    summary = {
        "backend": llm_backend_label(),
        "sample_balance": f"{n_true} true / {n_false} false",
        "always_false_baseline": (n_false / len(scored)) if scored else None,
        "n_scored": len(scored),
        "n_errors": sum(1 for r in results if r.get("error")),
        "accuracy": (tp + tn) / len(scored) if scored else None,
        "balanced_accuracy": (
            ((tpr or 0) + (tnr or 0)) / 2 if tpr is not None and tnr is not None else None
        ),
        "tpr": tpr,
        "tnr": tnr,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "n_review": n_review,
        "n_review_on_true": n_review_on_true,
        "n_review_on_false": n_review_on_false,
        "n_pred_true": n_pred_true,
        "n_pred_false": n_pred_false,
        "n_resolved": n_resolved,
        "resolution_rate": resolution_rate,
        "human_match_among_resolved": human_match_among_resolved,
        "constant_output": (
            bool(scored)
            and (
                n_pred_true == len(scored)
                or n_pred_false == len(scored)
                or n_review == len(scored)
            )
        ),
        "n_interval_shortcut": n_shortcut,
        "n_silent_failure": n_silent,
        "n_shortcut_on_review": n_shortcut_on_review,
        "n_missing_bounds_evidence": n_missing_be,
        "sample": sample,
    }
    payload = {"summary": summary, "results": results}
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\nSUMMARY", json.dumps(summary, indent=2), flush=True)
    print(f"Wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
