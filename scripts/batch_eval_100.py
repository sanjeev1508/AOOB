#!/usr/bin/env python3
"""Batch-run agent on 100 labeled alarms vs human ground truth.

Includes every human-true(+low) label (only 8 exist) plus the audit-set
Orders, then fills the rest with a seeded false sample.

Use:     py -3.13 scripts/batch_eval_100.py
Resume:  py -3.13 scripts/batch_eval_100.py --resume
Preview: py -3.13 scripts/batch_eval_100.py --preview
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from aoob_agent.agent import stream_investigate
from aoob_agent.case_compiler import compile_case
from aoob_agent.data_store import DataStore
from aoob_agent.session import clear_verdict_registry

GT_PATH = ROOT / "data" / "Full_alarms_with_result.csv"
OUT = ROOT / "reports" / "batch_eval_100.json"
DETAIL = ROOT / "reports" / "batch_eval_100_detail.json"
SEED = 20260824
N_TOTAL = 100
MUST_INCLUDE = (39, 57, 1464, 2475, 2481, 2484, 6355, 6366, 7256, 7260, 7262)
MAX_RETRIES = 4
RETRY_SLEEP_S = 8.0
RATE_LIMIT_SLEEP_S = 45.0
INTER_ALARM_SLEEP_S = 8.0


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


def _is_true(label: str) -> bool:
    return str(label).lower().startswith("true")


def _is_false(label: str) -> bool:
    return str(label).lower() == "false"


def build_sample(rng: random.Random) -> list[tuple[int, str]]:
    """All human-trues + MUST_INCLUDE + seeded falses, capped at N_TOTAL."""
    gt = _load_ground_truth()
    forced: list[tuple[int, str]] = []
    seen: set[int] = set()
    for oid in MUST_INCLUDE:
        if oid not in gt:
            raise RuntimeError(f"Must-include Order {oid} has no human label")
        forced.append((oid, gt[oid]))
        seen.add(oid)
    for oid, lab in sorted(gt.items()):
        if _is_true(lab) and oid not in seen:
            forced.append((oid, lab))
            seen.add(oid)
    falses = [(oid, lab) for oid, lab in gt.items() if _is_false(lab) and oid not in seen]
    need = N_TOTAL - len(forced)
    if need < 0:
        raise RuntimeError(f"Forced set ({len(forced)}) longer than N_TOTAL={N_TOTAL}")
    if need > len(falses):
        raise RuntimeError(f"Need {need} falses but only {len(falses)} available")
    picked_false = rng.sample(falses, need)
    picked_false.sort(key=lambda t: t[0])
    return forced + picked_false


def _is_timeout(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "timed out" in msg or "timeout" in msg or "read timeout" in msg


def _is_rate_limit(exc: BaseException) -> bool:
    msg = str(exc)
    return "429" in msg or "Too Many Requests" in msg


def _is_network_blip(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(
        token in msg
        for token in (
            "connection aborted",
            "connection reset",
            "connectionreseterror",
            "nameresolutionerror",
            "getaddrinfo failed",
            "failed to resolve",
            "max retries exceeded",
            "temporarily unavailable",
        )
    )


def _is_retryable(exc: BaseException) -> bool:
    return _is_timeout(exc) or _is_rate_limit(exc) or _is_network_blip(exc)


def _load_checkpoint() -> dict[int, dict]:
    if not OUT.exists():
        return {}
    try:
        data = json.loads(OUT.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    by_id: dict[int, dict] = {}
    for row in data.get("results") or []:
        oid = row.get("order_id")
        if oid is not None:
            by_id[int(oid)] = row
    return by_id


def _should_skip_resume(prev: Optional[dict]) -> bool:
    if not prev:
        return False
    if prev.get("agent_classification") in {"true", "false", "review"} and not prev.get(
        "error"
    ):
        return True
    return False


def _summarize_trace(trace: list[dict]) -> dict:
    tools: list[str] = []
    rejected = 0
    accepted = False
    inspects: list[str] = []
    for ev in trace:
        if ev.get("type") == "tool_call":
            name = str(ev.get("name") or "")
            tools.append(name)
            if name == "inspect":
                args = ev.get("args") or {}
                fn = args.get("function_name") or args.get("helper") or ""
                if fn:
                    inspects.append(str(fn))
        elif ev.get("type") == "tool_result" and ev.get("name") == "submit_verdict":
            preview = str(ev.get("preview") or "")
            if '"accepted": false' in preview or '"accepted": false' in preview.lower():
                rejected += 1
            if '"accepted": true' in preview:
                accepted = True
    return {
        "tool_sequence": tools,
        "n_tool_calls": len(tools),
        "n_submit_rejects": rejected,
        "verdict_accepted": accepted,
        "inspected": inspects,
        "failed_to_converge": (not accepted)
        and any("failed to converge" in str(ev.get("content") or "") for ev in trace),
    }


def _case_snapshot(store: DataStore, oid: int) -> dict[str, Any]:
    case = compile_case(store, oid)
    return {
        "location": case.location,
        "alarm_function": case.alarm_function,
        "alarm_line": case.alarm_line,
        "object": case.indexed_object.name,
        "kind": case.indexed_object.kind,
        "declared_size": case.array_size,
        "operand": case.operand_symbol,
        "index_expression": case.index_expression,
        "write_helpers": list(case.write_helpers),
        "arg_read_helpers": list(case.arg_read_helpers),
        "helpers": list(case.helpers),
        "guards": list(case.guards),
        "gaps": list(case.gaps),
        "path_len": len(case.path),
        "path_functions": [s.function for s in case.path],
        "astree_message": case.astree_message,
    }


def _detail_row(row: dict) -> dict[str, Any]:
    snap = row.get("case") or {}
    ts = row.get("trace_summary") or {}
    return {
        "order_id": row.get("order_id"),
        "human": row.get("human"),
        "agent": row.get("agent_classification"),
        "match": row.get("match"),
        "outcome": _outcome(row),
        "confidence": row.get("confidence"),
        "failed_to_converge": row.get("failed_to_converge"),
        "error": row.get("error"),
        "elapsed_s": row.get("elapsed_s"),
        "attempts": row.get("attempts"),
        "location": snap.get("location"),
        "alarm_function": snap.get("alarm_function") or row.get("function_name"),
        "object": snap.get("object"),
        "kind": snap.get("kind"),
        "declared_size": snap.get("declared_size", row.get("declared_size")),
        "operand": snap.get("operand"),
        "index_expression": snap.get("index_expression") or row.get("index_expression"),
        "write_helpers": snap.get("write_helpers") or [],
        "arg_read_helpers": snap.get("arg_read_helpers") or [],
        "guards": snap.get("guards") or row.get("guards_found") or [],
        "gaps": snap.get("gaps") or [],
        "path_functions": snap.get("path_functions") or [],
        "tools_used": row.get("tools_used") or ts.get("tool_sequence") or [],
        "inspected": ts.get("inspected") or [],
        "n_submit_rejects": ts.get("n_submit_rejects"),
        "comment": row.get("comment"),
        "summary": row.get("summary"),
        "index_origin_summary": row.get("index_origin_summary"),
    }


def _recall_safe(agent_cls: str | None, human: str) -> bool:
    """True when a human-true alarm is not classified false (review is OK)."""
    if not _is_true(human):
        return True
    return (agent_cls or "") != "false"


def _token_stats(results: list[dict]) -> dict:
    savings = [
        float((r.get("report") or {}).get("token_savings_percent") or 0)
        for r in results
        if (r.get("report") or {}).get("token_savings_percent") is not None
    ]
    tokens = [
        float((r.get("report") or {}).get("tokens_used") or 0)
        for r in results
        if (r.get("report") or {}).get("tokens_used")
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
        "token_savings_percent": _stats(savings),
        "tokens_used": _stats(tokens),
    }


def _outcome(row: dict) -> str:
    if row.get("error"):
        return "error"
    agent = row.get("agent_classification")
    human = row.get("human") or ""
    if agent == "review":
        return "review" + (" (fail-converge)" if row.get("failed_to_converge") else "")
    if agent == "true" and _is_true(human):
        return "TP"
    if agent == "false" and _is_false(human):
        return "TN"
    if agent == "true" and _is_false(human):
        return "FP"
    if agent == "false" and _is_true(human):
        return "FN"
    return str(agent or "unknown")


    if row.get("error"):
        return "error"
    agent = row.get("agent_classification")
    human = row.get("human") or ""
    if agent == "review":
        return "review" + (" (fail-converge)" if row.get("failed_to_converge") else "")
    if agent == "true" and _is_true(human):
        return "TP"
    if agent == "false" and _is_false(human):
        return "TN"
    if agent == "true" and _is_false(human):
        return "FP"
    if agent == "false" and _is_true(human):
        return "FN"
    return str(agent or "unknown")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Batch eval 100 alarms vs ground truth")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--retries", type=int, default=MAX_RETRIES)
    args = parser.parse_args(argv)

    rng = random.Random(SEED)
    sample = build_sample(rng)
    n_true = sum(1 for _, h in sample if _is_true(h))
    n_false = sum(1 for _, h in sample if _is_false(h))
    print(
        f"Sample n={len(sample)} seed={SEED} true={n_true} false={n_false} "
        f"forced={list(MUST_INCLUDE)} resume={args.resume}",
        flush=True,
    )
    for oid, human in sample:
        print(f"  {oid}\t{human}", flush=True)
    if args.preview:
        return 0

    prior = _load_checkpoint() if args.resume else {}
    clear_verdict_registry()
    store = DataStore.load(ROOT / "data")
    results: list[dict] = []

    for oid, human in sample:
        prev = prior.get(oid)
        if args.resume and _should_skip_resume(prev):
            results.append(prev)
            print(
                f"RESUME-SKIP {oid} agent={prev.get('agent_classification')} "
                f"match={prev.get('match')}",
                flush=True,
            )

    for i, (oid, human) in enumerate(sample, 1):
        if args.resume and _should_skip_resume(prior.get(oid)):
            continue
        if oid not in store.alarms:
            results.append(
                {
                    "order_id": oid,
                    "human": human,
                    "error": "missing from Full_alarms.csv",
                    "match": False,
                }
            )
            print(f"SKIP {oid} missing", flush=True)
            _write(sample, results)
            continue

        print(f"\n=== [{i}/{len(sample)}] RUN Order {oid} (human={human}) ===", flush=True)
        try:
            case_snap = _case_snapshot(store, oid)
        except Exception as exc:  # noqa: BLE001
            case_snap = {"compile_error": str(exc)}
        row: dict | None = None
        attempts = max(1, int(args.retries))
        for attempt in range(1, attempts + 1):
            t0 = time.time()
            try:
                dump = None
                trace: list[dict] = []
                err_msg = None
                for event in stream_investigate(oid, store):
                    et = event.get("type")
                    if et == "tool_call":
                        trace.append(
                            {
                                "type": "tool_call",
                                "name": event.get("name"),
                                "args": event.get("args"),
                            }
                        )
                    elif et == "tool_result":
                        preview = str(event.get("preview") or "")
                        trace.append(
                            {
                                "type": "tool_result",
                                "name": event.get("name"),
                                "preview": preview[:2500],
                            }
                        )
                    elif et == "agent":
                        trace.append(
                            {"type": "agent", "content": str(event.get("content") or "")[:2000]}
                        )
                    elif et == "status":
                        msg = str(event.get("message") or "")
                        if msg.startswith("Case compiled"):
                            trace.append({"type": "case", "message": msg})
                    elif et == "report":
                        dump = event.get("data")
                    elif et == "error":
                        err_msg = str(event.get("message") or "stream error")
                if dump is None:
                    raise RuntimeError(err_msg or "Investigation finished without a report")
                elapsed = round(time.time() - t0, 1)
                agent_cls = dump.get("classification")
                match = (
                    (agent_cls in {"true", "true (low)"})
                    if _is_true(human)
                    else agent_cls == "false"
                    if _is_false(human)
                    else False
                )

                be = dump.get("bounds_evidence") or {}
                comment = str(dump.get("comment") or "")
                row = {
                    "order_id": oid,
                    "human": human,
                    "agent_classification": agent_cls,
                    "match": match,
                    "recall_safe": _recall_safe(agent_cls, human),
                    "confidence": dump.get("confidence"),
                    "comment": comment,
                    "summary": dump.get("summary"),
                    "astree_message": dump.get("astree_message"),
                    "evidence_completeness": be.get("evidence_completeness"),
                    "index_origin_summary": be.get("index_origin_summary"),
                    "declared_size": be.get("declared_size"),
                    "index_expression": be.get("index_expression"),
                    "guards_found": be.get("guards_found"),
                    "affected_symbols": dump.get("affected_symbols"),
                    "function_name": dump.get("function_name"),
                    "tools_used": dump.get("tools_used"),
                    "elapsed_s": elapsed,
                    "attempts": attempt,
                    "trace_summary": _summarize_trace(trace),
                    "failed_to_converge": "failed to converge" in comment.lower(),
                    "case": case_snap,
                    "trace": trace,
                    "report": dump,
                }
                print(
                    f"DONE {oid} agent={agent_cls} human={human} match={match} "
                    f"conf={dump.get('confidence')} {elapsed}s "
                    f"(attempt {attempt}/{attempts})",
                    flush=True,
                )
                break
            except Exception as exc:  # noqa: BLE001
                elapsed = round(time.time() - t0, 1)
                if _is_retryable(exc) and attempt < attempts:
                    wait = (
                        RATE_LIMIT_SLEEP_S
                        if (_is_rate_limit(exc) or _is_network_blip(exc))
                        else RETRY_SLEEP_S
                    )
                    wait *= attempt
                    kind = (
                        "RATE-LIMIT"
                        if _is_rate_limit(exc)
                        else "NETWORK"
                        if _is_network_blip(exc)
                        else "TIMEOUT"
                    )
                    print(
                        f"{kind} {oid} attempt {attempt}/{attempts}: {exc} "
                        f"({elapsed}s) — retrying in {wait}s",
                        flush=True,
                    )
                    time.sleep(wait)
                    continue
                row = {
                    "order_id": oid,
                    "human": human,
                    "error": str(exc),
                    "elapsed_s": elapsed,
                    "attempts": attempt,
                    "match": False,
                    "case": case_snap,
                }
                print(f"FAIL {oid}: {exc} ({elapsed}s)", flush=True)
                break

        assert row is not None
        results = [r for r in results if r.get("order_id") != oid]
        results.append(row)
        _write(sample, results)
        time.sleep(INTER_ALARM_SLEEP_S)

    _write(sample, results)
    return 0


def _write(sample: list[tuple[int, str]], results: list[dict]) -> None:
    by_id = {int(r["order_id"]): r for r in results if r.get("order_id") is not None}
    ordered = [by_id[oid] for oid, _ in sample if oid in by_id]
    for r in results:
        oid = r.get("order_id")
        if oid is not None and int(oid) not in {o for o, _ in sample}:
            ordered.append(r)

    scored = [r for r in ordered if "agent_classification" in r]
    decided = [r for r in scored if r.get("agent_classification") in {"true", "false"}]
    tp = sum(1 for r in scored if _is_true(r["human"]) and r["agent_classification"] == "true")
    tn = sum(1 for r in scored if _is_false(r["human"]) and r["agent_classification"] == "false")
    fp = sum(1 for r in scored if _is_false(r["human"]) and r["agent_classification"] == "true")
    fn = sum(1 for r in scored if _is_true(r["human"]) and r["agent_classification"] == "false")
    n_review = sum(1 for r in scored if r.get("agent_classification") == "review")
    n_converge = sum(1 for r in scored if r.get("failed_to_converge"))
    true_rows = [r for r in scored if _is_true(r.get("human") or "")]
    recall_safe_n = sum(1 for r in true_rows if r.get("recall_safe", _recall_safe(r.get("agent_classification"), r.get("human", ""))))
    summary = {
        "n_requested": N_TOTAL,
        "n_sample": len(sample),
        "n_scored": len(scored),
        "n_decided": len(decided),
        "n_review": n_review,
        "n_failed_to_converge": n_converge,
        "n_errors": sum(1 for r in ordered if r.get("error")),
        "n_done": len(ordered),
        "accuracy_all": (tp + tn) / len(scored) if scored else None,
        "accuracy_decided": (tp + tn) / len(decided) if decided else None,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "n_human_true": len(true_rows),
        "recall_safe_on_true": recall_safe_n,
        "recall_violations": [
            r["order_id"]
            for r in true_rows
            if not r.get("recall_safe", _recall_safe(r.get("agent_classification"), r.get("human", "")))
        ],
        **_token_stats(scored),
        "seed": SEED,
        "must_include": list(MUST_INCLUDE),
        "sample": sample,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps({"summary": summary, "results": ordered}, indent=2),
        encoding="utf-8",
    )
    DETAIL.write_text(
        json.dumps(
            {"summary": summary, "alarms": [_detail_row(r) for r in ordered]},
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\nCHECKPOINT SUMMARY", json.dumps(summary, indent=2), flush=True)
    print(f"Wrote {OUT}", flush=True)
    print(f"Wrote {DETAIL}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
