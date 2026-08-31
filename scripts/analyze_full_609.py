#!/usr/bin/env python3
"""Analyze the full 609 alarm set — structural clusters or agent confusion matrix.

Structural (default):
  py -3.13 scripts/analyze_full_609.py

Agent pass (runs investigations, slow):
  py -3.13 scripts/analyze_full_609.py --agent [--limit N] [--resume]
"""
import argparse
import csv, json, sys, collections, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from aoob_agent.data_store import DataStore
from aoob_agent.case_compiler import compile_case

GT_PATH = ROOT / "data" / "Full_alarms_with_result.csv"
AGENT_OUT = ROOT / "reports" / "full_609_agent.json"
def _load_gt():
    text = GT_PATH.read_text(encoding="utf-8-sig")
    lines = text.splitlines()
    start = 1 if lines and lines[0].lower().startswith("sep=") else 0
    reader = csv.DictReader(lines[start:], delimiter=";")
    gt = {}
    for row in reader:
        raw = (row.get("Order") or "").replace(",", "").strip()
        try:
            oid = int(raw)
        except ValueError:
            continue
        label = (row.get("Classification") or "").strip()
        if label:
            gt[oid] = label
    return gt


def _is_true(label: str) -> bool:
    return str(label).lower().startswith("true")


def _agent_pass(store, gt: dict, *, limit: int | None, resume: bool) -> int:
    from aoob_agent.agent import stream_investigate

    prior: dict[int, dict] = {}
    if resume and AGENT_OUT.exists():
        try:
            data = json.loads(AGENT_OUT.read_text(encoding="utf-8"))
            for row in data.get("results") or []:
                oid = row.get("order_id")
                if oid is not None:
                    prior[int(oid)] = row
        except Exception:  # noqa: BLE001
            pass

    order_ids = sorted(gt.keys())
    if limit:
        order_ids = order_ids[:limit]
    results: list[dict] = list(prior.values())
    done_ids = {int(r["order_id"]) for r in results if r.get("order_id") is not None}

    for i, oid in enumerate(order_ids, 1):
        if oid in done_ids and resume:
            print(f"RESUME-SKIP {oid}", flush=True)
            continue
        if oid not in store.alarms:
            continue
        print(f"[{i}/{len(order_ids)}] Agent Order {oid} human={gt[oid]}", flush=True)
        t0 = time.time()
        dump = None
        err = None
        for event in stream_investigate(oid, store):
            if event.get("type") == "report":
                dump = event.get("data")
            elif event.get("type") == "error":
                err = event.get("message")
        elapsed = round(time.time() - t0, 1)
        agent_cls = dump.get("classification") if dump else None
        human = gt[oid]
        match = (
            agent_cls in {"true", "true (low)", "review"}
            if _is_true(human)
            else agent_cls == "false"
            if human == "false"
            else False
        )
        row = {
            "order_id": oid,
            "human": human,
            "agent_classification": agent_cls,
            "match": match,
            "error": err,
            "elapsed_s": elapsed,
            "report": dump,
        }
        results = [r for r in results if r.get("order_id") != oid]
        results.append(row)
        _write_agent_results(gt, results)
        time.sleep(2)

    _write_agent_results(gt, results)
    return 0


def _write_agent_results(gt: dict, results: list[dict]) -> None:
    scored = [r for r in results if r.get("agent_classification")]
    tp = sum(
        1
        for r in scored
        if _is_true(gt.get(int(r["order_id"]), ""))
        and r["agent_classification"] in {"true", "true (low)"}
    )
    tn = sum(
        1
        for r in scored
        if gt.get(int(r["order_id"])) == "false" and r["agent_classification"] == "false"
    )
    fp = sum(
        1
        for r in scored
        if gt.get(int(r["order_id"])) == "false" and r["agent_classification"] in {"true", "true (low)"}
    )
    fn = sum(
        1
        for r in scored
        if _is_true(gt.get(int(r["order_id"]), "")) and r["agent_classification"] == "false"
    )
    n_review = sum(1 for r in scored if r.get("agent_classification") == "review")
    savings = [
        float((r.get("report") or {}).get("token_savings_percent") or 0)
        for r in scored
        if (r.get("report") or {}).get("token_savings_percent") is not None
    ]
    summary = {
        "n_scored": len(scored),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "n_review": n_review,
        "accuracy": (tp + tn) / len(scored) if scored else None,
        "token_savings_mean": round(sum(savings) / len(savings), 4) if savings else None,
    }
    AGENT_OUT.parent.mkdir(parents=True, exist_ok=True)
    AGENT_OUT.write_text(
        json.dumps({"summary": summary, "results": sorted(results, key=lambda r: r["order_id"])}, indent=2),
        encoding="utf-8",
    )
    print("\nAGENT CONFUSION MATRIX", json.dumps(summary, indent=2), flush=True)
    print(f"Wrote {AGENT_OUT}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", action="store_true", help="Run full agent investigations")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    gt = _load_gt()
    store = DataStore.load(ROOT / "data")

    if args.agent:
        return _agent_pass(store, gt, limit=args.limit, resume=args.resume)

    return _structural_pass(store, gt)


def _structural_pass(store, gt: dict) -> int:
    """Compile all labeled alarms and emit cluster / gap statistics."""
    fn_cluster = collections.Counter()
    obj_cluster = collections.Counter()
    expr_cluster = collections.Counter()
    size_dist = collections.Counter()
    guard_count = collections.Counter()
    gap_patterns = collections.Counter()

    alarm_data = []
    errors = 0
    for oid in sorted(gt.keys()):
        if oid not in store.alarms:
            errors += 1
            continue
        try:
            case = compile_case(store, oid)
            entry = {
                "order_id": oid,
                "human": gt[oid],
                "function": case.alarm_function,
                "object": case.indexed_object.name,
                "size": case.array_size,
                "operand": case.operand_symbol,
                "index_expression": case.index_expression,
                "n_guards": len(case.guards),
                "guards": case.guards[:3],
                "gaps": list(case.gaps),
                "helpers": list(case.helpers),
                "write_helpers": list(case.write_helpers),
                "scope": case.operand_scope_at_alarm,
                "kind": case.indexed_object.kind,
                "path_len": len(case.path),
            }
            alarm_data.append(entry)
            fn_cluster[case.alarm_function] += 1
            obj_cluster[case.indexed_object.name] += 1
            expr_cluster[case.index_expression] += 1
            size_dist[case.array_size] += 1
            guard_count[len(case.guards)] += 1
            gap_tuple = tuple(sorted(case.gaps))
            gap_patterns[gap_tuple] += 1
        except Exception as exc:
            errors += 1
            print(f"  ERR Order {oid}: {exc}")

    print(f"Total labeled alarms: {len(gt)}")
    print(f"True labels: {sum(1 for v in gt.values() if v.startswith('true'))}")
    print(f"False labels: {sum(1 for v in gt.values() if v == 'false')}")
    print(f"Compiled: {len(alarm_data)}, Errors: {errors}")

    print(f"\n=== FUNCTION CLUSTERS (top 20) ===")
    for fn, count in fn_cluster.most_common(20):
        trues = [a for a in alarm_data if a["function"] == fn and a["human"].startswith("true")]
        print(f"  {fn}: {count} alarms, trues={len(trues)}")

    print(f"\n=== OBJECT CLUSTERS (top 20) ===")
    for obj, count in obj_cluster.most_common(20):
        trues = [a for a in alarm_data if a["object"] == obj and a["human"].startswith("true")]
        print(f"  {obj}: {count} alarms, trues={len(trues)}")

    print(f"\n=== INDEX EXPRESSION CLUSTERS (top 15) ===")
    for expr, count in expr_cluster.most_common(15):
        trues = [
            a for a in alarm_data if a["index_expression"] == expr and a["human"].startswith("true")
        ]
        print(f"  [{expr}]: {count} alarms, trues={len(trues)}")

    print(f"\n=== ARRAY SIZE DISTRIBUTION ===")
    for size, count in sorted(size_dist.most_common(20), key=lambda x: (x[0] is None, x[0] or 0)):
        trues = [a for a in alarm_data if a["size"] == size and a["human"].startswith("true")]
        print(f"  size={size}: {count} alarms, trues={len(trues)}")

    print(f"\n=== GAP PATTERNS ===")
    for gaps, count in gap_patterns.most_common(10):
        trues = [
            a
            for a in alarm_data
            if tuple(sorted(a["gaps"])) == gaps and a["human"].startswith("true")
        ]
        print(f"  {gaps or '(none)'}: {count} alarms, trues={len(trues)}")

    print(f"\n=== GUARD COUNT DISTRIBUTION ===")
    for n, count in sorted(guard_count.items()):
        trues = [a for a in alarm_data if a["n_guards"] == n and a["human"].startswith("true")]
        print(f"  {n} guards: {count} alarms, trues={len(trues)}")

    print(f"\n=== TRUE ALARM DETAILS ===")
    for a in alarm_data:
        if a["human"].startswith("true"):
            print(
                f"  Order {a['order_id']} ({a['human']}): fn={a['function']}, "
                f"obj={a['object']}, size={a['size']}, expr={a['index_expression']}, "
                f"scope={a['scope']}, guards={a['n_guards']}, gaps={a['gaps']}, "
                f"helpers={a['helpers']}, write_helpers={a['write_helpers']}"
            )

    output_path = ROOT / "reports" / "full_609_analysis.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(alarm_data, indent=2), encoding="utf-8")
    print(f"\nWrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
