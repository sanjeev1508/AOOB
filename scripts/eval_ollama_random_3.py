#!/usr/bin/env python3
"""Classify 1 true + 2 false alarms with the local Ollama model and score vs GT."""

from __future__ import annotations

import csv
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aoob_agent.agent import investigate_alarm, llm_backend_label, load_env
from aoob_agent.data_store import DataStore

GT_PATH = ROOT / "data" / "Full_alarms_with_result.csv"
OUT = ROOT / "reports" / "ollama_eval_3.json"


def _load_ground_truth() -> dict[int, str]:
    text = GT_PATH.read_text(encoding="utf-8-sig")
    lines = text.splitlines()
    header_idx = next(i for i, line in enumerate(lines) if line.startswith("Order;"))
    reader = csv.DictReader(lines[header_idx:], delimiter=";")
    labels: dict[int, str] = {}
    for row in reader:
        oid = DataStore._parse_order_id(row.get("Order") or "")
        label = (row.get("Classification") or "").strip()
        if oid is not None and label:
            labels[oid] = label
    return labels


def _pick_sample(labels: dict[int, str], seed: int) -> list[tuple[int, str]]:
    rng = random.Random(seed)
    positives = [(oid, lab) for oid, lab in labels.items() if lab.startswith("true")]
    negatives = [(oid, lab) for oid, lab in labels.items() if lab == "false"]
    if not positives or len(negatives) < 2:
        raise RuntimeError("Not enough labeled alarms to sample 1 true + 2 false")
    true_one = rng.choice(positives)
    false_two = rng.sample(negatives, 2)
    return [true_one, *false_two]


def main() -> int:
    load_env(ROOT)
    seed = int(time.time())
    labels = _load_ground_truth()
    sample = _pick_sample(labels, seed)
    print(
        f"backend={llm_backend_label()} seed={seed} sample={sample}",
        flush=True,
    )

    print("Loading data store ...", flush=True)
    store = DataStore.load(ROOT / "data")
    results = []
    for oid, human in sample:
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
            row = {
                "order_id": oid,
                "human": human,
                "agent_classification": agent_cls,
                "match": match,
                "confidence": dump.get("confidence"),
                "comment": dump.get("comment"),
                "astree_message": dump.get("astree_message"),
                "evidence_completeness": (dump.get("bounds_evidence") or {}).get(
                    "evidence_completeness"
                ),
                "index_origin_summary": (dump.get("bounds_evidence") or {}).get(
                    "index_origin_summary"
                ),
                "tools_used": dump.get("tools_used"),
                "elapsed_s": elapsed,
                "report": dump,
            }
            results.append(row)
            print(
                f"DONE {oid} agent={agent_cls} human={human} match={match} "
                f"conf={dump.get('confidence')} {elapsed}s",
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

        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(
            json.dumps({"seed": seed, "results": results}, indent=2),
            encoding="utf-8",
        )

    scored = [r for r in results if "agent_classification" in r]
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
    summary = {
        "backend": llm_backend_label(),
        "seed": seed,
        "n_scored": len(scored),
        "n_errors": sum(1 for r in results if r.get("error")),
        "accuracy": (tp + tn) / len(scored) if scored else None,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "sample": sample,
    }
    payload = {"summary": summary, "results": results}
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\nSUMMARY", json.dumps(summary, indent=2), flush=True)
    print(f"Wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
