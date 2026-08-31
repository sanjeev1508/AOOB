"""Analyze batch eval results in detail."""
import json, sys
from pathlib import Path

d = json.load(open(Path(__file__).resolve().parents[1] / "reports" / "batch_eval_100_detail.json"))
alarms = d["alarms"]

# Convergence failures
fail_conv = [a for a in alarms if (a.get("outcome") or "").startswith("review")]
print(f"=== FAIL-TO-CONVERGE ({len(fail_conv)}) ===")
for a in fail_conv[:8]:
    oid = a["order_id"]
    tools = a.get("tools_used", [])
    size = a.get("declared_size")
    guards = len(a.get("guards", []))
    gaps = a.get("gaps", [])
    print(f"  Order {oid}: size={size}, guards={guards}, gaps={gaps}, tools={tools}")

# Errors breakdown
errors = [a for a in alarms if a.get("outcome") == "error"]
print(f"\n=== ERRORS ({len(errors)}) ===")
error_types = {}
for a in errors:
    err = str(a.get("error", ""))[:80]
    error_types[err] = error_types.get(err, 0) + 1
for k, v in error_types.items():
    print(f"  {v}x: {k}")

# TN analysis
tns = [a for a in alarms if a.get("outcome") == "TN"]
print(f"\n=== TNs ({len(tns)}) - Average elapsed: {sum(a['elapsed_s'] for a in tns)/len(tns):.1f}s ===")

# FP analysis (agent said true, human said false)
fps = [a for a in alarms if a.get("outcome") == "FP"]
print(f"\n=== FPs ({len(fps)}) ===")
for a in fps:
    print(f"  Order {a['order_id']}: size={a.get('declared_size')}, guards={a.get('guards')}")
    print(f"    Comment: {a.get('comment')}")

# FN analysis (agent said false, human said true/true(low))
fns = [a for a in alarms if a.get("outcome") == "FN"]
print(f"\n=== FNs ({len(fns)}) ===")
for a in fns:
    print(f"  Order {a['order_id']}: human={a['human']}, size={a.get('declared_size')}, guards={a.get('guards')}")
    print(f"    Comment: {a.get('comment')}")

# true(low) outcomes
tlows = [a for a in alarms if a.get("outcome") == "true (low)"]
print(f"\n=== true(low) outcomes ({len(tlows)}) ===")
for a in tlows:
    print(f"  Order {a['order_id']}: human={a['human']}, size={a.get('declared_size')}")
    print(f"    Comment: {a.get('comment')}")

# What percentage of alarms have guards?
with_guards = sum(1 for a in alarms if a.get("guards"))
print(f"\n=== STRUCTURAL STATS ===")
print(f"  Alarms with guards: {with_guards}/{len(alarms)}")
print(f"  Alarms with gaps: {sum(1 for a in alarms if a.get('gaps'))}/{len(alarms)}")
sizes = [a.get("declared_size") for a in alarms if a.get("declared_size") is not None]
print(f"  Known declared_size: {len(sizes)}/{len(alarms)}, min={min(sizes)}, max={max(sizes)}, median={sorted(sizes)[len(sizes)//2]}")

# Distribution of sizes across outcomes
for outcome in ["TN", "FP", "FN", "true (low)", "error", "review (fail-converge)"]:
    subset = [a for a in alarms if a.get("outcome") == outcome]
    s = [a.get("declared_size") for a in subset if a.get("declared_size") is not None]
    if s:
        print(f"  {outcome}: sizes={sorted(set(s))}")
