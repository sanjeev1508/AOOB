---
name: aoob-alarm-investigator
description: Astrée Out-Of-Bounds (AOOB) static analysis alarm investigation skill with value-range simulation.
---

# AOOB Alarm Investigator Skill

This skill defines the domain expert investigation protocol for Astrée Out-Of-Bounds (AOOB) static analysis array alarms.

## Evidence Required (not a fixed tool order)

Before submitting a verdict, gather enough evidence to judge whether the index can exceed the indexed object's declared capacity at the alarm site:

1. **Source context** — A compact slice (`get_micro_window`, default 15 lines) or full function (`get_window`) showing the dereference, loops, and guards. Widen when `window_truncated` is true.

2. **Mechanical bounds facts** — `run_simulation()` returns raw facts (type bit-width, mask literal, derived min/max, guard text). **You** compare derived max vs `declared_array_capacity` and decide the classification. The tool never recommends false/true/review.

3. **Path gaps** — If the case file lists `gaps`, call `query_call_graph()` and/or `inspect()` on write_helpers before any non-`review` verdict. `missing_df` means the index origin is untraced — do not submit `false` without addressing gaps.

4. **Alarm-line guards** — When an if-condition dereferences `array[index]`, that condition may be the access Astrée flagged; the else branch alone is not proof of safety.

5. **Declarations & helpers** — Open `inspect_declaration()` or `inspect()` when capacity, sentinel slots, or write_helpers affect the index range.

## Confidence & stopping

- Pass `confidence_so_far` (low/medium/high) on **every** tool call.
- Investigate until `confidence_so_far=high` and `run_simulation()` has been called, then submit.
- If evidence stays incomplete, submit `review` with `reason_for_review` — never guess.
- At `confidence=medium` submit, provide `counterargument` (strongest opposite case) in a self-verification round.

## Verdict taxonomy

- `false` — Runtime-bounded (loop invariant, clamp, effective guard, tool over-approximation).
- `true (low)` — Potential OOB with low reachability.
- `true` — Clear OOB without effective guard.
- `undecided` — Needs dynamic/inter-procedural context.
- `review` — Incomplete evidence or safety ceiling; mandatory `reason_for_review`.

## Human tag patterns (optional)

- `'ALG TOOL: A1968 DELTA:'` — tool over-approximation on multi-core lookup arrays
- `'e'` — enclosing loop / handle invariant
- `'h'` — high-risk unguarded access
