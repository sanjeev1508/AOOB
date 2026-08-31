---
name: aoob-alarm-investigator
description: Astrée Out-Of-Bounds (AOOB) static analysis alarm investigation skill with value-range simulation.
version: 1.0.0
---

# AOOB Alarm Investigator Skill

This skill defines the domain expert investigation protocol for Astrée Out-Of-Bounds (AOOB) static analysis array alarms.

## Core Investigation Protocol

1. **Micro-slice Inspection (`get_micro_window`)**:
   - Inspect the 30-line source slice containing the array dereference, enclosing loop bounds, and immediate `if` guards.

2. **Value-Set & Range Simulation (`run_simulation`)**:
   - Run static value-range simulation to compare index variable min/max bounds against array declared capacity (`declared_size`).
   - Evaluate variable data types (e.g. `uint8` $0..255$) or bitwise masks (e.g. `& 0x07` $0..7$) against array capacity.

3. **Helper & Declaration Inspection (`inspect` / `inspect_declaration`)**:
   - Open helper functions that modify or compute index values (`write_helpers` / index expression callers).
   - Verify literal contents of lookup tables or array initializations.

4. **Verdict Submission (`submit_verdict`)**:
   - Submit final verdict: `false`, `true (low)`, `true`, `undecided`, or `review`.
