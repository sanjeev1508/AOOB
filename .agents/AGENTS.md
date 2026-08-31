# AOOB Agent Customization Rules & Guidelines

---
config:
  agent_name: aoob-alarm-investigator
  version: 2.0.0
  frontmatter:
    capabilities:
      - adaptive_micro_windowing
      - raw_bounds_fact_gathering
      - agentic_stopping_condition
      - webgl_graph_streaming
---

## Behavioral Rules

1. **Recall-First Objective**: Aim for zero false negatives on all static analysis alarms. A `review` verdict is always acceptable; a `false` on a known true alarm is not.

2. **Evidence-Driven Context**: Prefer `get_micro_window(window_size=15)` first; widen adaptively when `window_truncated` is true. Do not mandate a fixed 30-line window or rigid tool order.

3. **Agentic Classification**: `run_simulation()` returns mechanical bounds facts only. The agent must reason about risk and cite evidence in `submit_verdict` — no static lookup table may decide the classification.

4. **Agentic Stopping**: Continue until `confidence_so_far=high` (with at least one `run_simulation` call) or submit `review` with `reason_for_review`. Safety ceiling (20 tool calls) forces `review`, never a guessed false/true.

5. **Ground Truth Isolation**: Never load `Full_alarms_with_result.csv` during investigation — evaluation harnesses only.
