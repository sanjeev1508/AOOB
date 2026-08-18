# AOOB-Graph

Investigate Astrée **out-of-bounds (AOOB)** alarms over shared CSV / source exports with:

1. **Primary UI** (`viz_main.py --serve`) — left **streaming chatbot**, center **WebGL CF graph**, right **DF sequence** panel.  
   Enter an Order id; the agent streams status, tool calls/args, tool results, reasoning, and the final report while the graph highlights the DF path.

2. **CLI agent** (`main.py`) — same LangGraph investigation with **Ollama (local/network) or NVIDIA fallback**, writing a JSON report to stdout or a file (no UI).

3. **Batch eval** (`scripts/batch_eval_10.py`) — fixed 10-alarm sample scored against human labels in `Full_alarms_with_result.csv`.

**Design rule:** tools are **retrieval-only**. They never hardcode triage. The model authors `classification` (`true` / `false` / `review`), `comment`, and `confidence`. Human CSV Classification/Comment columns are **not** fed to the agent (prepare `Full_alarms.csv` with `data/convert.py` so those fields are empty).

---

## Table of contents

- [Features](#features)
- [Repository layout](#repository-layout)
- [Inputs](#inputs-required)
- [Requirements](#requirements)
- [Setup](#setup)
- [Quick start](#quick-start)
- [Pipeline A — Agent investigation](#pipeline-a--agent-investigation)
- [Pipeline B — CF visualization + chat UI](#pipeline-b--cf-visualization--chat-ui)
- [Batch evaluation](#batch-evaluation)
- [Shared data layer](#shared-data-layer)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Dependencies](#dependencies)
- [License](#license)

---

## Features

### Agent pipeline

- LangGraph loop: **agent → tools → (optional index-origin nudge) → structured report**
- Streaming events for the UI: `status`, `tool_call`, `tool_result`, `agent` reasoning, `report`, `error`, `done`
- Backend chooser: `AOOB_LLM_CHOOSED=local|network|nvidia` with runtime failover from Ollama to NVIDIA on timeout/connection errors (when NVIDIA key is present)
- **Single-model orchestration by default**: one shared model instance is used for both tool-calling/planning and report synthesis during the same investigation
- Five retrieval-only tools (variable scope, declaration info, trimmed sequence,
  function snippet, caller context)
- Robust tool-call parsing fallback: recovers tool calls from markdown-polluted JSON or XML-like `<tool_call>...</tool_call>` output when local models fail strict JSON function-calling
- Report synthesis resilience: reuses JSON from the latest no-tool reasoning turn when possible; if model output is empty/unparseable, emits a schema-valid low-confidence `review` fallback instead of crashing the stream
- Pydantic report with agent-authored:
  - `classification` (`true` = real bug, `false` = false positive, `review` = inconclusive)
  - `comment`, `confidence`, `astree_message`, optional `bounds_evidence`
- Astrée `Message` (abstract intervals) is passed through tools; human Classification/Comment are not
- **Process gates** (not verdict hardcoding): nudge until a useful **index-operand** slice exists; detect no-progress origin loops and consecutive no-tool-call deadlocks, then finalize with downgraded completeness/confidence instead of cycling forever
- Soft tool-message ceiling: **12**
- CLI (`main.py`) with optional JSON file output
- Order ids with thousand separators supported (`1,112` → `1112`)
- Prompt discipline: index-expression arithmetic before Astrée interval compare; sentinel/invalid constants need guards; unresolved evidence prefers uncertainty over the raw interval
### Chat + CF visualization UI

- One-time **full CF graph** build (~9.8k nodes / ~18k edges) with **server-side layout**
- Fast **sigma.js WebGL** UI (no browser force-layout freeze)
- **Left:** streaming chat — Order id → live tool calls, evidence previews, reasoning, report
- **Center:** full CF graph; auto-highlights DF sequence when you investigate
- **Right:** first access, following sequence timeline, alarms on path
- Sequence modes in UI:
  - **Writes + alarm read** (default): write/mixed nodes + queried alarm read node only
  - **Full sequence**: include intermediate read/bridge context nodes
- **Read** / **write** / **mixed** color coding (graph + timeline)
- Labels: **all function names** when not highlighting; **sequence-only labels** while highlighting
- Edges hidden by default; optional “show all CF edges”
- Camera: **Fit** (full graph) and **Focus sequence** (framed viewport fit to highlighted nodes)
- SSE heartbeats (~12s) while waiting on LLM calls
- REST + SSE: `/api/graph`, `/api/highlight`, `/api/investigate/stream`, `/api/health`, `/api/alarms`

---

## Repository layout

```text
AOOB-Graph/
├── main.py                     # Agent CLI (Order id → JSON report)
├── viz_main.py                 # Build full CF graph / serve chat+graph UI
├── requirements.txt
├── .env                        # LLM backend selection + credentials (do not commit)
├── .gitignore
├── README.md
│
├── aoob_agent/                 # Agent + shared Astrée data loader
│   ├── __init__.py
│   ├── agent.py                # LangGraph, gates/nudge, streaming investigate
│   ├── tools.py                # Retrieval tools (+ index operand helpers)
│   ├── data_store.py           # Indexes alarms / DF / CF / input.c
│   └── report.py               # AlarmInvestigationReport schema
│
├── cf_viz/                     # Control-flow visualization + chat UI
│   ├── __init__.py
│   ├── server.py               # FastAPI (UI + highlight + SSE investigate)
│   ├── export.py               # CF → laid-out JSON (v2 / sigma-webgl)
│   ├── graph_builder.py        # NetworkX CF graph + DF path helpers
│   ├── highlight.py            # Order id → DF sequence + alarms payload
│   └── static/
│       └── index.html          # Chat + WebGL graph + sequence panel
│
├── data/                       # Required Astrée inputs
│   ├── Full_alarms.csv         # Agent input (Message kept; Classification/Comment empty)
│   ├── Full_alarms_with_result.csv  # Optional human ground truth for eval
│   ├── convert.py              # Build Full_alarms.csv from *_with_result.csv
│   ├── data flow.csv
│   ├── control flow.csv
│   └── input.c
│
├── scripts/
│   └── batch_eval_10.py        # Fixed 10-alarm sample vs human labels
│
├── tests/
│   ├── test_evidence_tools.py  # Tool / index-operand acceptance checks
│   └── test_agent_gates.py     # Agent gate/fallback/prompt-flow checks
│
└── reports/                    # Generated outputs
    ├── cf_full_graph.json      # Complete CF graph artifact
    ├── batch_eval_10.json      # Latest batch-eval results (when run)
    └── alarm_*.json            # Optional per-alarm CLI reports
```

---

## Inputs (required)

Place these files under `data/` (or pass `--data-dir`):

| File | Role |
|------|------|
| `Full_alarms.csv` | Alarm Order, type, category, location, Astrée `Message` (intervals). **Leave Classification/Comment empty** — the agent authors those for the report |
| `data flow.csv` | Variable ↔ function read/write + location |
| `control flow.csv` | Caller → callee call sites |
| `input.c` | Preprocessed C source (line-aligned with Astrée locations) |

### Optional / eval inputs

| File | Role |
|------|------|
| `Full_alarms_with_result.csv` | Same alarms plus human Classification / Comment (ground truth for batch eval). **Not** loaded by the agent |
| `data/convert.py` | Copies `*_with_result.csv` → `Full_alarms.csv`: keeps Message (strips `ALARM (A) …:` prefix), clears Classification/Comment |

```powershell
py -3.13 data\convert.py
```

### Location format

```text
ALL_bc_with_context.c:331731.56-85
```

Meaning: `file:line.col_start-col_end`.

### Order ids

Astrée may export thousand separators (`1,112`). The loader strips commas → integer `1112`. Both `1112` and `1,112` work on the CLIs and in the UI.

CSV files use `;` separators and may start with an Astrée `sep=;` preamble; the loader skips that automatically.

### What the agent sees vs what humans label

| Field | Agent tools | Report |
|-------|-------------|--------|
| Astrée `Message` (abstract intervals) | Yes (`alarm.message` / `astree_message`) | Copied into `astree_message` |
| Human Classification / Comment | **No** (should be empty in `Full_alarms.csv`) | Agent writes `classification` + `comment` |

---

## Requirements

- **Python 3.11+** (developed/tested with **Python 3.13**)
- Prefer `py -3.13` on Windows if multiple interpreters exist (plain `python` may be 3.14 without packages)
- For the **agent**: one of these
  - local Ollama reachable at `OLLAMA_LOCAL_BASE_URL` (default `http://127.0.0.1:11434`)
  - network Ollama reachable at `OLLAMA_NETWORK_BASE_URL`
  - NVIDIA key from [build.nvidia.com](https://build.nvidia.com/settings/api-keys) (used directly or as fallback)
- For the **CF UI**: modern browser with WebGL; network once for CDN assets (sigma.js / graphology / fonts)
- Enough RAM to hold large CSVs + `input.c` in memory (~hundreds of MB)

---

## Setup

```powershell
cd f:\AOOB-Graph

# Prefer an explicit interpreter if several Pythons are installed
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
py -3.13 -m pip install -r requirements.txt
```

Create `.env` in the project root (agent / streaming UI):

```env
# Preferred source: local | network | nvidia
AOOB_LLM_CHOOSED=local

# Local Ollama
# Single-model mode: same model for tool planning + report synthesis
OLLAMA_LOCAL_MODEL=qwen3:14b
OLLAMA_LOCAL_BASE_URL=http://127.0.0.1:11434

# Performance tuning
OLLAMA_NUM_CTX=4096
AOOB_MAX_VISIBLE_TOOLS=3

# Network Ollama
OLLAMA_NETWORK_MODEL=gemma4:26b
OLLAMA_NETWORK_BASE_URL=http://your-ollama-host:11434

# Runtime behavior
# Default in code is 300; lower this for faster fallback in practice.
OLLAMA_TIMEOUT=180
AOOB_OLLAMA_RUNTIME_FALLBACK=1

# NVIDIA fallback / direct backend
NVIDIA_API_KEY=nvapi-...
# Optional override (default: meta/llama-3.1-70b-instruct)
# NVIDIA_MODEL=meta/llama-3.1-70b-instruct

# Legacy split-model overrides are intentionally not used in single-model mode:
# AOOB_TOOL_LOCAL_MODEL=
# AOOB_TOOL_NETWORK_MODEL=
# AOOB_TOOL_MODEL=
```

Do **not** commit `.env`.

---

## Quick start

### Primary: chat + graph UI

```powershell
# 1) Build complete CF graph once (skip if reports\cf_full_graph.json already exists)
py -3.13 viz_main.py --build-full -o reports\cf_full_graph.json

# 2) Serve UI (needs reachable Ollama for chosen mode, or NVIDIA_API_KEY)
py -3.13 viz_main.py --serve --host 127.0.0.1 --port 8765
```

Open [http://127.0.0.1:8765/](http://127.0.0.1:8765/), type Order `39`, press **Send**.  
The left chat streams tool calls and the report; the center graph and right timeline update together.

After UI/code changes: hard-refresh (`Ctrl+F5`). Restart `--serve` after Python API changes.

### Optional: CLI-only agent report

```powershell
py -3.13 main.py 39 -o reports\alarm_39.json
```

### Optional: tool unit checks

```powershell
py -3.13 tests\test_evidence_tools.py
```

---

## Pipeline A — Agent investigation

### Run

```powershell
# Print JSON to stdout
py -3.13 main.py 39

# Write report to disk
py -3.13 main.py 39 -o reports\alarm_39.json

# Thousand-separated Order id
py -3.13 main.py 1,112 -o reports\alarm_1112.json

# Custom data dir / model
py -3.13 main.py 39 --data-dir .\data --model meta/llama-3.1-70b-instruct
```

Progress logs go to **stderr**; report JSON goes to **stdout**.

### CLI options

| Argument | Description |
|----------|-------------|
| `order_id` | Alarm Order from `Full_alarms.csv` (commas allowed) |
| `--data-dir` | Input directory (default `./data`) |
| `--model` | Model override (Ollama tag in ollama mode, NVIDIA model id in nvidia mode) |
| `-o` / `--output` | Optional path for the JSON report |

### How the agent works

```text
┌─────────────┐     tool calls      ┌──────────────┐
│  agent node │ ──────────────────► │  tools node  │
│   (LLM)     │ ◄────────────────── │ (retrieval)  │
└──────┬──────┘   tool results      └──────────────┘
       │
       │ no useful index-origin slice yet?
       ▼
┌─────────────┐
│ nudge node  │  (process gate — retrieval reminder only)
└──────┬──────┘
       │
       │ enough evidence / tool ceiling
       ▼
┌─────────────┐
│ report node │  structured AlarmInvestigationReport
└─────────────┘
```

1. Load and index the four input files via `DataStore`.
2. The LLM chooses which tools to call (often several in one step).
3. Tools return **raw retrieval JSON** only.
4. **Index-origin gate (nudge):** for OOB-style work, the agent must call `get_function_snippet` and use `index_operands_at_alarm`, then run `get_trimmed_sequence` on that operand. If `origin_resolved=true`, no caller hop is required.
5. **Hard next-hop gate (Fix #2):** caller hopping is only for unresolved parameter origins (`origin_resolved=false` and operand scope `parameter`). Non-parameter unresolved cases are not forced through `get_caller_context`.
6. **Dynamic active nudge:** when the model reasons without tool calls, nudge prompts inject explicit next-tool templates (JSON/XML formats) with concrete arguments.
7. **Malformed-output recovery:** if the model omits native tool-calls but emits JSON/XML tool text, parser fallback converts it into executable tool calls.
8. **Consecutive no-tool-call guard:** if the model emits no tool calls in two consecutive agent turns, routing bypasses further nudges and proceeds directly to report generation.
9. **Loop-stall guard:** repeated identical tool-result signatures (including short repeated cycles) force route to report instead of endless retries.
10. Soft tool-message ceiling **12**, then force the report node.
11. Final LLM turn builds `AlarmInvestigationReport`.
12. **Forced re-prompt on fully-traced hedge (Fix #3):** if draft classification is `review` while evidence is `fully traced` and tool outputs include bound/guard/transform evidence, the model is re-prompted once to author `true`/`false`. Python still does not assign verdicts.
13. Soft post-process only: if no useful index slice, may downgrade `confidence` `high→medium` and `evidence_completeness` `fully traced→not traced`. Stalled-origin traces are also downgraded. Classification is never hardcoded in Python.

### Orchestration rules (strict)

- AOOB-only planner discipline: no generic architecture/tutorial prose during tool-planning turns.
- Planner turn output contract: emit tool calls when needed; otherwise emit exactly `{"type":"no_tool_call"}`.
- `{"type":"no_tool_call"}` control markers are not reused as candidate report JSON.
- If final report payload validation fails, the pipeline emits a guaranteed schema-valid degraded `review` report instead of raising runtime errors.

### Investigation start context

At stream start, the initial human message now includes a compact sequence map:

- Alarm site (read node)
- Primary site variable
- Write-node list for that variable (deduplicated by function/line)

The agent is expected to identify the index operand first, then navigate from alarm read node to write nodes (and call-sites/guards as needed) before final classification.

### Prompts (important semantics)

- Astrée `[lo, hi]` messages are **abstract domain** bounds, not concrete runtime indices.
- Forbidden over-reading: “`array_size < Astrée hi` ⇒ classification `true`”.
- Optimistic bounds rule: if guards or traced value constraints prove the index is inside the safe range at runtime, classify as `false` (an analyzer false positive).
- Fallback rule: for missing bounds context (especially dynamic pointer capacity/allocation), prefer `review` with medium confidence and document missing evidence in `bounds_evidence.index_origin_summary`.
- Prefer slicing **index operands** (e.g. `checkword`, `idxMRPChnl_u16`), not the array/table name and not sentinel end-pointers.
- `classification` / `comment` / `confidence` are always agent-authored from tool evidence.

### Tools

All tools are **retrieval-only**. They never emit bug/no-bug, severity, or confidence.

#### 1. `get_variable_scope`

Purpose: identify whether a symbol is `local`, `global`, `parameter`, `dynamic_heap`, or `unknown`.

Input: `symbol` (+ optional `alarm_order_id` for active context).

Output: scope facts, declaring function (when local/parameter), source location, and parse note when unresolved.

Member-path behavior: for symbols like `obj.field` or `ptr->field`, scope is resolved from the base symbol (`obj` / `ptr`) while preserving the full symbol in output context.

#### 2. `get_declaration_info`

Purpose: declaration lookup successor to old bounds lookup.

Input: `symbol`.

Output: `declared_type`, `kind`, `array_size` (literal only), `is_pointer`, `location`, `parse_note`, plus nested array/member fields when applicable.

Precision note: when the queried symbol is a struct/union object, the tool now attempts member-level fallback discovery (for example `obj.raw`) from declaration keys and indexed data-flow keys if typedef member parsing is incomplete.

#### 3. `get_trimmed_sequence`

Purpose: scope-aware data-flow ordered sequence for the traced variable, trimmed to write/mixed steps plus the queried alarm step.

Input: optional `alarm_order_id`, optional `variable`.

Implementation note: this wraps the shared DF sequence builder used by `cf_viz/highlight.py` (no duplicate sequence-walking logic in tools).

Deterministic trim rule:
1. Build the full cross-function DF timeline.
2. Keep steps with `access=write` (or `mixed`).
3. Keep the queried alarm step.
4. Drop unrelated pre-alarm reads.

Scope behavior:
- `scope=global`: search stays cross-function.
- `scope=local` or `scope=parameter`: DF retrieval is restricted to the current trace function (alarm function, or caller-hopped function after `get_caller_context`) to avoid bare-name collisions across unrelated functions.

Output includes `sequence`, `writes`, `reads`, `alarms_on_path`, `truncated`, `origin_resolved`, plus `scope_mode` (`global` or `function`) and `trace_function`.

`origin_resolved` semantics:
- `true`: first step already includes the queried alarm as a write/mixed step (self-contained origin at site; no caller hop needed).
- `false`: origin remains open (for parameters, caller context may be required).

#### 4. `get_function_snippet`

Purpose: return source snippet for the current trace function context (alarm function or caller-hopped function).

Input: none required (optional `alarm_order_id`, `context_lines`).

Output: function name/span/snippet, compact caller/callee lists, and `index_operands_at_alarm` extracted from alarm-line patterns.

Index parsing covers:
- Bracket index forms (including nested)
- Struct/member operands inside index expressions
- Pointer arithmetic dereference forms (for example `*(ptr + off)`)
- Multi-variable arithmetic expressions (for example `idx_a + idx_b`)

Each `index_operands_at_alarm` entry now also includes:
- `operand_symbol_candidates`: best-effort parsed symbol candidates from the operand text (including member forms like `a.b` / `a->b`).
- `resolved_operand_symbol`: the top candidate used as the recommended trace symbol.

#### 5. `get_caller_context`

Purpose: parameter-origin fallback only. This is no longer the default next step after every sequence call.

Input: required `parameter` (typically the operand from `index_operands_at_alarm`), optional `alarm_order_id`.

Output: current function, caller list with call-site locations and argument expressions, plus `unresolved` when no caller edge is available.

When to call it: only when `get_trimmed_sequence` reports `origin_resolved=false` for the traced operand and that operand scope is `parameter`.

Trade-off note: the 5-tool set intentionally removes dedicated guard/constant/index-expression tools. Guard/sentinel/shift evidence must be read directly from `get_function_snippet` output.

### Report schema

Defined in `aoob_agent/report.py`. Example shape:

```json
{
  "alarm_order_id": 39,
  "alarm_type": "Alarm (A)",
  "alarm_category": "Out-of-bound array access",
  "location": "ALL_bc_with_context.c:331731.56-85",
  "astree_message": "[0, 7] not included in array index range [0, 1]",
  "affected_symbols": [
    {
      "name": "MOCMEM_VBLSTRUCTEARLYCLEAREDRAM_MO_INST_LOOKUP",
      "kind": "array",
      "evidence": "ALL_bc_with_context.c:331731.8-87"
    }
  ],
  "function_name": "MoCMemStrtUp_Co_Hndlr",
  "function_snippet": "while ((MOCMEM_...LOOKUP[Mo_Inst_GetCoreIdx(checkword)]) != ...)",
  "manipulation_sequence": [
    {
      "order": 1,
      "function": "MoCMemStrtUp_Co_Hndlr",
      "access": "read",
      "variable": "MOCMEM_VBLSTRUCTEARLYCLEAREDRAM_MO_INST_LOOKUP",
      "location": "ALL_bc_with_context.c:331731.8-87",
      "note": ""
    }
  ],
  "summary": "…",
  "classification": "review",
  "comment": "Index operand checkword initialized to 0; Astrée interval is abstract…",
  "confidence": "medium",
  "bounds_evidence": {
    "symbol": "MOCMEM_VBLSTRUCTEARLYCLEAREDRAM_MO_INST_LOOKUP",
    "declared_size": 2,
    "index_expression": "Mo_Inst_GetCoreIdx(checkword)",
    "index_inferred_range": "[0, 1]",
    "index_safe_range": "[0, 1]",
    "target_capacity": 2,
    "guards_found": [
      "if (checkword < 2u)"
    ],
    "index_origin_summary": "checkword initialized to 0 in the same function…",
    "evidence_completeness": "partially traced"
  },
  "tools_used": [
    "get_variable_scope",
    "get_function_snippet",
    "get_declaration_info",
    "get_trimmed_sequence",
    "get_caller_context"
  ]
}
```

| Field | Notes |
|-------|--------|
| `astree_message` | Copy of Astrée Message from tools — not a human verdict |
| `classification` | `true` \| `false` \| `review` — **agent-authored**; never read from CSV |
| `comment` | Agent rationale — not CSV Comment |
| `affected_symbols[].kind` | `variable` \| `array` \| `pointer` \| `struct` \| `field` \| `other` \| `unknown` |
| `confidence` | `low` \| `medium` \| `high` — LLM-chosen; may be soft-capped by process gates |
| `bounds_evidence.evidence_completeness` | `fully traced` \| `partially traced` \| `not traced` |
| `bounds_evidence.index_expression` | Index expression tracked at alarm site (string) |
| `bounds_evidence.index_inferred_range` | Agent-inferred runtime index range (string) |
| `bounds_evidence.index_safe_range` | Derived safe range based on capacity/guards (string) |
| `bounds_evidence.target_capacity` | Capacity of indexed target (`int` or `"Dynamic"`) |
| `bounds_evidence.guards_found` | Guard statements cited from source snippets |

Acceptance checks for tools / index operands: `py -3.13 tests\test_evidence_tools.py`.

### Programmatic use

```python
from pathlib import Path
from aoob_agent.agent import investigate_alarm, stream_investigate, load_env
from aoob_agent.data_store import DataStore

root = Path(r"f:\AOOB-Graph")
load_env(root)
store = DataStore.load(root / "data")

# Blocking report
report = investigate_alarm(39, store)
print(report.model_dump_json(indent=2))

# Streaming events (same as the UI SSE path)
for event in stream_investigate(39, store):
    print(event.get("type"), event)
```

### Agent notes

- First load of data can take ~15–20s (`input.c` is large).
- Tool payloads are truncated (~3500 chars) for NIM context / latency.
- LLM timeout is backend/env-controlled (`OLLAMA_TIMEOUT`, `NVIDIA_TIMEOUT`); UI sends heartbeats while waiting.
- Heartbeats/status show the active shared model/backend in single-model mode.
- If Ollama call fails at runtime and fallback is enabled, the same investigation can switch to NVIDIA in-process.
- After enough tool messages, the graph moves to the report node to avoid long multi-turn timeouts.
- If report JSON parsing/validation fails, the pipeline degrades to a valid low-confidence `review` report rather than aborting.
- Report quality depends on model choice and Astrée export completeness.
- `input.c` line numbers must align with alarm / DF / CF locations.

---

## Pipeline B — CF visualization + chat UI

Same `data/` inputs; shared `DataStore` loader. The UI runs the **same agent** as the CLI via SSE.

### Workflow

1. **Build** the complete control-flow graph once from `control flow.csv`
2. **Serve** the WebGL + chat UI
3. Enter an alarm **Order id** and Send
4. Backend streams agent events; also resolves DF variable(s) → ordered access sequence → CF bridges
5. UI highlights that sequence on the full graph and fills the **right sidebar**

### Build the complete graph

```powershell
py -3.13 viz_main.py --build-full -o reports\cf_full_graph.json
```

Writes a **version 2** artifact (`renderer: sigma-webgl`) with:

- Compact nodes: `{ id, x, y, s }` (precomputed layout + degree-based size)
- Compact edges: `{ id, s, t, w }` (`from->to`, call count)
- Stats: node/edge counts, weakly connected components

Typical size for this project: **~9821 nodes / ~18809 edges**.

If you run `--serve` and the JSON is missing or not v2, the server rebuilds it automatically.

### Serve the UI

```powershell
py -3.13 viz_main.py --serve --host 127.0.0.1 --port 8765
```

Open [http://127.0.0.1:8765/](http://127.0.0.1:8765/). Use another port if needed (e.g. `--port 8762`).

### CLI options (`viz_main.py`)

| Argument | Description |
|----------|-------------|
| `--build-full` | Generate complete CF JSON and exit (default if no mode flag) |
| `--serve` | Start FastAPI + WebGL UI (builds JSON first if needed) |
| `--data-dir` | Input directory (default `./data`) |
| `-o` / `--output` | Graph JSON path (default `reports/cf_full_graph.json`) |
| `--process` | Optional Astrée process filter (e.g. `main-process`) |
| `--host` / `--port` | Bind address (default `127.0.0.1:8765`) |

Environment variables set by `--serve`:

| Variable | Meaning |
|----------|---------|
| `AOOB_DATA_DIR` | Data directory |
| `AOOB_GRAPH_JSON` | Path to `cf_full_graph.json` |
| `AOOB_PROCESS` | Optional process filter |

### Why it stays fast

| Technique | Effect |
|-----------|--------|
| Server-side layout | Browser does no force simulation |
| sigma.js **WebGL** | Handles ~10k nodes smoothly |
| Edges **hidden by default** | Avoids drawing 18k edges until needed |
| Sequence-only path edges on highlight | Only DF/CF bridge edges appear |
| Lean JSON (no pretty-print) | Faster download + parse |

### UI layout

```text
┌──────────────────┬──────────────────────────┬─────────────────┐
│ Left chat        │ Center (WebGL CF graph)  │ Right           │
│ stream + Order   │ auto-highlight on send   │ DF sequence     │
└──────────────────┴──────────────────────────┴─────────────────┘
```

#### Left panel (chatbot)

- Streaming transcript: status, **tool calls** (name + args), **tool results** (preview), **reasoning**, final **report** (including classification / comment / confidence)
- Composer: Order id (or “investigate 39”) → **Send**
- Graph toggles + variable dropdown + **Clear graph**
- Stats: nodes / edges / sequence steps

#### Center graph

- Full CF always loaded
- **No highlight:** all function-name labels visible; edges hidden
- **While highlighting:**
  - Sequence nodes colored by access type (read / write / mixed)
  - Bridge nodes (CF shortest-path intermediates) in gray
  - Everything else dimmed
  - **Labels only on sequence nodes** (`1. FnName`, `2. FnName`, …)
  - Only sequence path edges drawn (unless “show all edges”)
- Hover tooltip; click node copies id to clipboard
- **Fit** — reset camera to full graph (`animatedReset`)
- **Focus sequence** — zoom/pan to highlighted sequence using Sigma framed display coordinates (not raw layout coords)

#### Right panel (after highlight)

1. **First access** card — first DF event for the variable (function, read/write, location, process)
2. **Mini stats** — timeline steps, reads, writes, alarms on path
3. **Following sequence** — vertical timeline:
   - Cyan = **read**
   - Orange = **write**
   - Purple = **mixed** / other
   - Steps with alarms get a red-tinted card + nested alarm rows
  - Mode toggle:
    - **Writes + alarm read** (default): classification-focused path
    - **Full sequence**: includes intermediate read/bridge context
4. **Alarms in this sequence** — every Astrée alarm whose source line appears on this variable’s DF events (queried Order marked)

### Streaming investigate API

`POST /api/investigate/stream` with JSON `{ "order_id": "39" }` (optional `"model"`) returns **SSE** (`text/event-stream`) events:

| `type` | Meaning |
|--------|---------|
| `status` | Progress message (includes ~12s heartbeats while an LLM call is in flight) |
| `tool_call` | Tool name + args |
| `tool_result` | Truncated tool output preview |
| `no_tool_call` | Planner emitted explicit no-tool marker for this turn |
| `agent` | Model reasoning text (when no tool calls) |
| `report` | Final `AlarmInvestigationReport` JSON |
| `error` | Failure message |
| `done` | Stream finished successfully |

Requires at least one configured backend in `.env` (reachable Ollama in selected mode, or NVIDIA key).

### Color legend

| Color | Meaning |
|-------|---------|
| Cyan `#38bdf8` | Read |
| Orange `#fb923c` | Write |
| Purple `#c084fc` | Mixed read+write on same function |
| Gray | CF bridge / context |
| Dim slate | Non-sequence nodes while highlighting |
| Rose | Alarm markers in the right panel |

### What “highlight sequence” means (backend)

For Order id `N`:

1. Load alarm `N` → parse location line
2. Collect DF variables at that line; pick primary (or user override)
3. Gather that variable’s DF events ordered by source line
4. Build ordered **function sequence** (collapse consecutive same-function runs in the timeline)
5. For each consecutive pair `(A, B)` in the sequence, find a CF bridge:
   - `direct` — call edge `A → B` exists
   - `shortest` — shortest directed (or undirected fallback) path
   - `disconnected` — no path
6. Index all alarms by line; attach any alarm whose line matches a DF event on this path
7. Return highlight sets + `first_access` + `timeline` + `alarms_in_sequence`

### HTTP API

Base URL: `http://127.0.0.1:8765` (or your `--port`)

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/` | Web UI |
| `GET` | `/api/health` | Load status, stats, graph version, backend readiness |
| `GET` | `/api/graph` | Complete CF payload (nodes/edges/stats) |
| `POST` | `/api/highlight` | Order id → sequence highlight payload |
| `POST` | `/api/investigate/stream` | SSE agent investigation |
| `GET` | `/api/alarms?limit=50` | Sample alarm list |

#### `POST /api/highlight`

Request:

```json
{ "order_id": "39", "variable": null }
```

`variable` is optional; when omitted, the primary DF variable at the alarm site is used.

Response (main fields):

```json
{
  "alarm": { "order": 39, "type": "Alarm (A)", "category": "...", "location": "..." },
  "site_variables": ["MOCMEM_VBLSTRUCTEARLYCLEAREDRAM_MO_INST_LOOKUP"],
  "variable": "MOCMEM_VBLSTRUCTEARLYCLEAREDRAM_MO_INST_LOOKUP",
  "first_access": {
    "order": 1,
    "function": "MoCEMM_Co_ChkwordFirstCoreTsk_MoCMem_st",
    "access": "read",
    "access_kind": "read",
    "location": "ALL_bc_with_context.c:278906.11-90",
    "line": 278906,
    "process": "main-process",
    "alarms": [],
    "is_first": true
  },
  "function_sequence": ["...", "..."],
  "function_access": { "SomeFn": "read" },
  "bridge_nodes": ["..."],
  "highlight_nodes": ["..."],
  "highlight_edge_ids": ["A->B"],
  "segments": [{ "from": "A", "to": "B", "kind": "direct", "nodes": ["A","B"], "edges": ["A->B"] }],
  "events": [],
  "timeline": [],
  "alarms_in_sequence": [],
  "counts": { "events": 71, "reads": 71, "writes": 0, "alarms": 29, "timeline_steps": 13 }
}
```

#### Examples (PowerShell)

```powershell
Invoke-RestMethod http://127.0.0.1:8765/api/health

Invoke-RestMethod http://127.0.0.1:8765/api/highlight `
  -Method POST -ContentType "application/json" `
  -Body '{"order_id":"39"}'
```

---

## Batch evaluation

Script: `scripts/batch_eval_10.py`  
Output: `reports/batch_eval_10.json`

Runs the agent on a fixed sample of alarms and records:

- classification outputs in the current space: `true` / `false` / `review`
- resolved-decision metrics (`n_resolved`, `resolution_rate`, `human_match_among_resolved`)
- internal-consistency flags (`interval_shortcut`, `silent_failure`, `missing_bounds_evidence`)

Human labels from `Full_alarms_with_result.csv` are still reported for reference, but consistency flags are scored from internal reasoning quality (not human-label match).

Sample Order ids (as shipped in the script):

| Order | Human label |
|-------|-------------|
| 2475 | true |
| 6413 | true (low) |
| 6355 | true (low) |
| 39, 41, 48, 91, 1129, 1926, 2858 | false |

```powershell
py -3.13 scripts\batch_eval_10.py
```

Each row records agent classification (`true|false|review`), match/reference fields, confidence, comment, Astrée message, evidence completeness, tools used, timing, errors, and reasoning-consistency flags (`interval_shortcut`, `silent_failure`, `missing_bounds_evidence`).

Scoring is coarse (binary true vs false). Residual false positives on abstract-interval-heavy alarms and occasional API timeouts are expected; re-run after prompt/tool changes to compare.

---

## Shared data layer

`aoob_agent/data_store.py` is used by **both** pipelines:

| Index | Source |
|-------|--------|
| `alarms[order]` | `Full_alarms.csv` |
| `data_flow_by_line[line]` | `data flow.csv` |
| `data_flow_by_variable[name]` | `data flow.csv` (bare + full names) |
| `control_by_caller` / `control_by_callee` / `control_by_line` | `control flow.csv` |
| `source_lines` | `input.c` (1-based) |
| Declaration index | Parsed from `input.c` for `get_declaration_info` |

CF visualization builds a NetworkX `DiGraph` from control-flow rows and overlays DF sequences for highlighting.

---

## Configuration

| Variable / flag | Used by | Meaning | Default |
|-----------------|---------|---------|---------|
| `AOOB_LLM_CHOOSED` | Agent / UI stream | Backend source selector: `local` / `network` / `nvidia` | unset (auto preference) |
| `OLLAMA_LOCAL_MODEL` | Agent | Local Ollama model tag | unset |
| `OLLAMA_LOCAL_BASE_URL` | Agent | Local Ollama endpoint | `http://127.0.0.1:11434` |
| `OLLAMA_NETWORK_MODEL` | Agent | Network Ollama model tag | unset |
| `OLLAMA_NETWORK_BASE_URL` | Agent | Network Ollama endpoint | unset |
| `OLLAMA_NUM_CTX` | Agent | Ollama context window (`num_ctx`) | `16384` |
| `OLLAMA_TIMEOUT` | Agent | Ollama generation timeout (seconds) | `300` |
| `AOOB_MAX_VISIBLE_TOOLS` | Agent | Max tool schemas shown to model per turn | unset (all tools) |
| `AOOB_OLLAMA_RUNTIME_FALLBACK` | Agent | If `1`, fallback to NVIDIA on Ollama timeout/connection errors | `1` |
| `NVIDIA_API_KEY` | Agent / UI stream | NVIDIA API Catalog key | optional (required if `AOOB_LLM_CHOOSED=nvidia` or fallback needed) |
| `NVIDIA_MODEL` | Agent | NVIDIA model id | `meta/llama-3.1-70b-instruct` |
| `NVIDIA_TIMEOUT` | Agent | NVIDIA timeout (seconds) | `300` |
| `--model` | Agent CLI / stream body | Runtime model override (backend-dependent) | — |
| `--data-dir` | Both | Input directory | `./data` |
| `AOOB_DATA_DIR` | CF server | Data directory | from `--data-dir` |
| `AOOB_GRAPH_JSON` | CF server | Full graph JSON path | `reports/cf_full_graph.json` |
| `AOOB_PROCESS` | CF server | Optional process filter | unset |

Legacy (not used in current single-model orchestration): `AOOB_TOOL_LOCAL_MODEL`, `AOOB_TOOL_NETWORK_MODEL`, `AOOB_TOOL_MODEL`.

Any tool-calling-capable Ollama or NVIDIA chat model can work for the agent; larger models are slower and more timeout-prone.

---

## Troubleshooting

| Symptom | What to try |
|---------|-------------|
| `No LLM configured` | Set `AOOB_LLM_CHOOSED` and corresponding Ollama vars, or set `NVIDIA_API_KEY`; restart `--serve` |
| Slow first tool call (`Still waiting on Ollama ...`) | Reduce `OLLAMA_NUM_CTX` (e.g. `4096`), set `AOOB_MAX_VISIBLE_TOOLS=3`, or switch to a smaller `OLLAMA_*_MODEL` |
| `alarm Order … not found` | Check `Full_alarms.csv`; try `1112` vs `1,112` |
| `ModuleNotFoundError` | Install into the **same** Python you run: `py -3.13 -m pip install -r requirements.txt` |
| Agent timeout (Ollama/NVIDIA) | Lower `OLLAMA_TIMEOUT` to trigger fallback sooner; use smaller model; verify endpoint/network |
| Agent keeps hopping callers unnecessarily | Confirm `get_trimmed_sequence` returned `origin_resolved`; if it is `true`, caller hop should stop. Also ensure tool 3 is tracing the index operand from `index_operands_at_alarm`, not the indexed array symbol |
| Agent keeps reasoning but does not emit tool calls | Dynamic nudge + no-tool-call guard should now force either explicit schema-following tool calls or report fallback after two consecutive no-tool-call turns |
| Validation error like `summary Field required` in final report | Planner control payload (for example `{"type":"no_tool_call"}`) should now be filtered from report reuse; update to latest `aoob_agent/agent.py` and restart `--serve` |
| Empty / weak agent report | Confirm DF rows exist at the alarm line; ensure `get_trimmed_sequence` ran on the index operand and (if `origin_resolved=false` with parameter scope) `get_caller_context` follow-up was attempted; try another model |
| Complex pointer/member alarm remains inconclusive | Check whether `get_variable_scope` returned `dynamic_heap`; if yes, inspect allocation/capacity evidence and keep `review` unless safe bounds can be proven |
| Overconfident `true` from interval alone | Confirm prompts/gates are current; Message is abstract — check `index_origin_summary` |
| Human labels leaking into agent | Regenerate via `py -3.13 data\convert.py` so Classification/Comment are empty |
| UI stuck / browser freeze (old build) | Use current WebGL UI; rebuild `cf_full_graph.json` with `--build-full`; hard-refresh |
| **Focus sequence** black screen | Hard-refresh; Focus uses framed display coords (Fit still uses reset). Outdated cached `index.html` causes this |
| Port already in use | Stop the old process or use `--port 8766` |
| UI looks outdated | Hard-refresh (`Ctrl+F5`); restart `--serve` after Python API changes |
| No variable / empty right panel | Alarm line may have no DF rows; try another Order or pick a variable from the dropdown |
| Labels missing while not highlighting | Hard-refresh; default is all labels on until you highlight |
| CDN / sigma failed to load | Check network access to `cdn.jsdelivr.net` and Google Fonts |
| Batch eval timeouts | Re-run failed Order ids individually via `main.py`; check Ollama/NVIDIA latency and timeout settings |

---

## Dependencies

From `requirements.txt`:

| Package | Purpose |
|---------|---------|
| `langchain` / `langchain-core` | Agent tooling |
| `langchain-ollama` | Ollama chat backend |
| `langchain-nvidia-ai-endpoints` | NVIDIA NIM chat |
| `langgraph` | Agent graph |
| `python-dotenv` | `.env` loading |
| `pydantic` | Report / request schemas |
| `networkx` | CF graph + path finding |
| `fastapi` / `uvicorn` | Chat + graph UI server |

Browser (CF UI): **graphology** + **sigma.js** (+ `sigma/utils` for camera fit helpers) loaded from jsDelivr ESM CDN.

---

## License

Project-internal use unless otherwise specified by the repository owner.
