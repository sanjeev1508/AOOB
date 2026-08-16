# AOOB-Graph

Investigate Astrée **out-of-bounds (AOOB)** alarms over shared CSV / source exports with:

1. **Primary UI** (`viz_main.py --serve`) — left **streaming chatbot**, center **WebGL CF graph**, right **DF sequence** panel.  
   Enter an Order id; the agent streams status, tool calls/args, tool results, reasoning, and the final report while the graph highlights the DF path.

2. **CLI agent** (`main.py`) — same LangGraph / NVIDIA NIM investigation, writing a JSON report to stdout or a file (no UI).

3. **Batch eval** (`scripts/batch_eval_10.py`) — fixed 10-alarm sample scored against human labels in `Full_alarms_with_result.csv`.

**Design rule:** tools are **retrieval-only**. They never hardcode triage. The model authors `classification` (`true` / `false`), `comment`, and `confidence`. Human CSV Classification/Comment columns are **not** fed to the agent (prepare `Full_alarms.csv` with `data/convert.py` so those fields are empty).

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
- NVIDIA API Catalog / NIM via `langchain-nvidia-ai-endpoints` (default model: `meta/llama-3.1-70b-instruct`)
- Nine retrieval-only tools (symbols, snippet, DF sequence, declaration bounds, backward slice,
  condition guards, symbolic constants, index-expression structure, flat all-writes)- Pydantic report with agent-authored:
  - `classification` (`true` = real bug, `false` = false positive)
  - `comment`, `confidence`, `astree_message`, optional `bounds_evidence`
- Astrée `Message` (abstract intervals) is passed through tools; human Classification/Comment are not
- **Process gates** (not verdict hardcoding): nudge until a useful **index-operand** slice exists; soft-cap `confidence` / `evidence_completeness` if never traced
- Soft tool-message ceiling: **10**
- CLI (`main.py`) with optional JSON file output
- Order ids with thousand separators supported (`1,112` → `1112`)
- Prompt discipline: index-expression arithmetic before Astrée interval compare; sentinel/invalid constants need guards; unresolved evidence prefers uncertainty over the raw interval
### Chat + CF visualization UI

- One-time **full CF graph** build (~9.8k nodes / ~18k edges) with **server-side layout**
- Fast **sigma.js WebGL** UI (no browser force-layout freeze)
- **Left:** streaming chat — Order id → live tool calls, evidence previews, reasoning, report
- **Center:** full CF graph; auto-highlights DF sequence when you investigate
- **Right:** first access, following sequence timeline, alarms on path
- **Read** / **write** / **mixed** color coding (graph + timeline)
- Labels: **all function names** when not highlighting; **sequence-only labels** while highlighting
- Edges hidden by default; optional “show all CF edges”
- Camera: **Fit** (full graph) and **Focus sequence** (framed viewport fit to highlighted nodes)
- SSE heartbeats (~12s) while waiting on NVIDIA
- REST + SSE: `/api/graph`, `/api/highlight`, `/api/investigate/stream`, `/api/health`, `/api/alarms`

---

## Repository layout

```text
AOOB-Graph/
├── main.py                     # Agent CLI (Order id → JSON report)
├── viz_main.py                 # Build full CF graph / serve chat+graph UI
├── requirements.txt
├── .env                        # NVIDIA_API_KEY (do not commit)
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
│   └── test_evidence_tools.py  # Tool / index-operand acceptance checks
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
- For the **agent**: NVIDIA API key from [build.nvidia.com](https://build.nvidia.com/settings/api-keys) and network access to `integrate.api.nvidia.com`
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
NVIDIA_API_KEY=nvapi-...
# Optional override (default: meta/llama-3.1-70b-instruct)
# NVIDIA_MODEL=meta/llama-3.1-70b-instruct
```

Do **not** commit `.env`.

---

## Quick start

### Primary: chat + graph UI

```powershell
# 1) Build complete CF graph once (skip if reports\cf_full_graph.json already exists)
py -3.13 viz_main.py --build-full -o reports\cf_full_graph.json

# 2) Serve UI (needs NVIDIA_API_KEY in .env for the agent stream)
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
| `--model` | NVIDIA model id (overrides `NVIDIA_MODEL`) |
| `-o` / `--output` | Optional path for the JSON report |

### How the agent works

```text
┌─────────────┐     tool calls      ┌──────────────┐
│  agent node │ ──────────────────► │  tools node  │
│ (NVIDIA LLM)│ ◄────────────────── │ (retrieval)  │
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
4. **Index-origin gate:** for OOB-style work, if no useful slice on an `index_operand_candidates` symbol was seen, a nudge message is injected (up to 3 times) before allowing finalize. This does **not** force `true`/`false`.
5. Soft tool-message ceiling **10**, then force the report node.
6. Final LLM turn builds `AlarmInvestigationReport`. Soft post-process only: if no useful index slice, may downgrade `confidence` `high→medium` and `evidence_completeness` `fully traced→not traced`. Classification is never flipped in Python.

### Prompts (important semantics)

- Astrée `[lo, hi]` messages are **abstract domain** bounds, not concrete runtime indices.
- Forbidden over-reading: “`array_size < Astrée hi` ⇒ classification `true`”.
- Prefer slicing **index operands** (e.g. `checkword`, `idxMRPChnl_u16`), not the array/table name and not sentinel end-pointers.
- `classification` / `comment` / `confidence` are always agent-authored from tool evidence.

### Tools

All tools are **retrieval-only**. They never emit bug/no-bug, severity, or confidence.

#### 1. `get_affected_symbols`

Given an alarm Order id, returns:

- Alarm metadata (including Astrée `message` when present) + parsed location
- Data-flow records near the alarm line
- Control-flow edges at that line
- Source line text from `input.c`
- `indexed_objects` — arrays/objects being indexed (from source parse)
- `index_operand_candidates` — symbols to prefer for `get_backward_slice`

#### 2. `get_function_snippet`

Returns:

- Enclosing function name and span
- Bounded source snippet around the alarm
- Compact caller / callee lists

#### 3. `get_variable_manipulation_sequence`

Returns ordered read/write events (and related call edges) for variables at the alarm site, or for a `variable_name` the agent supplies after tool 1. Counts toward index-origin coverage when useful.

#### 4. `get_declaration_bounds`

Looks up a bare symbol in the declaration index from `input.c`:

- `declared_type`, `array_size` (**literal sizes only**), `is_pointer`, `location`
- `parse_note` when size is an expression/macro/multi-dim — **no guessing**
- Handles common typedefs / `1u`-style sizes / brace-initializer layouts

#### 5. `get_backward_slice`

Walks backward from a location over DF writes, same-function source initializers/assignments, and CF callers. Returns:

- `writes_found` (with `assigned_expression_text`, `call_depth`, `before_location` per call site)
- `unresolved_paths` (e.g. no visible caller in CF graph)
- `truncated` (depth / write-count limit)
- Lookup diagnostics when the symbol/location cannot be matched

Pass an **index / pointer operand**, not the indexed array (unless you intentionally trace that object).

#### 6. `get_condition_guards`

Finds `if` / `else if` / `switch` / `assert` / ternary text that mentions a variable before the access (and in CF callers). Raw condition text only — does not judge sufficiency.

#### 7. `resolve_symbolic_constant`

Looks up an enum member or `#define` in `input.c` and returns a literal integer when resolvable (`parse_note` otherwise). Prefer this over inferring meaning from identifier spelling (e.g. `…Invalid…`).

#### 8. `get_index_expression_structure`

Parses a C index / subscript expression into `operand` + `operations` (shifts, masks, arithmetic, casts, fields). Structure only — no range evaluation.

#### 9. `get_all_writes_to_symbol`

Flat exhaustive DF write listing for a symbol (no path/depth filter). Cross-check when `get_backward_slice` is truncated; reachability still needs slice/CF reasoning.

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
  "classification": "false",
  "comment": "Index operand checkword initialized to 0; Astrée interval is abstract…",
  "confidence": "medium",
  "bounds_evidence": {
    "symbol": "MOCMEM_VBLSTRUCTEARLYCLEAREDRAM_MO_INST_LOOKUP",
    "declared_size": 2,
    "index_origin_summary": "checkword initialized to 0 in the same function…",
    "evidence_completeness": "partially traced"
  },
  "tools_used": [
    "get_affected_symbols",
    "get_function_snippet",
    "get_variable_manipulation_sequence",
    "get_declaration_bounds",
    "get_backward_slice"
  ]
}
```

| Field | Notes |
|-------|--------|
| `astree_message` | Copy of Astrée Message from tools — not a human verdict |
| `classification` | `true` \| `false` — **agent-authored**; never read from CSV |
| `comment` | Agent rationale — not CSV Comment |
| `affected_symbols[].kind` | `variable` \| `array` \| `pointer` \| `struct` \| `field` \| `other` \| `unknown` |
| `confidence` | `low` \| `medium` \| `high` — LLM-chosen; may be soft-capped if index origin missing |
| `bounds_evidence.evidence_completeness` | `fully traced` \| `partially traced` \| `not traced` |

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
- LLM timeout is generous (~180s); UI sends heartbeats while waiting.
- After enough tool messages, the graph moves to the report node to avoid long multi-turn timeouts.
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
4. **Alarms in this sequence** — every Astrée alarm whose source line appears on this variable’s DF events (queried Order marked)

### Streaming investigate API

`POST /api/investigate/stream` with JSON `{ "order_id": "39" }` (optional `"model"`) returns **SSE** (`text/event-stream`) events:

| `type` | Meaning |
|--------|---------|
| `status` | Progress message (includes ~12s heartbeats while NVIDIA is busy) |
| `tool_call` | Tool name + args |
| `tool_result` | Truncated tool output preview |
| `agent` | Model reasoning text (when no tool calls) |
| `report` | Final `AlarmInvestigationReport` JSON |
| `error` | Failure message |
| `done` | Stream finished successfully |

Requires `NVIDIA_API_KEY` in `.env` (loaded when the server starts).

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
| `GET` | `/api/health` | Load status, stats, graph version, NVIDIA key present? |
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

Runs the agent on a fixed sample of **3 positive-ish** and **7 false** alarms, comparing agent `classification` to human labels from `Full_alarms_with_result.csv` (`true` / `true (low)` both count as positive for scoring).

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

Each row records agent classification, match flag, confidence, comment, Astrée message, evidence completeness, tools used, timing, and errors (e.g. NVIDIA timeout).

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
| Declaration index | Parsed from `input.c` for `get_declaration_bounds` |

CF visualization builds a NetworkX `DiGraph` from control-flow rows and overlays DF sequences for highlighting.

---

## Configuration

| Variable / flag | Used by | Meaning | Default |
|-----------------|---------|---------|---------|
| `NVIDIA_API_KEY` | Agent / UI stream | NVIDIA API Catalog key | required for investigation |
| `NVIDIA_MODEL` | Agent | Chat model id | `meta/llama-3.1-70b-instruct` |
| `--model` | Agent CLI / stream body | Override model | — |
| `--data-dir` | Both | Input directory | `./data` |
| `AOOB_DATA_DIR` | CF server | Data directory | from `--data-dir` |
| `AOOB_GRAPH_JSON` | CF server | Full graph JSON path | `reports/cf_full_graph.json` |
| `AOOB_PROCESS` | CF server | Optional process filter | unset |

Any tool-calling-capable NVIDIA chat model can work for the agent; larger models are slower and more timeout-prone.

---

## Troubleshooting

| Symptom | What to try |
|---------|-------------|
| `NVIDIA_API_KEY missing` | Set key in `.env` or environment; restart `--serve` |
| `alarm Order … not found` | Check `Full_alarms.csv`; try `1112` vs `1,112` |
| `ModuleNotFoundError` | Install into the **same** Python you run: `py -3.13 -m pip install -r requirements.txt` |
| Agent `Read timed out` to NVIDIA | Retry; smaller/faster model; stable network; tool payloads already truncated |
| Empty / weak agent report | Confirm DF rows exist at the alarm line; ensure index slice ran; try another model |
| Overconfident `true` from interval alone | Confirm prompts/gates are current; Message is abstract — check `index_origin_summary` |
| Human labels leaking into agent | Regenerate via `py -3.13 data\convert.py` so Classification/Comment are empty |
| UI stuck / browser freeze (old build) | Use current WebGL UI; rebuild `cf_full_graph.json` with `--build-full`; hard-refresh |
| **Focus sequence** black screen | Hard-refresh; Focus uses framed display coords (Fit still uses reset). Outdated cached `index.html` causes this |
| Port already in use | Stop the old process or use `--port 8766` |
| UI looks outdated | Hard-refresh (`Ctrl+F5`); restart `--serve` after Python API changes |
| No variable / empty right panel | Alarm line may have no DF rows; try another Order or pick a variable from the dropdown |
| Labels missing while not highlighting | Hard-refresh; default is all labels on until you highlight |
| CDN / sigma failed to load | Check network access to `cdn.jsdelivr.net` and Google Fonts |
| Batch eval timeouts | Re-run failed Order ids individually via `main.py`; check NVIDIA quota / latency |

---

## Dependencies

From `requirements.txt`:

| Package | Purpose |
|---------|---------|
| `langchain` / `langchain-core` | Agent tooling |
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
