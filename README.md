# AOOB-Graph

Triage Astrée **out-of-bounds** alarms. Python compiles the case (operand, object size, origin→alarm path) and **opens every path window**. The LLM only `inspect`s listed helpers and `submit_verdict`s `true` / `false` / `review`. Human CSV labels are never fed to the agent.

## Inputs (`data/`)

| File | Role |
|------|------|
| `Full_alarms.csv` | Alarms (keep `Message`; leave Classification/Comment empty) |
| `Full_alarms_with_result.csv` | Human labels for eval only |
| `data flow.csv` / `control flow.csv` | DF / CF |
| `input.c` | Preprocessed source (line-aligned with Astrée locations) |

```powershell
py -3.13 data\convert.py
```

clears human labels into `Full_alarms.csv`. Locations look like `file:line.col-col`. Order ids may include commas (`1,112` → `1112`).

## Setup

Python 3.11+ (tested 3.13). `py -3.13 -m pip install -r requirements.txt`

`.env`:

```env
AOOB_LLM_CHOOSED=local          # local | network | nvidia  (honored; not a hint)
OLLAMA_LOCAL_MODEL=qwen3:14b
OLLAMA_LOCAL_BASE_URL=http://127.0.0.1:11434
OLLAMA_TIMEOUT=180
# NVIDIA only if choosed=nvidia, or after an Ollama error when:
# AOOB_OLLAMA_RUNTIME_FALLBACK=1
# NVIDIA_API_KEY=nvapi-...
```

## Run

```powershell
py -3.13 viz_main.py --build-full -o reports\cf_full_graph.json
py -3.13 viz_main.py --serve --host 127.0.0.1 --port 8765
# UI: http://127.0.0.1:8765/  — type an Order id (e.g. 39)

py -3.13 main.py 39 -o reports\alarm_39.json

py -3.13 scripts\batch_eval_20.py
py -3.13 scripts\batch_eval_20.py --resume --retries 4
```

Send on the UI: highlight paints the compiled path on the CF graph; SSE runs the same investigation as `main.py`.

## Pipeline

```
Order id
  → compile_case (no LLM): object, operand, origin path (from the CF/DF graph), Python guards
  → pack_path_windows: every path step is opened in Python, in origin→alarm order (budget-capped, [WINDOW TRUNCATED] marks any cut)
  → LLM: gets the case brief (object declaration + operand scope) and the packed windows before any tool call;
         may call get_window / move_window (walk the same graph-ordered path step-by-step) or inspect(helper) to
         re-open a step or helper — these always return the COMPLETE function body, never a clipped slice
  → submit_verdict: true / false / review
  → report JSON (classification / comment / confidence)
```

`get_window`, `move_window`, and `inspect` never truncate: whatever function they open, they return it start-to-end (only `pack_path_windows`'s upfront bulk preview applies a line budget, and it flags any cut with a `[WINDOW TRUNCATED]` marker so the LLM knows to re-open that step for the full body). Returned lines are dedented (shared leading whitespace stripped, structural anchors like bare braces / `# N "file.c"` markers ignored when computing the shared amount) to cut tokens on deeply-nested legacy C without changing any code content.

**The alarm step must be opened explicitly.** Python's automatic `pack_path_windows()` pass marks every step "opened" for bookkeeping/preview purposes, but that alone no longer satisfies `submit_verdict`'s gate. The LLM must itself call `get_window` or `move_window` (e.g. `step=path_length`) so it lands on the alarm-role step before a verdict is accepted — otherwise `submit_verdict` rejects it and tells the model which step to open.

The `cf_viz` UI's live tool-result preview (SSE) sizes its cap to the tool: `get_window`/`move_window`/`inspect` previews are shown up to ~90K chars (comfortably covering any single function, since `find_enclosing_function` already caps a function span at 1200 lines) instead of a flat 1200-character cut that could land mid-snippet. This only affects what's rendered live in the browser — the LLM's own message history always held the full tool output.

Astrée `[lo, hi]` is an abstract interval, not a concrete runtime index. Missing size/origin ⇒ `review`. `true`/`false` with unknown size is rejected.

Layout: `aoob_agent/` (compiler, tools, LangGraph), `cf_viz/` (WebGL + SSE), `scripts/batch_eval_20.py`.

## Config

| Variable | Meaning |
|----------|---------|
| `AOOB_LLM_CHOOSED` | `local` / `network` / `nvidia` |
| `OLLAMA_LOCAL_MODEL` / `_BASE_URL` | Local Ollama |
| `OLLAMA_NETWORK_MODEL` / `_BASE_URL` | Remote Ollama |
| `OLLAMA_NUM_CTX` | Context size (default 16384) |
| `NVIDIA_API_KEY` / `NVIDIA_MODEL` | NVIDIA backend |
| `OLLAMA_TIMEOUT` / `NVIDIA_TIMEOUT` | Seconds (default 300) |

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `No LLM configured` | Set `AOOB_LLM_CHOOSED` + Ollama vars or `NVIDIA_API_KEY` |
| Alarm not found | Check `Full_alarms.csv`; try `1112` vs `1,112` |
| Human labels leaking | Re-run `data\convert.py` |
| UI stale | Hard-refresh; restart `--serve` after Python changes |
| Port in use | `--port 8766` |

License: project-internal unless the owner says otherwise.