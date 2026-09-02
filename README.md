# AOOB-Graph

Triage Astrée **out-of-bounds** alarms. Python compiles the case (operand, object size, origin→alarm path, helpers, guards). The LLM opens source with tools and `submit_verdict`s `true` / `false` / `review`. Human CSV labels are never fed to the agent.

## Inputs (`data/`)

| File                                 | Role                                                        |
| ------------------------------------ | ----------------------------------------------------------- |
| `Full_alarms.csv`                    | Alarms (keep `Message`; leave Classification/Comment empty) |
| `Full_alarms_with_result.csv`        | Human labels for eval only                                  |
| `data flow.csv` / `control flow.csv` | DF / CF                                                     |
| `input.c`                            | Preprocessed source (line-aligned with Astrée locations)    |

```powershell
py -3.13 data\convert.py
```

clears human labels into `Full_alarms.csv`. Locations look like `file:line.col-col`. Order ids may include commas (`1,112` → `1112`).

## Setup

Python 3.11+ (tested 3.13). `py -3.13 -m pip install -r requirements.txt`

`.env` (never commit this file; it contains credentials):

```env
# Tool calls and intermediate investigation reasoning
AOOB_TOOL_LLM_CHOOSED=network  # local | network | nvidia | bosch

# Final classification/report generation
AOOB_CLASSIFY_LLM_CHOOSED=bosch # local | network | nvidia | bosch

OLLAMA_LOCAL_MODEL=qwen3:14b
OLLAMA_LOCAL_BASE_URL=http://127.0.0.1:11434
OLLAMA_NETWORK_MODEL=gemma4:26b
OLLAMA_NETWORK_BASE_URL=http://your-network-ollama-host:8503
OLLAMA_TIMEOUT=180

# NVIDIA (when either role is set to nvidia)
NVIDIA_API_KEY=nvapi-...
# NVIDIA_MODEL=nvidia/nemotron-3.5-lightning-30b-a3b

# Bosch AOAI Model Farm (when either role is set to bosch)
MODEL_FARM_API_KEY=...
BOSCH_BASE_URL=https://aoai-farm.bosch-temp.com/api/openai/deployments/<deployment>
BOSCH_MODEL=gpt-5.6-luna
BOSCH_API_VERSION=2024-05-01-preview
BOSCH_TIMEOUT=180
```

`AOOB_TOOL_LLM_CHOOSED` controls the model that calls investigation tools and performs
intermediate reasoning. `AOOB_CLASSIFY_LLM_CHOOSED` controls the separate model that
produces the final classification. The legacy `AOOB_LLM_CHOOSED` variable is still
accepted as a fallback for both roles.

## Run

```powershell
py -3.13 viz_main.py --build-full -o reports\cf_full_graph.json
py -3.13 viz_main.py --serve --host 127.0.0.1 --port 8765

# UI: http://127.0.0.1:8765/ — type an Order id (e.g. 39)

py -3.13 main.py 39 -o reports\alarm_39.json

py -3.13 scripts\batch_eval_100.py --preview
py -3.13 scripts\batch_eval_100.py --retries 4
py -3.13 scripts\batch_eval_100.py --resume --retries 4
```

Send on the UI: highlight paints the compiled path on the CF graph; SSE runs the same investigation as `main.py`.

## Pipeline

```text
Order id
  → compile_case (no LLM): object, operand, origin→alarm path,
                            helpers, Python-extracted guards, gaps
  → tool/reasoning LLM: case brief in the first human message
         get_window / move_window / inspect / inspect_declaration
         (each open returns the complete function or declaration)
  → submit_verdict: true | false | review
  → classification LLM: final classification from the gathered evidence
  → report JSON (classification / comment / confidence)
```

The investigation **ends only on an accepted `submit_verdict`**. If the model keeps browsing until the tool budget, or goes idle without a verdict, the next turn binds `tool_choice=submit_verdict` so it has to close. Python does not invent a `review` in place of that call (except as a last-resort fallback if the forced submit still produces no verdict).

`get_window`, `move_window`, and `inspect` never truncate for the LLM: whatever function they open, they return start-to-end (a 2000-line safety valve exists only for pathological generated functions). Returned lines are dedented (shared leading whitespace stripped; structural anchors like bare braces / `# N "file.c"` markers ignored when computing the shared amount).

`submit_verdict` accepts `true` / `false` / `review`. The only evidence gate: `true`/`false` is rejected when declared size is unknown — submit `review` instead. There is no comment-phrase scanner, no write-helper inspect gate, and no Python rewrite of a submitted classification.

Astrée `[lo, hi]` is an abstract interval, not a concrete runtime index.

The `cf_viz` UI's live tool-result preview (SSE) sizes its cap to the tool: `get_window` / `move_window` / `inspect` previews up to ~90K chars. This only affects the browser; the LLM message history keeps the full tool output.

## Layout

`aoob_agent/` (compiler, tools, LangGraph), `cf_viz/` (WebGL + SSE), `scripts/batch_eval_100.py`.

## Config

| Variable                                      | Meaning                                                   |
| --------------------------------------------- | --------------------------------------------------------- |
| `AOOB_TOOL_LLM_CHOOSED`                      | Tool/reasoning source: `local` / `network` / `nvidia` / `bosch` |
| `AOOB_CLASSIFY_LLM_CHOOSED`                  | Final classification source: `local` / `network` / `nvidia` / `bosch` |
| `AOOB_LLM_CHOOSED`                            | Legacy fallback for both role-specific source variables |
| `OLLAMA_LOCAL_MODEL` / `_BASE_URL`            | Local Ollama                                              |
| `OLLAMA_NETWORK_MODEL` / `_BASE_URL`          | Remote Ollama                                             |
| `OLLAMA_NUM_CTX`                              | Context size (default 16384)                              |
| `NVIDIA_API_KEY` / `NVIDIA_MODEL`             | NVIDIA backend                                            |
| `MODEL_FARM_API_KEY` / `BOSCH_*`              | Bosch AOAI Model Farm backend                             |
| `OLLAMA_TIMEOUT` / `NVIDIA_TIMEOUT` / `BOSCH_TIMEOUT` | Backend request timeout in seconds (default 300) |

## Troubleshooting

| Symptom              | Fix                                                      |
| -------------------- | -------------------------------------------------------- |
| `No LLM configured`  | Set the role-specific source variable and its backend credentials |
| Alarm not found      | Check `Full_alarms.csv`; try `1112` vs `1,112`           |
| Human labels leaking | Re-run `data\convert.py`                                 |
| UI stale             | Hard-refresh; restart `--serve` after Python changes     |
| Port in use          | `--port 8766`                                            |

License: project-internal unless the owner says otherwise.
