"""FastAPI UI: CF graph (WebGL) + streaming agent chat for Order-id investigation."""

from __future__ import annotations

import asyncio
import json
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from aoob_agent.agent import llm_backend_label, stream_investigate, using_ollama
from aoob_agent.data_store import DataStore
from cf_viz.export import build_and_export_full_graph, load_graph_payload
from cf_viz.graph_builder import build_control_flow_graph
from cf_viz.highlight import highlight_for_alarm

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data"
DEFAULT_GRAPH_JSON = ROOT / "reports" / "cf_full_graph.json"
STATIC_DIR = Path(__file__).resolve().parent / "static"

_state: dict[str, Any] = {
    "store": None,
    "graph": None,
    "payload": None,
    "data_dir": DEFAULT_DATA,
    "graph_json": DEFAULT_GRAPH_JSON,
    "process": None,
}


class HighlightRequest(BaseModel):
    order_id: str = Field(..., description="Alarm Order id, e.g. 39 or 1,112")
    variable: Optional[str] = Field(
        None, description="Optional override if multiple DF vars at the site"
    )


class InvestigateRequest(BaseModel):
    order_id: str = Field(..., description="Alarm Order id, e.g. 39 or 1,112")
    model: Optional[str] = Field(None, description="Optional model override (Ollama tag or NVIDIA id)")


def _ensure_loaded() -> None:
    if _state["store"] is None or _state["graph"] is None or _state["payload"] is None:
        raise HTTPException(503, "Graph not loaded yet")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv(ROOT / ".env")
    data_dir = Path(os.getenv("AOOB_DATA_DIR", str(DEFAULT_DATA)))
    graph_json = Path(os.getenv("AOOB_GRAPH_JSON", str(DEFAULT_GRAPH_JSON)))
    process = os.getenv("AOOB_PROCESS") or None
    _state["data_dir"] = data_dir
    _state["graph_json"] = graph_json
    _state["process"] = process

    print(f"[cf_viz] Loading data store from {data_dir} ...", flush=True)
    store = DataStore.load(data_dir)
    print("[cf_viz] Building in-memory CF graph ...", flush=True)
    graph = build_control_flow_graph(store, process=process)

    need_build = True
    if graph_json.exists():
        try:
            payload = load_graph_payload(graph_json)
            if payload.get("version") == 2 and payload.get("nodes"):
                need_build = False
                print(f"[cf_viz] Using prebuilt graph JSON: {graph_json}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[cf_viz] Prebuilt JSON unreadable ({exc}); rebuilding", flush=True)

    if need_build:
        print(f"[cf_viz] Exporting laid-out full graph → {graph_json}", flush=True)
        payload = build_and_export_full_graph(data_dir, graph_json, process=process)

    _state["store"] = store
    _state["graph"] = graph
    _state["payload"] = payload
    stats = payload.get("stats") or {}
    has_key = bool(os.getenv("NVIDIA_API_KEY"))
    local = using_ollama()
    print(
        f"[cf_viz] Ready — nodes={stats.get('nodes')} edges={stats.get('edges')} "
        f"alarms={len(store.alarms)} llm={llm_backend_label()} "
        f"nvidia_key={'yes' if has_key else 'NO'} ollama={'yes' if local else 'NO'}",
        flush=True,
    )
    yield


app = FastAPI(title="AOOB Graph + Agent Chat", lifespan=lifespan)
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    payload = _state.get("payload") or {}
    return {
        "ok": _state["payload"] is not None,
        "stats": payload.get("stats"),
        "graph_json": str(_state["graph_json"]),
        "data_dir": str(_state["data_dir"]),
        "version": payload.get("version"),
        "renderer": payload.get("renderer"),
        "nvidia_configured": bool(os.getenv("NVIDIA_API_KEY")),
        "ollama_configured": using_ollama(),
        "llm_backend": llm_backend_label(),
    }


@app.get("/api/graph")
def get_graph() -> dict[str, Any]:
    _ensure_loaded()
    return _state["payload"]


@app.post("/api/highlight")
def highlight(req: HighlightRequest) -> dict[str, Any]:
    _ensure_loaded()
    order = DataStore._parse_order_id(req.order_id)
    if order is None:
        raise HTTPException(400, f"Invalid order id: {req.order_id!r}")
    result = highlight_for_alarm(
        _state["graph"],
        _state["store"],
        order,
        variable=req.variable,
        process=_state["process"],
    )
    if result.get("error") and "alarm" not in result:
        raise HTTPException(404, result["error"])
    return result


@app.post("/api/investigate/stream")
async def investigate_stream(req: InvestigateRequest) -> StreamingResponse:
    """SSE stream with heartbeats while the model call is in progress."""
    _ensure_loaded()
    order = DataStore._parse_order_id(req.order_id)
    if order is None:
        raise HTTPException(400, f"Invalid order id: {req.order_id!r}")
    if order not in _state["store"].alarms:
        raise HTTPException(404, f"No alarm with Order id {order}")
    if not using_ollama() and not os.getenv("NVIDIA_API_KEY"):
        raise HTTPException(
            503,
            "No LLM configured. Set OLLAMA_LOCAL_PATH or NVIDIA_API_KEY in .env "
            "and restart the server.",
        )

    store = _state["store"]
    model = req.model
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    print(f"[cf_viz] investigate stream start order={order}", flush=True)

    def worker() -> None:
        try:
            for event in stream_investigate(order, store, model=model):
                fut = asyncio.run_coroutine_threadsafe(queue.put(event), loop)
                fut.result()
        except Exception as exc:  # noqa: BLE001
            print(f"[cf_viz] investigate worker error: {exc}", flush=True)
            fut = asyncio.run_coroutine_threadsafe(
                queue.put({"type": "error", "message": str(exc)}),
                loop,
            )
            fut.result()
        finally:
            fut = asyncio.run_coroutine_threadsafe(queue.put(None), loop)
            fut.result()
            print(f"[cf_viz] investigate stream end order={order}", flush=True)

    threading.Thread(target=worker, daemon=True, name=f"investigate-{order}").start()

    async def event_gen():
        idle_rounds = 0
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=12.0)
            except asyncio.TimeoutError:
                idle_rounds += 1
                msg = (
                    f"Still waiting on {llm_backend_label()}… "
                    f"({idle_rounds * 12}s). First tool-call can take 1–3 minutes."
                )
                yield f"data: {json.dumps({'type': 'status', 'message': msg})}\n\n"
                continue
            if item is None:
                break
            idle_rounds = 0
            yield f"data: {json.dumps(item, default=str)}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/alarms")
def list_alarms(limit: int = Query(50, ge=1, le=500)) -> dict[str, Any]:
    _ensure_loaded()
    store: DataStore = _state["store"]
    items = []
    for order in sorted(store.alarms)[:limit]:
        a = store.alarms[order]
        items.append(
            {"order": a.order, "category": a.category, "location": a.location}
        )
    return {"count": len(store.alarms), "items": items}
