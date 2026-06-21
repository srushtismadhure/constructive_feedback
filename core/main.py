"""Mars Construction — single-file Modal + FastAPI backend.

Everything lives here so Modal mounts one file and finds all app/function
definitions. modal_config.py and agent.py are NOT needed in the container.

Run from repo root:
    python3 -m modal serve core/main.py      # dev
    python3 -m modal deploy core/main.py     # prod
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Literal

# main.py sits in core/ next to orchestration/ and robot_env/. Under `modal serve`
# (run with -m) the repo root is on the path, not core/, so add this file's own dir
# — that makes `add_local_python_source("orchestration")` and the function imports
# below resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

import modal
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Modal infrastructure — image, app, shared queue
# ---------------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libegl1", "libgl1", "libgles2", "libglfw3")
    .pip_install(
        "fastapi>=0.115.0",
        "pydantic>=2.0.0",
        "uvicorn[standard]>=0.30.0",
        "httpx>=0.27.0",
        "python-dotenv>=1.0.0",
        "langgraph>=0.2.0",
        "openai>=1.30.0",
        "numpy>=1.26.0",
        "mujoco>=3.2.0",
        "Pillow>=10.0.0",
    )
    .add_local_dir("robot_env", remote_path="/root/robot_env", copy=True)
    # Non-copied sources must be the final image operation in current Modal.
    .add_local_python_source("orchestration")
)

app = modal.App("mars-construction", image=image)

# Partitioned by simulation_id so each run has its own event lane.
events_queue = modal.Queue.from_name("construction-events", create_if_missing=True)
# The API and worker may run in different containers; keep only compact status
# metadata and the latest JPEG in Modal's shared dictionary.
live_sessions = modal.Dict.from_name("atomz-live-sessions", create_if_missing=True)

HABITAT_TYPES = frozenset(("regolith_dome", "ellipsoid_habitat"))
HabitatType = Literal["regolith_dome", "ellipsoid_habitat"]


def _live_status_key(simulation_id: str) -> str:
    return f"status:{simulation_id}"


def _live_frame_key(simulation_id: str) -> str:
    return f"frame:{simulation_id}"


def _put_live_status(simulation_id: str, **values: object) -> None:
    key = _live_status_key(simulation_id)
    previous = live_sessions.get(key, {})
    live_sessions.put(key, {**previous, **values, "updated_at": time.time()})


async def _put_live_status_async(simulation_id: str, **values: object) -> None:
    """Async variant for FastAPI request handlers."""
    key = _live_status_key(simulation_id)
    previous = await live_sessions.get.aio(key, {})
    await live_sessions.put.aio(key, {**previous, **values, "updated_at": time.time()})


@app.function(image=image, gpu="T4", timeout=900, max_containers=4)
def run_live_habitat(simulation_id: str, habitat_type: HabitatType) -> dict:
    """Own one isolated headless MuJoCo construction session on a GPU worker."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    sys.path.insert(0, "/root/robot_env")
    from io import BytesIO
    from PIL import Image
    from live_runner import create_runner

    runner = None
    try:
        _put_live_status(simulation_id, state="running", progress=0, habitat_type=habitat_type, error=None)
        runner = create_runner(habitat_type)
        # 10 FPS JPEG feed. Physics advances once each published frame so every
        # browser sees a coherent, deterministic sequence for its own worker.
        while True:
            status = runner.step()
            frame = runner.render()
            output = BytesIO()
            Image.fromarray(frame).save(output, format="JPEG", quality=85, optimize=True)
            live_sessions.put(_live_frame_key(simulation_id), output.getvalue())
            if live_sessions.get(_live_status_key(simulation_id), {}).get("cancel_requested"):
                _put_live_status(simulation_id, state="cancelled", progress=status.progress)
                break
            _put_live_status(simulation_id, state="complete" if status.complete else "running", progress=status.progress)
            if status.complete:
                break
            time.sleep(0.1)
    except Exception as exc:
        print(f"[live simulation] {type(exc).__name__}: {exc}")
        _put_live_status(simulation_id, state="error", error=f"{type(exc).__name__}: {exc}")
    finally:
        if runner is not None:
            runner.close()
    return live_sessions.get(_live_status_key(simulation_id), {})

# ---------------------------------------------------------------------------
# Orchestration agent — runs on Modal, streams events to the queue
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    timeout=600,
    secrets=[modal.Secret.from_name("fireworks-api-key")],
)
def run_construction_agent(
    blueprint_id: str,
    simulation_id: str,
    coordinator_mode: str = "llm",
    seed: int = 42,
) -> dict:
    """Run the LangGraph orchestration loop and stream events to the queue."""
    from orchestration.contracts import Robot, RobotRegistry, SiteData
    from orchestration.env_interface import MockEnv
    from orchestration.graph import build_graph

    registry = RobotRegistry(
        robots=[
            Robot(id="excavator-1", role="excavator", capabilities=["excavate"], position=(1, 1)),
            Robot(id="excavator-2", role="excavator", capabilities=["excavate"], position=(6, 6)),
            Robot(id="hauler-1",    role="hauler",    capabilities=["haul", "pickup"], position=(2, 2)),
            Robot(id="hauler-2",    role="hauler",    capabilities=["haul", "pickup"], position=(5, 5)),
            Robot(id="welder-1",    role="welder",    capabilities=["weld", "place"], position=(3, 1)),
            Robot(id="welder-2",    role="welder",    capabilities=["weld", "place"], position=(4, 6)),
        ]
    )

    env = MockEnv()
    obs = env.reset(blueprint_id=blueprint_id, seed=seed)
    site = SiteData(
        terrain=obs.terrain,
        buildable_mask=[[True] * len(obs.terrain[0]) for _ in obs.terrain],
        hazard_zones=obs.hazards,
        resource_nodes=[],
    )

    initial_state = {
        "blueprint_id": blueprint_id,
        "blueprint": None,
        "site": site,
        "task_graph": None,
        "registry": registry,
        "inventory": dict(obs.resources),
        "pending_assignments": None,
        "last_trigger": None,
        "last_observation": obs,
        "step": 0,
        "done": False,
        "coordinator_mode": coordinator_mode,
        "validate_retries": 0,
        "event_log": [],
        "_pending_status_reports": [],
    }

    graph = build_graph(env)
    seen_events: set[int] = set()
    final_state = initial_state
    result: dict = {}
    _WALL_CAP = 540
    t_start = time.monotonic()

    try:
        for step_output in graph.stream(initial_state, config={"recursion_limit": 500}):
            for node_name, node_state in step_output.items():
                final_state = {**final_state, **node_state}
                log = node_state.get("event_log") or []
                for i, entry in enumerate(log):
                    if i not in seen_events:
                        seen_events.add(i)
                        payload: dict = {
                            "type": "step",
                            "node": node_name,
                            "event": entry,
                            "step": node_state.get("step", final_state.get("step", 0)),
                        }
                        batch = node_state.get("pending_assignments")
                        if batch and getattr(batch, "llm_fallback", False):
                            payload["warning"] = batch.reasoning
                        events_queue.put(payload, partition=simulation_id)

            if time.monotonic() - t_start > _WALL_CAP:
                events_queue.put(
                    {"type": "warning", "event": "Wall-clock cap reached; agent stopped early"},
                    partition=simulation_id,
                )
                break

        task_graph = final_state.get("task_graph")
        tasks = task_graph.tasks if task_graph else []
        done_count = sum(1 for t in tasks if t.status == "done")
        total = len(tasks)
        result = {
            "simulation_id": simulation_id,
            "blueprint_id": blueprint_id,
            "status": "complete" if final_state.get("done") else "incomplete",
            "completion_pct": round(100 * done_count / total) if total else 0,
            "steps": final_state.get("step", 0),
        }

    except Exception as exc:
        print(f"[agent] {type(exc).__name__}: {exc}")
        result = {
            "simulation_id": simulation_id,
            "blueprint_id": blueprint_id,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "completion_pct": 0,
            "steps": 0,
        }

    finally:
        if not result:
            result = {"simulation_id": simulation_id, "blueprint_id": blueprint_id,
                      "status": "error", "completion_pct": 0, "steps": 0}
        events_queue.put({"type": "done", **result}, partition=simulation_id)
        print(f"[agent] finished: {result}")

    return result


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

web_app = FastAPI(title="Mars Construction API", version="0.1.0")

web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class StartSimulationRequest(BaseModel):
    # habitat_type selects the Atomz live MuJoCo workflow.  blueprint_id is
    # retained for the existing orchestration workflow.
    habitat_type: HabitatType | None = None
    blueprint_id: str | None = None
    coordinator_mode: str = "llm"
    seed: int = 42


class StartSimulationResponse(BaseModel):
    simulation_id: str
    blueprint_id: str | None = None
    habitat_type: HabitatType | None = None
    status: str
    message: str
    status_url: str | None = None
    frame_url: str | None = None
    cancel_url: str | None = None


class SimulationStatusResponse(BaseModel):
    simulation_id: str
    status: str
    completion_pct: int | None = None
    steps: int | None = None
    habitat_type: HabitatType | None = None
    progress: int | None = None
    error: str | None = None


_simulations: dict[str, dict] = {}


@web_app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "mars-construction-api"}


@web_app.post("/simulation/start", response_model=StartSimulationResponse)
async def start_simulation(req: StartSimulationRequest) -> StartSimulationResponse:
    simulation_id = str(uuid.uuid4())
    if req.habitat_type:
        # Pydantic already validates the literal; this guard makes the contract
        # explicit for callers that reach the handler through tests/mocks.
        if req.habitat_type not in HABITAT_TYPES:
            raise HTTPException(status_code=422, detail="Unsupported habitat type")
        await _put_live_status_async(
            simulation_id,
            state="starting",
            progress=0,
            habitat_type=req.habitat_type,
            cancel_requested=False,
            error=None,
        )
        try:
            await run_live_habitat.spawn.aio(simulation_id=simulation_id, habitat_type=req.habitat_type)
        except Exception as exc:
            await _put_live_status_async(simulation_id, state="error", error=f"Unable to start worker: {exc}")
        base = f"/simulation/{simulation_id}"
        return StartSimulationResponse(
            simulation_id=simulation_id,
            habitat_type=req.habitat_type,
            status="starting",
            message="Live habitat worker starting.",
            status_url=base,
            frame_url=f"{base}/frames",
            cancel_url=f"{base}/cancel",
        )

    if not req.blueprint_id:
        raise HTTPException(status_code=422, detail="Provide habitat_type or blueprint_id")
    try:
        fc = run_construction_agent.spawn(
            blueprint_id=req.blueprint_id,
            simulation_id=simulation_id,
            coordinator_mode=req.coordinator_mode,
            seed=req.seed,
        )
        _simulations[simulation_id] = {
            "blueprint_id": req.blueprint_id,
            "status": "running",
            "function_call": fc,
        }
    except Exception as exc:
        print(f"[api] spawn failed: {exc}")
        _simulations[simulation_id] = {
            "blueprint_id": req.blueprint_id,
            "status": "error",
            "function_call": None,
        }

    return StartSimulationResponse(
        simulation_id=simulation_id,
        blueprint_id=req.blueprint_id,
        status="running",
        message=f"Agent spawned. Stream events at /simulation/{simulation_id}/stream",
    )


@web_app.get("/simulation/{simulation_id}/stream")
async def stream_simulation(simulation_id: str):
    """SSE endpoint. Reads from the Modal Queue and forwards to the client."""
    async def event_generator():
        while True:
            try:
                item = await events_queue.get.aio(partition=simulation_id, timeout=10)
                yield f"data: {json.dumps(item)}\n\n"
                if item.get("type") == "done":
                    break
            except Exception:
                yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@web_app.get("/simulation/{simulation_id}", response_model=SimulationStatusResponse)
async def get_simulation_status(simulation_id: str) -> SimulationStatusResponse:
    live = await live_sessions.get.aio(_live_status_key(simulation_id))
    if live:
        return SimulationStatusResponse(
            simulation_id=simulation_id,
            status=live.get("state", "starting"),
            habitat_type=live.get("habitat_type"),
            progress=live.get("progress", 0),
            error=live.get("error"),
        )
    sim = _simulations.get(simulation_id)
    if not sim:
        raise HTTPException(status_code=404, detail="Simulation not found")

    fc = sim.get("function_call")
    if fc and sim["status"] == "running":
        try:
            result = fc.get(timeout=0)
            sim["status"] = result.get("status", "complete")
            sim["completion_pct"] = result.get("completion_pct")
            sim["steps"] = result.get("steps")
        except Exception:
            pass

    return SimulationStatusResponse(
        simulation_id=simulation_id,
        status=sim["status"],
        completion_pct=sim.get("completion_pct"),
        steps=sim.get("steps"),
    )


@web_app.get("/simulation/{simulation_id}/frames")
async def stream_live_frames(simulation_id: str):
    """Multipart JPEG feed. The last frame remains available after completion."""
    if not await live_sessions.get.aio(_live_status_key(simulation_id)):
        raise HTTPException(status_code=404, detail="Simulation not found")

    async def frame_generator():
        last_frame: bytes | None = None
        while True:
            frame = await live_sessions.get.aio(_live_frame_key(simulation_id))
            if frame and frame != last_frame:
                last_frame = frame
                yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n"
            state = (await live_sessions.get.aio(_live_status_key(simulation_id), {})).get("state")
            if state in {"complete", "cancelled", "error"}:
                # Emit the final frame once, then let browsers retain it.
                break
            import asyncio
            await asyncio.sleep(0.1)

    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")


@web_app.post("/simulation/{simulation_id}/cancel")
async def cancel_live_simulation(simulation_id: str) -> dict:
    status = await live_sessions.get.aio(_live_status_key(simulation_id))
    if not status:
        raise HTTPException(status_code=404, detail="Simulation not found")
    if status.get("state") not in {"complete", "cancelled", "error"}:
        await _put_live_status_async(simulation_id, cancel_requested=True)
    latest = await live_sessions.get.aio(_live_status_key(simulation_id), {})
    return {"simulation_id": simulation_id, "status": latest.get("state")}


@web_app.get("/blueprints")
async def list_blueprints() -> dict:
    return {"blueprints": [
        {"id": "habitat-dome",        "name": "Habitat Dome",           "category": "habitat"},
        {"id": "research-lab",        "name": "Research Laboratory",    "category": "research"},
        {"id": "greenhouse-module",   "name": "Greenhouse Module",      "category": "production"},
        {"id": "solar-array",         "name": "Solar Array Station",    "category": "infrastructure"},
        {"id": "underground-shelter", "name": "Underground Shelter",    "category": "habitat"},
        {"id": "comm-tower",          "name": "Communication Tower",    "category": "infrastructure"},
        {"id": "water-extractor",     "name": "Water Extraction Plant", "category": "production"},
        {"id": "landing-pad",         "name": "Landing Pad",            "category": "infrastructure"},
    ]}


# ---------------------------------------------------------------------------
# Mount FastAPI onto Modal
# ---------------------------------------------------------------------------

@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    return web_app


# ---------------------------------------------------------------------------
# Local dev:  python3 main.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(web_app, host="0.0.0.0", port=8000, reload=True)
