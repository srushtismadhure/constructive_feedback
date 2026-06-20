"""
Mars Construction — Modal + FastAPI backend.

Serves as the orchestration layer between the Next.js frontend and the
AI construction agent. The FastAPI app is deployed via Modal's ASGI support.
"""

import uuid
import modal
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Modal image — install Python dependencies in the container
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "fastapi>=0.115.0",
        "pydantic>=2.0.0",
        "httpx>=0.27.0",
        "python-dotenv>=1.0.0",
        "uvicorn[standard]>=0.30.0",
    )
)

app = modal.App("mars-construction-api", image=image)

# ---------------------------------------------------------------------------
# FastAPI app definition
# ---------------------------------------------------------------------------
web_app = FastAPI(title="Mars Construction API", version="0.1.0")

web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: restrict to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class StartSimulationRequest(BaseModel):
    blueprint_id: str


class StartSimulationResponse(BaseModel):
    simulation_id: str
    blueprint_id: str
    status: str
    message: str


class SimulationStatusResponse(BaseModel):
    simulation_id: str
    status: str
    stream_url: str | None = None


# ---------------------------------------------------------------------------
# In-memory simulation registry (replace with a real DB / Redis later)
# ---------------------------------------------------------------------------
_simulations: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@web_app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "mars-construction-api"}


@web_app.post("/simulation/start", response_model=StartSimulationResponse)
async def start_simulation(req: StartSimulationRequest) -> StartSimulationResponse:
    """
    Receive a blueprint selection from the frontend and kick off the
    AI construction agent on Modal.
    """
    simulation_id = str(uuid.uuid4())

    # Spawn the agent asynchronously (fire-and-forget for now).
    # TODO: pass a callback / websocket so the agent can push status updates.
    try:
        from agent import run_construction_agent  # imported inside Modal container

        run_construction_agent.spawn(
            blueprint_id=req.blueprint_id,
            simulation_id=simulation_id,
        )
    except Exception as exc:
        # Agent not available locally — log and continue so the frontend can
        # still navigate to the placeholder simulation page.
        print(f"[warn] Could not spawn agent: {exc}")

    _simulations[simulation_id] = {
        "blueprint_id": req.blueprint_id,
        "status": "initializing",
        "stream_url": None,
    }

    return StartSimulationResponse(
        simulation_id=simulation_id,
        blueprint_id=req.blueprint_id,
        status="initializing",
        message="Construction agent dispatched. Simulation is initializing.",
    )


@web_app.get("/simulation/{simulation_id}", response_model=SimulationStatusResponse)
async def get_simulation_status(simulation_id: str) -> SimulationStatusResponse:
    """Poll the current status and stream URL for a running simulation."""
    sim = _simulations.get(simulation_id)
    if not sim:
        raise HTTPException(status_code=404, detail="Simulation not found")

    return SimulationStatusResponse(
        simulation_id=simulation_id,
        status=sim["status"],
        stream_url=sim.get("stream_url"),
    )


@web_app.get("/blueprints")
async def list_blueprints() -> dict:
    """
    Return available blueprint presets. The frontend has its own static list;
    this endpoint is here for parity / future dynamic blueprint loading.
    """
    blueprints = [
        {"id": "habitat-dome",       "name": "Habitat Dome",            "category": "habitat"},
        {"id": "research-lab",       "name": "Research Laboratory",     "category": "research"},
        {"id": "greenhouse-module",  "name": "Greenhouse Module",       "category": "production"},
        {"id": "solar-array",        "name": "Solar Array Station",     "category": "infrastructure"},
        {"id": "underground-shelter","name": "Underground Shelter",     "category": "habitat"},
        {"id": "comm-tower",         "name": "Communication Tower",     "category": "infrastructure"},
        {"id": "water-extractor",    "name": "Water Extraction Plant",  "category": "production"},
        {"id": "landing-pad",        "name": "Landing Pad",             "category": "infrastructure"},
    ]
    return {"blueprints": blueprints}


# ---------------------------------------------------------------------------
# Mount FastAPI onto Modal as an ASGI web endpoint
# ---------------------------------------------------------------------------
@app.function()
@modal.asgi_app()
def fastapi_app():
    return web_app


# ---------------------------------------------------------------------------
# Local dev entry point:  python main.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(web_app, host="0.0.0.0", port=8000, reload=True)
