# Mars Construction Multi-Agent System — Project Memory

## Overview
Autonomous Mars construction orchestrated by a multi-agent LLM system. Two decoupled pieces:
- **Environment**: MuJoCo sim — exposes Gym-style interface
- **Brain**: LangGraph orchestration — plans, assigns, validates, dispatches, replans

---

## Repo Layout

```
constructive_feedback/
├── core/                       ← ALL Python, one uv env (pyproject.toml here)
│   ├── pyproject.toml          ← langgraph, pydantic, openai, mujoco, numpy, modal, fastapi… (package=false)
│   ├── main.py                 ← Single-file Modal + FastAPI backend (the orchestration API)
│   ├── .env                    ← FIREWORKS_API_KEY, FRONTEND_ORIGIN (gitignored)
│   ├── orchestration/          ← The Brain
│   │   ├── __init__.py
│   │   ├── contracts.py        ← All Pydantic schemas
│   │   ├── env_interface.py    ← EnvInterface Protocol + MockEnv stub
│   │   ├── blueprints/         ← JSON blueprint files
│   │   │   ├── habitat_dome.json   (8 components: 2 foundation, 3 wall, 1 roof, 2 panel)
│   │   │   ├── comm_tower.json
│   │   │   └── landing_pad.json
│   │   ├── blueprint_parser.py ← JSON → ParsedBlueprint + topo sort
│   │   ├── sequencer.py        ← ParsedBlueprint → TaskGraph
│   │   ├── coordinator.py      ← greedy_coordinator + llm_coordinator (Fireworks MiniMax M3)
│   │   ├── validator.py        ← pre-dispatch sanity checks
│   │   ├── monitor.py          ← StatusReport → state update + ReplanTrigger
│   │   ├── graph.py            ← LangGraph state machine (main loop)
│   │   └── mujoco_adapter.py   ← RoverEnvAdapter: robot_env → EnvInterface (the sim seam)
│   ├── robot_env/              ← Mars MuJoCo sim
│   │   ├── mars_scene.xml      ← terrain + differential-drive rover on a flat drive plane
│   │   ├── hud_mujoco_bridge.py ← MarsMujocoBridge (HUD RobotBridge) + drive-to-goal reward
│   │   └── run_hud_demo.py     ← passive-viewer demo driving the bridge
│   └── tests/
│       └── smoke_test.py       ← end-to-end integration check (run: uv run python tests/smoke_test.py)
├── frontend/                   ← Next.js app (standalone; talks to backend over HTTP)
└── robot_training/             ← Git submodule (worldsim-template) — RL training/eval reference
```

All Python (`main.py`, `orchestration/`, `robot_env/`) sits under `core/` and shares one
interpreter: the API runs the Brain and the adapter imports the sim in-process. Imports are
top-level (`from orchestration...`, robot_env via sys.path) — `core/` is the package root,
put on the path by the test (`parents[1]`) and by `main.py` (`parent`, for `add_local_python_source`).

---

## Key Architecture Decisions

### Single-file Modal pattern
**CRITICAL**: Everything Modal-related lives in `core/main.py`. Modal only mounts the entry-point file + declared `add_local_python_source(...)` packages. Splitting into `modal_config.py` + `agent.py` caused `ModuleNotFoundError` in container. Never split Modal infra across files again.

### LLM: Fireworks MiniMax M3 (NOT Anthropic)
- Base URL: `https://api.fireworks.ai/inference/v1`
- Model: `accounts/fireworks/models/minimax-m3`
- Uses OpenAI-compatible client (`openai` SDK, custom `base_url`)
- Tool-call format: `call.function.arguments` is a JSON **string** (parse with `json.loads`)
- Falls back to greedy coordinator if API call fails or key missing

### Coordinator modes
- `"greedy"`: nearest idle eligible robot, O(n×m), no LLM
- `"llm"`: MiniMax M3 via Fireworks tool-use, falls back to greedy on error

### LangGraph state machine flow
```
init → coordinate → validate →(pass)→ dispatch → monitor → check_done
                            ↗(fail, retry<3)                    |
                   coordinate ← replan ←──────────(trigger)────┘
                                                ↓(all done)
                                               END
```
- `_MAX_VALIDATE_RETRIES = 3`
- `_MAX_STEPS = 200`
- `recursion_limit = 500` in graph.invoke config

### EnvInterface seam
```python
class EnvInterface(Protocol):
    def reset(self, blueprint_id: str, seed: int = 42) -> Observation: ...
    def step(self, action: Action) -> tuple[Observation, float, bool, dict]: ...
```
`MockEnv` in `env_interface.py` is the no-sim stub; `RoverEnvAdapter` in
`mujoco_adapter.py` is the real MuJoCo implementation (see below).

### robot_env + RoverEnvAdapter (MuJoCo integration)

**robot_env** is a standalone MuJoCo sim (NOT Newton, NOT the `scenes/` layout). One
differential-drive rover; `MarsMujocoBridge` is a HUD `RobotBridge`:
- action = `[forward_speed, turn_speed]` ∈ [-1,1] → differential wheel torques
- state = `[x, y, z, yaw, vx, vy, vz, yaw_rate]` (8) + a 256×256 RGB frame
- **reward** = drive-to-goal: dense progress (`prev_dist - dist`) + `SUCCESS_BONUS` on
  arrival within `GOAL_TOLERANCE` (0.4 m). `set_goal(x, y)` re-points the goal without
  re-loading the scene, so one bridge serves many sequential goals.

**RoverEnvAdapter** (`core/orchestration/mujoco_adapter.py`) wraps the bridge as an
`EnvInterface` so the Brain can drive it:
- **One rover ≠ a fleet.** The Brain plans for 6 robots; the sim has 1. The adapter
  treats the rover as the shared embodiment — it runs each dispatched Action in order,
  driving to that task's grid cell. Fleet positions stay in the Brain's registry.
- **grid → world**: cell `(gx, gy)` → `(origin + gx*cell, origin + gy*cell)`, default
  `origin=(-3.5,-3.5)`, `cell=1.0 m` (8×8 grid lands inside the ±8 m drive plane).
- **Action → maneuver**: `step()` sets the goal, runs a pure-pursuit controller
  (pivot in place until facing the goal, then drive) for up to `max_drive_steps` (220).
  Reached → `place`/`weld`/`excavate` marks the cell built. Not reached →
  `info["rejection_reason"]` set → the Brain marks the task failed → replan.
- **async note**: `MarsMujocoBridge.reset` is async (does only sync work); the adapter
  calls it via `asyncio.run` (safe — `graph.invoke` is sync, no running loop).
- **what the graph actually reads** from `step()`: `info["rejection_reason"]` and
  `reward`. The returned `Observation`/`done` are built for completeness, not consumed.

**One core, two front doors**: `MarsMujocoBridge` is the single physics+reward core.
Brain → `RoverEnvAdapter` (EnvInterface); VLA policy / `hud eval` → HUD `RobotEndpoint`.

---

## Role Assignment Rules (encoded in parser, not JSON)

| Component type | Required role |
|---|---|
| foundation | excavator |
| wall, roof | welder |
| panel | welder (place action) |
| material transport | hauler |

## Task Expansion (sequencer.py)

| Component | Tasks generated |
|---|---|
| foundation | excavate (excavator) |
| wall/roof | haul (hauler) → weld (welder) |
| panel | haul (hauler) → place (welder) |

## Action command mapping (dispatch node)
```python
_CMD_MAP = {"haul": "pickup", "excavate": "excavate", "place": "place", "weld": "weld"}
```

---

## Modal Infrastructure

### Image definition (in main.py)
```python
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("fastapi>=0.115.0", "pydantic>=2.0.0", "uvicorn[standard]>=0.30.0",
                 "httpx>=0.27.0", "python-dotenv>=1.0.0", "langgraph>=0.2.0",
                 "openai>=1.30.0", "numpy>=1.26.0")
    .add_local_python_source("orchestration")  # mounts the orchestration package
)
```
main.py sits in `core/` next to `orchestration/`, but `modal serve/deploy` runs with `-m`
(repo root on the path, not `core/`), so main.py prepends its own dir to `sys.path`
(`Path(__file__).resolve().parent`) before this runs — that's what lets
`add_local_python_source("orchestration")` resolve. The backend still uses `MockEnv` (no
`robot_env`/`mujoco` in the image yet; add them here + to the image to drive the real sim).

### Modal app
- App name: `"mars-construction"`
- Queue name: `"construction-events"` (partitioned by `simulation_id`)
- Modal secret name: `"fireworks-api-key"` (contains `FIREWORKS_API_KEY`)
- ASGI function: `fastapi_app` (mounts FastAPI at Modal URL)
- Worker function: `run_construction_agent` (timeout=600s)

### API Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/simulation/start` | Spawn agent, return `simulation_id` immediately |
| GET | `/simulation/{id}/stream` | SSE stream of events from Modal Queue |
| GET | `/simulation/{id}` | Poll status |
| GET | `/blueprints` | List available blueprints |

### Event types on queue
- `{"type": "step", "node": "...", "event": "...", "step": N}` — per-node update
- `{"type": "heartbeat"}` — keep-alive (queue timeout)
- `{"type": "warning", "event": "..."}` — wall-clock cap or LLM fallback
- `{"type": "done", "status": "...", "completion_pct": N, "steps": N}` — final

---

## Secrets and Environment

### core/.env (local, gitignored)
```
FIREWORKS_API_KEY=fw_...
FRONTEND_ORIGIN=http://localhost:3000
```

### Modal secret (one-time setup)
```bash
python3 -m modal secret create fireworks-api-key FIREWORKS_API_KEY=fw_...
```

---

## Robot Fleet (default)
```python
Robot(id="excavator-1", role="excavator", capabilities=["excavate"], position=(1, 1))
Robot(id="excavator-2", role="excavator", capabilities=["excavate"], position=(6, 6))
Robot(id="hauler-1",    role="hauler",    capabilities=["haul", "pickup"], position=(2, 2))
Robot(id="hauler-2",    role="hauler",    capabilities=["haul", "pickup"], position=(5, 5))
Robot(id="welder-1",    role="welder",    capabilities=["weld", "place"], position=(3, 1))
Robot(id="welder-2",    role="welder",    capabilities=["weld", "place"], position=(4, 6))
```

---

## Verified Tests (passing)
1. **Greedy pipeline**: `habitat-dome` → 14 tasks, 14 steps, all done
2. **Replan test**: injected `excavate` failure → replan triggered → build recovered in 15 steps
3. **Rover controller**: `RoverEnvAdapter` drives to grid targets across the plane (6/6 reached)
4. **Brain ↔ rover e2e**: `run_orchestration("habitat-dome", env=RoverEnvAdapter(), ...)` →
   14 tasks done in 14 steps; rover physically drives to each target in MuJoCo.

Run all of the above from `core/`: `cd core && uv sync && uv run python tests/smoke_test.py`.

---

## How to Run

### Dev (hot-reload, run from repo root)
```bash
python3 -m modal serve core/main.py
```
URL: `https://constructive-feedback--mars-construction-fastapi-app-dev.modal.run`

### Test greedy (no LLM needed)
```bash
curl -X POST <URL>/simulation/start \
  -H "Content-Type: application/json" \
  -d '{"blueprint_id": "habitat-dome", "coordinator_mode": "greedy"}'

curl -N <URL>/simulation/<sim_id>/stream
```

### Test LLM mode
```bash
curl -X POST <URL>/simulation/start \
  -H "Content-Type: application/json" \
  -d '{"blueprint_id": "habitat-dome", "coordinator_mode": "llm"}'
```
Look for `[MiniMax M3]` in reasoning events. `FALLBACK:` prefix means key not reaching container.

### Deploy to prod
```bash
python3 -m modal deploy core/main.py
```

### Local FastAPI only (no Modal)
```bash
cd core && make dev      # uvicorn main:web_app --reload
```
