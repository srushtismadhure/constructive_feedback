# Mars Construction Multi-Agent System — Project Memory

## Overview
Hackathon project: autonomous Mars construction orchestrated by a multi-agent LLM system. Two decoupled pieces:
- **Environment**: MuJoCo sim (teammate owns) — exposes Gym-style interface
- **Brain**: LangGraph orchestration (this repo) — plans, assigns, validates, dispatches, replans

**User owns**: orchestration layer only. Teammate owns MuJoCo sim + frontend.

---

## Repo Layout

```
constructive_feedback/
├── orchestration/              ← The Brain (fully built & tested)
│   ├── __init__.py
│   ├── contracts.py            ← All Pydantic schemas
│   ├── env_interface.py        ← EnvInterface Protocol + MockEnv stub
│   ├── blueprints/             ← JSON blueprint files
│   │   ├── habitat_dome.json   (8 components: 2 foundation, 3 wall, 1 roof, 2 panel)
│   │   ├── comm_tower.json
│   │   └── landing_pad.json
│   ├── blueprint_parser.py     ← JSON → ParsedBlueprint + topo sort
│   ├── sequencer.py            ← ParsedBlueprint → TaskGraph
│   ├── coordinator.py          ← greedy_coordinator + llm_coordinator (Fireworks MiniMax M3)
│   ├── validator.py            ← pre-dispatch sanity checks
│   ├── monitor.py              ← StatusReport → state update + ReplanTrigger
│   └── graph.py                ← LangGraph state machine (main loop)
└── web/
    └── backend/
        ├── main.py             ← Single-file Modal + FastAPI backend (ALL infra here)
        ├── .env                ← FIREWORKS_API_KEY, FRONTEND_ORIGIN
        ├── .env.example
        └── pyproject.toml
```

---

## Key Architecture Decisions

### Single-file Modal pattern
**CRITICAL**: Everything Modal-related lives in `web/backend/main.py`. Modal only mounts the entry-point file + declared `add_local_python_source(...)` packages. Splitting into `modal_config.py` + `agent.py` caused `ModuleNotFoundError` in container. Never split Modal infra across files again.

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
Teammate implements this for MuJoCo. `MockEnv` in `env_interface.py` used until then.

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

### web/backend/.env (local, gitignored)
```
FIREWORKS_API_KEY=fw_...
FRONTEND_ORIGIN=http://localhost:3000
```

### Modal secret (one-time setup)
```bash
python3 -m modal secret create fireworks-api-key FIREWORKS_API_KEY=fw_...
```

---

## Robot Fleet (default, until MuJoCo adapter provides state)
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

---

## How to Run

### Dev (hot-reload, run from repo root)
```bash
python3 -m modal serve web/backend/main.py
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
python3 -m modal deploy web/backend/main.py
```

### Local FastAPI only (no Modal)
```bash
python3 web/backend/main.py
```

---

## Teammate Integration
Teammate implements `EnvInterface` for MuJoCo:
```python
class MyMuJoCoEnv:
    def reset(self, blueprint_id: str, seed: int = 42) -> Observation: ...
    def step(self, action: Action) -> tuple[Observation, float, bool, dict]: ...
```
Pass as `env=` to `run_construction_agent` or `run_orchestration`. The `info` dict from `step()` should include `"rejection_reason": str | None`.

---

## Demo Narrative
1. Show blueprint + empty Mars grid
2. Hit run → robots construct in dependency order
3. Trigger dust storm → robot goes broken
4. Show coordinator reasoning text → tasks reassigned → build recovers
5. Metrics: LLM coordinator vs greedy baseline (steps to completion)
