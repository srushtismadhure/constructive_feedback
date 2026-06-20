# constructive_feedback

We are building a verifiable multi-agent construction RL environment + a swarm policy trained in it + a demo wrapper.

Users pick a Mars construction blueprint in the web UI, which dispatches an AI agent (via Modal) to run a physics-based build simulation. The simulation runs in the `robot_training` environment (Newton/MuJoCo), and the live output streams back to the browser.

---

## Repository Layout

```
constructive_feedback/
├── web/
│   ├── frontend/   Next.js (TypeScript) — blueprint selector UI + simulation viewer
│   └── backend/    FastAPI + Modal — agent orchestration API
└── robot_training/ Git submodule — worldsim-template RL environment (Newton physics)
```

### `web/frontend`

Next.js 15 / TypeScript / Tailwind CSS.

The landing page renders a grid of Mars construction blueprint cards (habitat domes, research labs, greenhouses, etc.). Clicking a card fires a request to the backend and navigates to the simulation viewer, which will stream live video from the physics environment once the pipeline is wired up.

**Key files**
- `src/app/page.tsx` — blueprint card grid
- `src/app/simulation/[id]/page.tsx` — live simulation viewer (placeholder)
- `src/components/BlueprintCard.tsx` — card component
- `src/data/blueprints.ts` — preset definitions

### `web/backend`

Python 3.11 / FastAPI / [Modal](https://modal.com), managed by `uv`.

Receives blueprint selections from the frontend, spawns an AI agent on Modal that configures and launches the physics simulation, and exposes status/stream endpoints back to the UI.

**Key files**
- `main.py` — Modal ASGI app wrapping FastAPI; defines `/simulation/start` and `/simulation/:id`
- `agent.py` — Modal function stub for the construction agent (MuJoCo orchestration TODO)

### `robot_training`

Git submodule → [`hud-evals/worldsim-template`](https://github.com/hud-evals/worldsim-template)

Newton/MuJoCo-based robotics RL environment running on the HUD SDK. Contains four manipulation tasks (`move-object`, `pick-object`, `force-grasp`, `open-drawer`) plus VLA policy eval infrastructure. The construction agent in `web/backend/agent.py` will delegate simulation tasks here.

Requires Python 3.12 and a HUD API key. See `robot_training/README.md` for full setup and task documentation.

---

## Getting Started

### 0. Clone (with submodule)

```bash
git clone --recurse-submodules https://github.com/srushtismadhure/constructive_feedback.git
cd constructive_feedback

# If you already cloned without --recurse-submodules:
git submodule update --init --recursive
```

### 1. Frontend

```bash
cd web/frontend
make install   # npm install
make dev       # starts Next.js on http://localhost:3000
```

Copy `.env.local.example` to `.env.local` and set `NEXT_PUBLIC_API_URL` if the backend isn't on `localhost:8000`.

| Command | Description |
|---|---|
| `make dev` | Start dev server with hot reload |
| `make build` | Production build |
| `make lint` | Run ESLint |
| `make clean` | Remove `.next/` and `node_modules/` |

### 2. Backend

Requires Python 3.11+ and [`uv`](https://docs.astral.sh/uv/).

```bash
cd web/backend
make install   # uv sync
make dev       # FastAPI on http://localhost:8000 with --reload
```

To deploy to Modal:

```bash
# Authenticate once
modal token new

make deploy    # modal deploy main.py
```

Copy `.env.example` to `.env` and fill in your Modal credentials and frontend origin.

| Command | Description |
|---|---|
| `make dev` | Local FastAPI server with auto-reload |
| `make deploy` | Deploy to Modal |
| `make lint` | Ruff lint |
| `make format` | Ruff format |
| `make check` | Pyright type check |
| `make clean` | Remove `.venv/`, `__pycache__/`, caches |

### 3. Robot Training Environment

Requires Python 3.12 and a [HUD API key](https://hud.so).

```bash
cd robot_training
uv sync                              # installs all deps incl. bundled Newton wheel
source .venv/bin/activate
hud set HUD_API_KEY=your-key-here

# Readiness check (~1 min first run, compiles Warp)
python scripts/check_setup.py

# Run an example agent
python examples/example_agent.py
```

See `robot_training/README.md` for the full task list, VLA policy eval, and scene authoring guide.

---

## Updating the Submodule

```bash
# Pull latest upstream changes into robot_training/
git submodule update --remote robot_training
git add robot_training
git commit -m "chore: bump robot_training submodule"
```
