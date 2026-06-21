# TRAINING.md — Swarm VLA: record → SFT → RL finetune → eval

A deterministic, end-to-end pipeline that **reinforcement-finetunes a pi0.5 VLA** on the
Mars swarm pick-and-place task and runs it back in the same HUD environment. It exists to
demonstrate the RL machinery is feasible — not to beat the scripted oracle (the env is
intentionally deterministic).

The split: the policy drives only the **5-DoF arm**; navigation stays scripted inside the
bridge. Reward is the bridge's shaped placement score (approach < grasped `0.5` < placed
`1.0`). HUD has no RL trainer for robots (it's LLM/token-only), so the loop is
**reward-weighted / filtered BC (DAgger-lite)**: keep the policy's successful rollouts,
expert-relabel the failures, re-SFT, repeat.

All commands run from `core/`. Everything lives under `core/` (`robot_env/`, `train/`).

## Prerequisites

```bash
cd core
uv sync --extra robot --extra serve         # hud + openpi/0 wire (no torch) — for env/eval
# torch/lerobot are only needed where training/inference actually runs (Modal images
# install them). For LOCAL recording/training add: uv sync --extra record
```

Remote (Modal A100) needs a Modal account and an HF token secret:

```bash
modal token new
modal secret create huggingface-secret HF_TOKEN=hf_...   # write scope (used by SFT + RL)
huggingface-cli login                                     # for the local dataset push
```

Replace `<user>` below with your HF username.

## Quickstart (no GPU — validate the wiring)

```bash
# 1. the manip env end-to-end with the scripted oracle (expect score 1.0)
uv run python robot_env/swarm_manip_bridge.py

# 2. the HUD env over the openpi/0 wire with a no-op policy (expect low reward, no grasp)
uv run python robot_env/run_swarm_vla.py --noop --group 1

# 3. the RL loop's rollout/filter/relabel mechanics, no torch
uv run python train/rl_loop.py --dry-run --group 2
```

## Full pipeline

### 1. Record the dataset (Path B: one episode per rover)

```bash
uv sync --extra record    # pulls lerobot + torch
uv run python robot_env/record_dataset.py --robot swarm --cubes-per-rover 4 \
    --repo-id <user>/mars-construction-swarm --push
```

`observation.state` is proprioception only (9-dim); the privileged cube/target coords go
in a separate `godmode` column. Validate first with `--dry-run` (no lerobot needed).

### 2. SFT on Modal — pi05_base → checkpoint

Deliberately **under-train** for the demo so the baseline lands below the oracle.

```bash
modal run core/train/sft_modal.py \
    --dataset-repo <user>/mars-construction-swarm \
    --output-repo  <user>/pi05-mars-swarm-sft \
    --steps 2000
```

Trains from `lerobot/pi05_base`, uploads the checkpoint to `<user>/pi05-mars-swarm-sft`.

### 3. Baseline eval — serve the checkpoint, score it

```bash
# terminal A: serve on a Modal A100 (prints ws://HOST:PORT, stays up)
modal run core/robot_env/serve/pi05_modal_mars.py --checkpoint <user>/pi05-mars-swarm-sft

# terminal B: run the sim + loop here (CPU-only), point at that HOST:PORT
uv run python robot_env/run_swarm_vla.py --remote HOST:PORT --group 4
```

Expect a success rate < 100% (the gap RL closes).

### 4. RL finetune (filtered-BC / DAgger) on Modal

One A100 container runs the whole loop — policy inference **and** lerobot retraining each
iteration (no cross-machine tunnel to manage):

```bash
modal run core/train/rl_loop.py \
    --base-checkpoint <user>/pi05-mars-swarm-sft \
    --dataset-repo    <user>/mars-construction-swarm-rl \
    --output-repo     <user>/pi05-mars-swarm-rl \
    --iters 3 --group 2 --sft-steps 500
```

Each iteration: roll out → score → keep successes + oracle-relabel failures → re-SFT.
Prints the **per-iteration mean-reward curve** (the "RL is working" signal) and uploads the
final checkpoint to `<user>/pi05-mars-swarm-rl`.

### 5. Final eval

Same as step 3, pointing the server at the RL checkpoint:

```bash
modal run core/robot_env/serve/pi05_modal_mars.py --checkpoint <user>/pi05-mars-swarm-rl
uv run python robot_env/run_swarm_vla.py --remote HOST:PORT --group 4
```

Success rate should rise toward 100%, deterministically and repeatably (fixed seeds).

## Local (own GPU box) instead of Modal

`pi05_modal_mars.py` and `sft_modal.py` just wrap these — on a GPU box run them directly:

```bash
# serve a checkpoint over openpi/0
python robot_env/serve/policy_server_mars.py --checkpoint <user>/pi05-mars-swarm-sft --port 8000
# the wire-only no-GPU variant:
python robot_env/serve/policy_server_mars.py --noop --port 8000
```

## Layout

```
core/
├── robot_env/
│   ├── swarm_bridge.py          # 3-rover sim (get_observation per rover, added here)
│   ├── swarm_manip_bridge.py    # SwarmManipBridge + CONTRACT — the RL env (1 rover, pick→place)
│   ├── swarm_hud_env.py         # HUD Environment serving it over robot (openpi/0)
│   ├── swarm_agents.py          # NoopAgent / RemoteAgent (5-DoF)
│   ├── run_swarm_vla.py         # eval runner (--noop / --remote)
│   ├── record_dataset.py        # scripted oracle → LeRobot v3 dataset (--robot swarm)
│   └── serve/
│       ├── policy_server_mars.py   # single-camera openpi/0 policy server
│       └── pi05_modal_mars.py      # serve a checkpoint on a Modal A100
└── train/
    ├── sft_modal.py             # Modal SFT: pi05_base → checkpoint → HF
    └── rl_loop.py               # filtered-BC / DAgger loop (--dry-run locally; Modal for real)
```

## Notes / caveats

- **Determinism is intentional.** Grouped rollouts are identical, so RL can't exceed the
  oracle; the demo works by under-training SFT and letting the loop close the gap.
- **lerobot CLI flags** in `sft_modal.py` / `rl_loop.py` target the pinned lerobot commit
  (`b8ad81b…`); adjust if you bump lerobot. Checkpoints are uploaded via `huggingface_hub`
  (version-independent of lerobot's push flags).
- The single-camera inference path (`build_mars_infer`) mirrors the franka
  `build_pi05_infer` in `robot_training/serve/` — it maps `observation/image` onto the
  policy's first image slot and truncates the action chunk to the 5 arm dims.
