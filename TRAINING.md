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

The following are saved datasets/checkpoints from training during the hackathon:
- https://huggingface.co/datasets/changminbark/mars-construction-swarm
- https://huggingface.co/changminbark/pi05-mars-swarm-sft
- https://huggingface.co/changminbark/pi05-mars-swarm-rl

## Prerequisites

```bash
cd core
# Sync ALL the extras you need in ONE command — `uv sync --extra X` is NOT additive,
# so a later `uv sync --extra record` would drop hud and break the eval/wire commands.
uv sync --extra robot --extra serve --extra record   # hud + openpi/0 wire + lerobot
# Minimal env/eval only (no torch/lerobot): uv sync --extra robot --extra serve
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
uv run python robot_env/record_dataset.py --robot swarm --cubes-per-rover 4 \
    --repo-id <user>/mars-construction-swarm --push
```

`observation.state` is proprioception only (9-dim); the privileged cube/target coords go
in a separate `godmode` column. Validate first with `--dry-run` (no lerobot needed).

### 2. SFT on Modal — pi05_base → checkpoint

Deliberately **under-train** for the demo so the baseline lands below the oracle.

```bash
modal run train/sft_modal.py \
    --dataset-repo <user>/mars-construction-swarm \
    --output-repo  <user>/pi05-mars-swarm-sft \
    --steps 2000
```

Trains from `lerobot/pi05_base`, uploads the checkpoint to `<user>/pi05-mars-swarm-sft`.

### 3. Baseline eval — serve the checkpoint, score it

```bash
# terminal A: serve on a Modal A100 (prints ws://HOST:PORT, stays up)
modal run robot_env/serve/pi05_modal_mars.py --checkpoint <user>/pi05-mars-swarm-sft

# terminal B: run the sim + loop here (CPU-only), point at that HOST:PORT
uv run python robot_env/run_swarm_vla.py --remote HOST:PORT --group 4
```

Expect a success rate < 100% (the gap RL closes).

### 4. RL finetune (filtered-BC / DAgger) on Modal

One A100 container runs the whole loop — policy inference **and** lerobot retraining each
iteration (no cross-machine tunnel to manage):

```bash
modal run train/rl_loop.py \
    --base-checkpoint <user>/pi05-mars-swarm-sft \
    --dataset-repo    <user>/mars-construction-swarm \
    --output-repo     <user>/pi05-mars-swarm-rl \
    --iters 3 --group 2 --sft-steps 500
```

Each iteration: roll out → score → keep successes + oracle-relabel failures → re-SFT.
Prints the **per-iteration mean-reward curve** (the "RL is working" signal) and uploads the
final checkpoint to `<user>/pi05-mars-swarm-rl`.

### 5. Final eval

Same as step 3, pointing the server at the RL checkpoint:

```bash
modal run robot_env/serve/pi05_modal_mars.py --checkpoint <user>/pi05-mars-swarm-rl
uv run python robot_env/run_swarm_vla.py --remote HOST:PORT --group 4
```

Success rate should rise toward 100%, deterministically and repeatably (fixed seeds).

### 6. Watch the policy (visualize)

`run_swarm_vla.py` is headless (HUD runs the env in its own process). To *see* the policy,
`visualize_swarm.py` drives the bridge directly and renders — pulling actions from the same
served checkpoint:

```bash
# live MuJoCo window (macOS needs mjpython), policy on the remote server from step 5:
mjpython robot_env/visualize_swarm.py --remote HOST:PORT

# headless → overview MP4 (for the demo reel; needs: uv sync --extra viz):
uv run python robot_env/visualize_swarm.py --remote HOST:PORT --video build.mp4

# sanity-check the viz with no GPU/server — the scripted oracle:
mjpython robot_env/visualize_swarm.py --oracle
```

(`--checkpoint <repo>` runs a checkpoint locally instead of `--remote`; `--noop` holds still.)

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
- **lerobot is pinned to one commit everywhere** (`b8ad81b…`, lerobot 0.5.2) — local
  (`core/pyproject.toml`, the `record`/`vla` extras) and the Modal images — so the dataset
  is written and read by the same version. The training entrypoint is
  `python -m lerobot.scripts.lerobot_train` (0.5.x renamed it from `lerobot.scripts.train`),
  with the `[training,pi]` extras for the `datasets`/policy deps. Checkpoints upload via
  `huggingface_hub` (version-independent of lerobot's push flags). If you bump lerobot,
  re-check the entrypoint + flags and re-record the dataset.
- **numpy is forced to 2.x** (`[tool.uv] override-dependencies`): the pinned lerobot needs
  `numpy>=2.0`, hud's openpi-client caps `<2.0`, but that cap is conservative (verified: the
  no-op wire-check runs fine on numpy 2.2). This lets hud + lerobot share one local env.
- **Modal image deps are split to avoid that numpy conflict on pip** (which, unlike uv,
  has no override): `sft_modal` = lerobot only; `rl_loop` = lerobot + mujoco, **no hud**
  (it drives the bridge directly and runs `build_mars_infer`, which is hud-free); the
  serve image = lerobot + `openpi-client` installed `--no-deps` so its numpy<2 cap is
  skipped (the codec runs fine on numpy 2.x). `build_mars_infer` is plain lerobot
  (`postprocess(predict_action_chunk(preprocess(batch)))`), no hud.
- **SFT uses `--policy.type=pi05 --policy.pretrained_path=pi05_base`** (derive features
  from our 1-cam/9-state/5-action dataset; load base weights) run through a wrapper that
  forces **fresh processors** — pi05_base's saved processor uses a step name this lerobot
  renamed, so loading it would KeyError; we build fresh from the policy config instead.
- **Headless rendering on Modal**: `rl_loop` sets `MUJOCO_GL=osmesa` (+ `libosmesa6`) so the
  rover camera renders without a display.
- **Memory**: pi05 is 4B params, so a full finetune's optimizer states alone exceed a 40GB
  A100. Both training paths use `--policy.train_expert_only=true` (freeze the PaliGemma VLM,
  train only the ~300M action expert + projections — the right way to finetune pi05 on a new
  embodiment) + `--policy.gradient_checkpointing=true`, on an **A100-80GB**, batch size 4,
  with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
