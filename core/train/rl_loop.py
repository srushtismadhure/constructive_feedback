"""Filtered-BC / DAgger reinforcement-finetuning loop for the Mars swarm VLA.

Stage 2 of the SFT→RL pipeline, and the demo's centrepiece: it shows a *closed* RL loop
— rollout → reward → data selection → weight update → improved checkpoint — running end
to end. Because the env is deterministic (by design), grouped rollouts are identical, so
the loop can't exceed the scripted oracle; its real job is the honest one: **fix the SFT
policy's compounding-error / distribution shift** by retraining on its own successful
on-policy rollouts plus expert-relabeled corrections for the states where it failed.

Each iteration:
  1. Roll out the current policy on the deterministic manip task(s), in-process (drive
     `SwarmManipBridge` directly — same bridge + reward as the HUD eval path).
  2. Score each rollout (the bridge's shaped placement reward).
  3. Keep frames from rollouts with reward >= threshold; for the rest, substitute the
     scripted ORACLE's trajectory for that seed (DAgger-lite expert relabeling).
  4. Write the kept+relabeled frames as a LeRobot dataset and re-run SFT -> new checkpoint.
  5. Log mean reward (the rising curve).

Run the full loop on a Modal A100 (policy inference + lerobot training in one container):
    modal run core/train/rl_loop.py --base-checkpoint <user>/pi05-mars-swarm-sft --iters 3

Validate the loop MECHANICS locally with no GPU/torch (oracle + noop stand-in policies):
    uv run python core/train/rl_loop.py --dry-run
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from collections import deque
from pathlib import Path
from typing import Callable

import numpy as np

ROBOT_ENV = Path(__file__).resolve().parents[1] / "robot_env"
sys.path.insert(0, str(ROBOT_ENV))

from swarm_manip_bridge import PROMPT, SwarmManipBridge  # noqa: E402

GODMODE_IDX = [8, 9, 10, 11, 12, 13]  # cube_xyz, target_xyz (privileged side channel)
HORIZON = 10
ROLLOUT_MAX_STEPS = 200

# A policy is a chunk function: observation dict -> [T, 5] arm-action chunk.
ChunkFn = Callable[[dict], np.ndarray]


# ── rollout collection (in-process; bypasses the openpi wire) ─────────────────
def collect_rollout(chunk_fn: ChunkFn, seed: int) -> tuple[list[dict], dict]:
    """Drive one episode of the policy through SwarmManipBridge. Returns the per-step
    frames (obs before action) and the bridge's result dict."""
    b = SwarmManipBridge(render=True)
    asyncio.run(b.reset(seed=seed))
    rover = b.bridge.rovers[b.ROVER_IDX]
    frames: list[dict] = []
    chunk: deque = deque()
    steps = 0
    while not b.terminated and steps < ROLLOUT_MAX_STEPS:
        obs, _ = b.get_observation()
        full = b.bridge._rover_state(rover)
        if not chunk:
            chunk = deque(np.atleast_2d(np.asarray(chunk_fn(obs), dtype=np.float32)))
        action = np.asarray(chunk.popleft(), dtype=np.float32).reshape(-1)[:5]
        frames.append({
            "observation.image": obs["observation/image"],
            "observation.state": obs["observation/state"].astype(np.float32),
            "godmode": full[GODMODE_IDX].astype(np.float32),
            "action": action,
            "task": PROMPT,
        })
        b.step(action)
        steps += 1
    res = b.result()
    b.close()
    return frames, res


def oracle_rollout(seed: int) -> tuple[list[dict], dict]:
    """The scripted expert's trajectory for a seed, captured as frames (the relabel
    source and the dry-run 'good policy'). Reuses RoverAgent's IK action builders.
    Returns (frames, result)."""
    from run_swarm_demo import RoverAgent

    b = SwarmManipBridge(render=True)
    asyncio.run(b.reset(seed=seed))
    rover = b.bridge.rovers[b.ROVER_IDX]
    agent = RoverAgent(b.ROVER_IDX, b.bridge)
    agent.current_pile_cube_idx = b._reserved_cube
    frames: list[dict] = []

    def _drive(actions: list[np.ndarray]) -> None:
        for a in actions:
            if b.terminated:
                return
            obs, _ = b.get_observation()
            full = b.bridge._rover_state(rover)
            frames.append({
                "observation.image": obs["observation/image"],
                "observation.state": obs["observation/state"].astype(np.float32),
                "godmode": full[GODMODE_IDX].astype(np.float32),
                "action": np.asarray(a, dtype=np.float32).reshape(-1)[:5],
                "task": PROMPT,
            })
            b.step(a)

    _drive(agent._build_pick_actions())
    if b.phase == "place":
        _drive(agent._build_place_actions())
    res = b.result()
    b.close()
    return frames, res


def noop_chunk_fn(_obs: dict) -> np.ndarray:
    """Stand-in 'bad policy' for the dry run: hold arm, gripper open → never grasps."""
    return np.tile([0.0, 0.0, 0.0, 0.0, 1.0], (HORIZON, 1)).astype(np.float32)


# ── filter + expert relabel (the RL data-selection step) ──────────────────────
def filter_and_relabel(chunk_fn: ChunkFn, seeds: list[int],
                       threshold: float) -> tuple[list[dict], list[float]]:
    """Keep frames from successful on-policy rollouts; for failures, substitute the
    oracle's trajectory for that seed. Returns (training_frames, per-seed rewards)."""
    frames: list[dict] = []
    rewards: list[float] = []
    for seed in seeds:
        roll, res = collect_rollout(chunk_fn, seed)
        rewards.append(float(res["score"]))
        if res["score"] >= threshold:
            frames += roll                      # on-policy success → self-imitation
        else:
            frames += oracle_rollout(seed)[0]   # failure → expert (DAgger-lite) relabel
    return frames, rewards


# ── LeRobot dataset write (reuses the recorder's feature schema) ──────────────
def write_dataset(frames: list[dict], repo_id: str, root: str, fps: int = 10):
    import shutil

    from record_dataset import _features, _swarm_spec
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    spec = _swarm_spec()
    target = Path(root)
    if target.exists():
        shutil.rmtree(target)
    ds = LeRobotDataset.create(repo_id=repo_id, fps=fps, features=_features(spec),
                               robot_type=spec.robot_type, root=str(target), use_videos=True)
    for frame in frames:
        ds.add_frame(frame)
    ds.save_episode()
    ds.finalize()
    return str(ds.root)


# ── the loop ──────────────────────────────────────────────────────────────────
def load_policy_chunk_fn(checkpoint: str, device: str = "cuda") -> ChunkFn:
    """In-process pi0.5 chunk fn (GPU). Mirrors serve/build_mars_infer but returns the
    [HORIZON, 5] chunk directly (no websocket)."""
    sys.path.insert(0, str(ROBOT_ENV / "serve"))
    from policy_server_mars import build_mars_infer

    infer = build_mars_infer(checkpoint, device=device, horizon=HORIZON)

    def chunk_fn(obs: dict) -> np.ndarray:
        obs = {**obs, "prompt": obs.get("prompt", PROMPT)}
        return np.asarray(infer(obs)["actions"], dtype=np.float32)

    return chunk_fn


def run_loop(base_checkpoint: str, dataset_repo: str, iters: int, group: int,
             threshold: float, sft_steps: int, sft_fn) -> list[float]:
    """sft_fn(prev_ckpt, dataset_repo, steps) -> new_ckpt. Returns the per-iter mean reward."""
    seeds = list(range(group))
    ckpt = base_checkpoint
    curve: list[float] = []
    for it in range(iters):
        chunk_fn = load_policy_chunk_fn(ckpt)
        frames, rewards = filter_and_relabel(chunk_fn, seeds, threshold)
        mean_r = float(np.mean(rewards))
        curve.append(mean_r)
        print(f"[rl] iter {it}: mean reward={mean_r:.4f}  rewards={[round(r,3) for r in rewards]}  "
              f"train frames={len(frames)}", flush=True)
        root = f"/tmp/lerobot/{dataset_repo.replace('/', '__')}-it{it}"
        write_dataset(frames, dataset_repo, root)
        ckpt = sft_fn(ckpt, dataset_repo, root, sft_steps)
    print(f"[rl] reward curve: {[round(c, 4) for c in curve]}", flush=True)
    return curve


def _dry_run(group: int, threshold: float) -> None:
    """Validate rollout + filter + relabel mechanics with NO torch: a noop policy
    (fails → relabeled) and the oracle (succeeds → kept)."""
    seeds = list(range(group))

    print("== noop policy (expect failures → oracle relabel) ==")
    frames_noop, r_noop = filter_and_relabel(noop_chunk_fn, seeds, threshold)
    print(f"  rewards={[round(r,3) for r in r_noop]}  frames={len(frames_noop)} (all relabeled)")

    print("== oracle policy (expect successes → kept on-policy) ==")
    frames_oracle: list[dict] = []
    r_oracle: list[float] = []
    for seed in seeds:
        roll, res = oracle_rollout(seed)
        r_oracle.append(float(res["score"]))
        frames_oracle += roll
    print(f"  rewards={[round(r,3) for r in r_oracle]}  training frames={len(frames_oracle)}")

    f0 = frames_oracle[0]
    print(f"\nframe shapes: image={f0['observation.image'].shape} "
          f"state={f0['observation.state'].shape} "
          f"godmode={f0['godmode'].shape} action={f0['action'].shape} task={f0['task']!r}")
    print("dry-run OK — rollout, reward, filter, and expert-relabel mechanics validated.")


# ── Modal: run the whole loop on one A100 (inference + training in-container) ──
# Defined at module level (modal is a base dep); only exercised by `modal run`.
import modal  # noqa: E402

_LEROBOT = "lerobot @ git+https://github.com/huggingface/lerobot.git@b8ad81bf397d59dda69ccfc7e74e847f0a9d4fbf"
_CORE = Path(__file__).resolve().parents[1]

_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install(
        _LEROBOT, "hud-python[robot]",
        "torch", "transformers", "accelerate", "safetensors", "huggingface_hub",
        "mujoco>=3.2.0", "numpy", "pillow", "scipy", "trimesh", "websockets", "einops",
    )
    .add_local_dir(str(_CORE / "robot_env"), "/root/robot_env", copy=True)
    .add_local_dir(str(_CORE / "train"), "/root/train", copy=True)
    .env({"HF_HOME": "/cache", "MUJOCO_GL": "egl", "PYTHONPATH": "/root/robot_env:/root/train"})
)
_app = modal.App("mars-swarm-rl")
_cache = modal.Volume.from_name("mars-swarm-pi05-cache", create_if_missing=True)
_hf = modal.Secret.from_name("huggingface-secret")


@_app.function(image=_image, gpu="A100", timeout=12 * 3600,
               volumes={"/cache": _cache}, secrets=[_hf])
def rl_train(base_checkpoint: str, dataset_repo: str, output_repo: str, iters: int = 3,
             group: int = 2, threshold: float = 0.999, sft_steps: int = 500) -> list[float]:
    import subprocess
    import sys as _sys

    def sft_fn(prev_ckpt: str, ds_repo: str, ds_root: str, steps: int) -> str:
        out = f"/cache/rl_ckpt/{output_repo.replace('/', '__')}"
        subprocess.run([
            _sys.executable, "-m", "lerobot.scripts.train",
            f"--dataset.repo_id={ds_repo}", f"--dataset.root={ds_root}",
            "--policy.type=pi05", f"--policy.pretrained_path={prev_ckpt}",
            f"--output_dir={out}", f"--steps={steps}", "--batch_size=8",
            f"--save_freq={steps}", "--policy.device=cuda", "--wandb.enable=false",
        ], check=True)
        cands = sorted(Path(out).rglob("pretrained_model"), key=lambda p: p.stat().st_mtime)
        if not cands:
            raise RuntimeError(f"no checkpoint produced under {out}")
        return str(cands[-1])

    curve = run_loop(base_checkpoint, dataset_repo, iters, group, threshold, sft_steps, sft_fn)

    # Publish the final checkpoint (the last sft_fn output).
    from huggingface_hub import HfApi
    out_root = Path(f"/cache/rl_ckpt/{output_repo.replace('/', '__')}")
    last = sorted(out_root.rglob("pretrained_model"), key=lambda p: p.stat().st_mtime)[-1]
    api = HfApi()
    api.create_repo(output_repo, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=str(last), repo_id=output_repo, repo_type="model")
    print(f"[rl] final → https://huggingface.co/{output_repo}  curve={curve}", flush=True)
    return curve


@_app.local_entrypoint()
def modal_main(base_checkpoint: str, dataset_repo: str, output_repo: str,
               iters: int = 3, group: int = 2, sft_steps: int = 500) -> None:
    curve = rl_train.remote(base_checkpoint, dataset_repo, output_repo,
                            iters=iters, group=group, sft_steps=sft_steps)
    print(f"RL reward curve: {curve}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Filtered-BC / DAgger RL loop for the swarm VLA.")
    ap.add_argument("--dry-run", action="store_true", help="validate loop mechanics, no torch/GPU")
    ap.add_argument("--group", type=int, default=2, help="rollouts per iteration (seeds 0..N-1)")
    ap.add_argument("--threshold", type=float, default=0.999, help="reward >= R keeps on-policy")
    args = ap.parse_args()
    if args.dry_run:
        _dry_run(args.group, args.threshold)
        return
    raise SystemExit("Non-dry-run RL training runs on Modal: `modal run core/train/rl_loop.py ...` "
                     "(see module docstring). Local entry is --dry-run only.")


if __name__ == "__main__":
    main()
