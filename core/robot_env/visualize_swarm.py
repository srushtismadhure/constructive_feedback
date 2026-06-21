"""Watch the swarm pick-and-place policy run — live MuJoCo window or an overview MP4.

`run_swarm_vla.py` is headless (HUD runs the env in its own process), so this script owns
the loop instead: it drives `SwarmManipBridge` directly and renders, while pulling actions
from the same policy you'd evaluate. The sim runs locally, so the policy can live on a
remote GPU server (serve/pi05_modal_mars.py) — only obs→action crosses the network.

    # live window (macOS needs mjpython), policy on a remote GPU server:
    mjpython robot_env/visualize_swarm.py --remote HOST:PORT

    # the scripted oracle — no GPU/server, just to sanity-check the visualization:
    mjpython robot_env/visualize_swarm.py --oracle

    # a local checkpoint (needs the vla extra: uv sync --extra vla):
    mjpython robot_env/visualize_swarm.py --checkpoint <user>/pi05-mars-swarm-rl

    # headless → write an overview MP4 instead of opening a window (needs imageio[ffmpeg]):
    uv run python robot_env/visualize_swarm.py --remote HOST:PORT --video out.mp4
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from collections import deque
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from swarm_manip_bridge import PROMPT, SwarmManipBridge  # noqa: E402

HORIZON = 10
MAX_STEPS = 200


# ── action sources: obs dict -> [T, 5] arm-action chunk ───────────────────────
def _remote_chunk_fn(host: str, port: str | int):
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    client = WebsocketClientPolicy(host=host, port=int(port))
    print(f"[viz] connected to policy server ws://{host}:{port}", flush=True)

    def fn(obs: dict) -> np.ndarray:
        result = client.infer({
            "observation/image": obs["observation/image"],
            "observation/state": obs["observation/state"],
            "prompt": PROMPT,
        })
        return np.asarray(result["actions"], dtype=np.float32)
    return fn


def _checkpoint_chunk_fn(checkpoint: str):
    sys.path.insert(0, str(Path(__file__).resolve().parent / "serve"))
    from policy_server_mars import build_mars_infer
    infer = build_mars_infer(checkpoint, horizon=HORIZON)

    def fn(obs: dict) -> np.ndarray:
        return np.asarray(infer({**obs, "prompt": PROMPT})["actions"], dtype=np.float32)
    return fn


def _noop_chunk_fn(_obs: dict) -> np.ndarray:
    return np.tile([0.0, 0.0, 0.0, 0.0, 1.0], (HORIZON, 1)).astype(np.float32)


def _overview_camera() -> mujoco.MjvCamera:
    """A fixed third-person camera framing the whole dome build (matches the swarm demo)."""
    cam = mujoco.MjvCamera()
    cam.distance, cam.elevation, cam.azimuth = 12.0, -55.0, 90.0
    cam.lookat[:] = [0.0, 0.0, 3.5]
    return cam


def run(args: argparse.Namespace) -> None:
    bridge = SwarmManipBridge(render=not args.oracle)  # policy modes need the obs camera
    asyncio.run(bridge.reset(seed=args.seed))
    model, data = bridge.bridge.model, bridge.bridge.data
    cam = _overview_camera()

    # Sinks: a live passive viewer and/or an offscreen overview renderer for video.
    viewer = None if args.video else mujoco.viewer.launch_passive(model, data)
    if viewer is not None:
        viewer.cam.distance, viewer.cam.elevation = cam.distance, cam.elevation
        viewer.cam.azimuth, viewer.cam.lookat[:] = cam.azimuth, cam.lookat
    renderer = mujoco.Renderer(model, height=480, width=640) if args.video else None
    frames: list[np.ndarray] = []

    def show() -> None:
        if viewer is not None:
            viewer.sync()
            time.sleep(0.04)
        if renderer is not None:
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render().copy())

    if args.oracle:
        _drive_oracle(bridge, show)
    else:
        chunk_fn = (_remote_chunk_fn(*args.remote.split(":")) if args.remote
                    else _checkpoint_chunk_fn(args.checkpoint) if args.checkpoint
                    else _noop_chunk_fn)
        _drive_policy(bridge, chunk_fn, show)

    print(f"[viz] result: {bridge.result()}", flush=True)
    if viewer is not None:
        viewer.close()
    if renderer is not None:
        renderer.close()
        _write_video(frames, args.video, args.fps)
    bridge.close()


def _drive_policy(bridge: SwarmManipBridge, chunk_fn, show) -> None:
    chunk: deque = deque()
    steps = 0
    while not bridge.terminated and steps < MAX_STEPS:
        obs, _ = bridge.get_observation()
        if not chunk:
            chunk = deque(np.atleast_2d(chunk_fn(obs)))
        action = np.asarray(chunk.popleft(), dtype=np.float32).reshape(-1)[:5]
        bridge.step(action)
        show()
        steps += 1


def _drive_oracle(bridge: SwarmManipBridge, show) -> None:
    from run_swarm_demo import RoverAgent
    agent = RoverAgent(bridge.ROVER_IDX, bridge.bridge)
    agent.current_pile_cube_idx = bridge._reserved_cube

    def drive(actions) -> None:
        for a in actions:
            if bridge.terminated:
                return
            bridge.step(a)
            show()

    drive(agent._build_pick_actions())
    if bridge.phase == "place":
        drive(agent._build_place_actions())


def _write_video(frames: list[np.ndarray], path: str, fps: int) -> None:
    if not frames:
        print("[viz] no frames captured; nothing to write", flush=True)
        return
    try:
        import imageio.v2 as imageio
    except ImportError:
        raise SystemExit("video needs imageio: `uv pip install 'imageio[ffmpeg]'`") from None
    imageio.mimsave(path, frames, fps=fps)
    print(f"[viz] wrote {len(frames)} frames → {path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize the swarm policy (live viewer or MP4).")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--remote", metavar="HOST:PORT", help="openpi/0 policy server (serve/)")
    src.add_argument("--checkpoint", metavar="REPO", help="local pi0.5 checkpoint (needs torch)")
    src.add_argument("--noop", action="store_true", help="hold still (wire/viewer check)")
    src.add_argument("--oracle", action="store_true", help="scripted expert (no GPU needed)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--video", metavar="PATH", help="write an overview MP4 instead of a window")
    ap.add_argument("--fps", type=int, default=20)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
