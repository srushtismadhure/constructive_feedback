"""openpi/0 websocket policy server for the Mars swarm rover arm — run on the GPU box.

The Mars analogue of robot_training/serve/policy_server.py. It holds a finetuned pi0.5
checkpoint and answers stateless observation→action-chunk requests over the openpi/0
(msgpack-numpy) websocket. The eval machine runs the sim + loop and connects with
`run_swarm_vla.py --remote HOST:PORT` (a `RemoteAgent`), so only this box needs a GPU.

Difference from the franka server: the swarm rover has ONE camera (`observation/image`)
and a 9-dim proprio state, and the action is the 5-DoF arm delta. The checkpoint is any
HF repo (e.g. the SFT or RL-iter checkpoints this pipeline produces) — pass `--checkpoint`.

    # real policy (GPU box, needs torch + lerobot):
    python serve/policy_server_mars.py --checkpoint <user>/pi05-mars-swarm-sft --port 8000

    # weightless wire check (no GPU / no model) — pairs with run_swarm_vla.py --remote:
    python serve/policy_server_mars.py --noop --port 8000
"""
from __future__ import annotations

import argparse
import asyncio
from typing import Any, Callable

import numpy as np
import websockets.asyncio.server as wss
import websockets.exceptions
from openpi_client import msgpack_numpy

# The swarm rover's single camera, mapped onto the policy's first image slot.
ENV_IMAGE_KEYS = ["observation/image"]
ARM_ACTION_DIM = 5

InferFn = Callable[[dict[str, Any]], dict[str, Any]]


async def serve_openpi(host: str, port: int, infer: InferFn, *,
                       metadata: dict | None = None) -> None:
    """Serve `infer` over the openpi/0 websocket forever (one inference per request)."""
    packer = msgpack_numpy.Packer()

    async def handler(ws: Any) -> None:
        await ws.send(packer.pack(metadata or {}))  # openpi handshake: metadata first
        try:
            while True:
                obs = msgpack_numpy.unpackb(await ws.recv())
                result = await asyncio.to_thread(infer, obs)
                await ws.send(packer.pack(result))
        except websockets.exceptions.ConnectionClosed:
            pass

    async with wss.serve(handler, host, port, compression=None, max_size=None) as server:
        print(f"[serve] openpi/0 mars policy server: ws://{host}:{port}", flush=True)
        await server.serve_forever()


def build_mars_infer(checkpoint: str, device: str | None = None, horizon: int = 10) -> InferFn:
    """Load a finetuned pi0.5 checkpoint and return its openpi `infer` (needs GPU + lerobot).

    Mirrors robot_training/serve/policy_server.build_pi05_infer, but maps the swarm's
    single `observation/image` onto the model's first image slot (any remaining slots
    auto zero-pad), passes the 9-dim `observation/state`, and truncates the returned
    chunk to the first 5 action dims (the arm delta) and to `horizon` steps so the client
    replans every `horizon` ticks.
    """
    import torch
    from hud.agents.robot.model import LeRobotModel
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[serve] loading policy: {checkpoint} (device={device})", flush=True)
    policy = PI05Policy.from_pretrained(checkpoint).to(device).eval()
    preprocess, postprocess = make_pre_post_processors(
        policy.config, checkpoint,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    model = LeRobotModel(policy, preprocess, postprocess)
    image_keys = list(policy.config.image_features)  # model slots in contract order

    def infer(obs: dict[str, Any]) -> dict[str, Any]:
        state = np.asarray(obs["observation/state"], dtype=np.float32)
        batch: dict[str, Any] = {
            "observation.state": torch.from_numpy(state),
            "task": obs.get("prompt", ""),
        }
        for model_key, env_key in zip(image_keys, ENV_IMAGE_KEYS, strict=False):
            img = torch.from_numpy(np.asarray(obs[env_key])).permute(2, 0, 1).float() / 255.0
            batch[model_key] = img
        chunk = model.infer(batch)[0, :horizon, :ARM_ACTION_DIM]  # [N,T,A] -> [horizon, 5]
        return {"actions": np.asarray(chunk, dtype=np.float32)}

    return infer


def build_noop_infer(horizon: int = 10) -> InferFn:
    """Weightless infer: hold the arm, gripper open (5-DoF). Remote analogue of NoopAgent."""
    chunk = np.tile([0.0, 0.0, 0.0, 0.0, 1.0], (horizon, 1)).astype(np.float32)

    def infer(obs: dict[str, Any]) -> dict[str, Any]:
        return {"actions": chunk}

    return infer


def main() -> None:
    ap = argparse.ArgumentParser(description="Serve a Mars-swarm VLA over the openpi/0 websocket.")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--noop", action="store_true", help="weightless no-op policy (no GPU/model)")
    ap.add_argument("--checkpoint", default=None, help="HF checkpoint for the finetuned pi0.5")
    ap.add_argument("--horizon", type=int, default=10, help="actions per chunk (replan period)")
    args = ap.parse_args()

    if args.noop:
        infer = build_noop_infer(args.horizon)
    elif args.checkpoint:
        infer = build_mars_infer(args.checkpoint, horizon=args.horizon)
    else:
        ap.error("pass --noop or --checkpoint REPO")
    meta = {"checkpoint": args.checkpoint or "noop"}
    asyncio.run(serve_openpi(args.host, args.port, infer, metadata=meta))


if __name__ == "__main__":
    main()
