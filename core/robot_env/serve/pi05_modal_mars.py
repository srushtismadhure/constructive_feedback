"""Serve the Mars-swarm pi0.5 policy on a Modal GPU. Prints ws://HOST:PORT.

The zero-infrastructure way to run a finetuned checkpoint on a remote GPU: no box to
rent or SSH into. Mirrors robot_training/serve/pi05_modal.py but serves OUR single-camera
server (policy_server_mars.py) and takes a `--checkpoint` so the RL loop can stand up each
iteration's checkpoint in turn.

    pip install modal && modal token new                       # one-time
    modal run robot_env/serve/pi05_modal_mars.py --checkpoint <user>/pi05-mars-swarm-sft
    # then on the (CPU-only) eval machine:
    uv run python robot_env/run_swarm_vla.py --remote HOST:PORT --group 4
"""
from __future__ import annotations

import sys
from pathlib import Path

import modal

PORT = 8000
CACHE = "/cache"  # HF cache (checkpoints + processors), Volume-backed so it persists

# lerobot pinned to a commit with pi05 (PyPI 0.5.1 lacks it), matching robot_training.
_LEROBOT = "lerobot[pi] @ git+https://github.com/huggingface/lerobot.git@b8ad81bf397d59dda69ccfc7e74e847f0a9d4fbf"
_SERVE_DIR = Path(__file__).resolve().parent

# No hud here: build_mars_infer is hud-free (lerobot only), and the openpi/0 wire needs
# just the standalone `openpi-client` codec. lerobot pins numpy>=2 while openpi-client
# caps numpy<2, so openpi-client is installed with --no-deps (its other deps are listed
# explicitly) and runs on numpy 2.x (verified) — avoiding an unsatisfiable pip resolve.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg")
    .pip_install(
        _LEROBOT,
        "torch", "transformers", "accelerate", "safetensors", "huggingface_hub",
        "numpy>=2.0,<2.3", "websockets>=11.0", "msgpack>=1.0.5", "msgpack-numpy",
        "dm-tree>=0.1.8", "pillow", "scipy", "einops",
    )
    .run_commands("python -m pip install --no-deps openpi-client")  # skip its numpy<2 cap
    .add_local_dir(str(_SERVE_DIR), "/root/serve", copy=True)
    .env({"HF_HOME": CACHE, "PYTHONPATH": "/root"})
)

app = modal.App("mars-swarm-pi05-serve")
cache_vol = modal.Volume.from_name("mars-swarm-pi05-cache", create_if_missing=True)


@app.function(image=image, gpu="A100", timeout=24 * 3600, volumes={CACHE: cache_vol})
def serve(checkpoint: str) -> None:
    import asyncio

    sys.path.insert(0, "/root/serve")
    from policy_server_mars import build_mars_infer, serve_openpi

    infer = build_mars_infer(checkpoint, device="cuda")
    with modal.forward(PORT, unencrypted=True) as tunnel:
        host, port = tunnel.tcp_socket
        print(f"[serve] pi0.5 ready — run_swarm_vla.py --remote {host}:{port}", flush=True)
        asyncio.run(serve_openpi("0.0.0.0", PORT, infer, metadata={"checkpoint": checkpoint}))


@app.local_entrypoint()
def main(checkpoint: str) -> None:
    serve.remote(checkpoint)
