"""Supervised-finetune pi0.5 on the Mars swarm dataset, on a Modal A100.

Stage 1 of the SFT→RL pipeline. Runs lerobot's trainer from `lerobot/pi05_base` on a
Path-B swarm dataset (one episode per rover, recorded by record_dataset.py --robot swarm),
then uploads the checkpoint to the HF Hub so the policy server (pi05_modal_mars.py) and
the RL loop (rl_loop.py) can pull it.

For the feasibility demo, run with a SMALL `--steps` so the policy lands clearly below the
oracle (initial eval reward < 1.0) — the RL loop then drives it up.

    modal run core/train/sft_modal.py \
        --dataset-repo <user>/mars-construction-swarm \
        --output-repo <user>/pi05-mars-swarm-sft \
        --steps 2000

Requires a Modal secret `huggingface-secret` carrying HF_TOKEN (write scope):
    modal secret create huggingface-secret HF_TOKEN=hf_...
"""
from __future__ import annotations

import modal

CACHE = "/cache"
_LEROBOT = "lerobot @ git+https://github.com/huggingface/lerobot.git@b8ad81bf397d59dda69ccfc7e74e847f0a9d4fbf"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg")
    .pip_install(
        _LEROBOT,
        "torch", "transformers", "accelerate", "safetensors", "huggingface_hub",
        "wandb", "pillow", "scipy", "einops",
    )
    .env({"HF_HOME": CACHE, "WANDB_MODE": "disabled"})
)

app = modal.App("mars-swarm-sft")
cache_vol = modal.Volume.from_name("mars-swarm-pi05-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")


@app.function(image=image, gpu="A100", timeout=6 * 3600,
              volumes={CACHE: cache_vol}, secrets=[hf_secret])
def train(dataset_repo: str, output_repo: str, steps: int = 2000,
          batch_size: int = 8, base: str = "lerobot/pi05_base") -> str:
    """Finetune pi0.5 → local dir → push to HF. Returns the output repo id."""
    import subprocess
    import sys

    out_dir = f"{CACHE}/ckpt/{output_repo.replace('/', '__')}"

    # lerobot's trainer (draccus config CLI). Single camera + 9-dim state + 5-dim action
    # come straight from the dataset's feature schema and normalization stats.
    cmd = [
        sys.executable, "-m", "lerobot.scripts.train",
        f"--dataset.repo_id={dataset_repo}",
        "--policy.type=pi05",
        f"--policy.pretrained_path={base}",
        f"--output_dir={out_dir}",
        f"--steps={steps}",
        f"--batch_size={batch_size}",
        f"--save_freq={steps}",
        "--save_checkpoint=true",
        "--policy.device=cuda",
        "--wandb.enable=false",
    ]
    print(f"[sft] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)

    # Upload the final checkpoint dir to HF (version-independent of lerobot's push flags).
    # lerobot writes checkpoints under <out_dir>/checkpoints/last/pretrained_model.
    from pathlib import Path

    from huggingface_hub import HfApi

    ckpt = Path(out_dir) / "checkpoints" / "last" / "pretrained_model"
    if not ckpt.exists():  # fall back to the newest pretrained_model under out_dir
        cands = sorted(Path(out_dir).rglob("pretrained_model"), key=lambda p: p.stat().st_mtime)
        if not cands:
            raise RuntimeError(f"no pretrained_model produced under {out_dir}")
        ckpt = cands[-1]

    api = HfApi()
    api.create_repo(output_repo, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=str(ckpt), repo_id=output_repo, repo_type="model")
    print(f"[sft] pushed → https://huggingface.co/{output_repo}", flush=True)
    return output_repo


@app.local_entrypoint()
def main(dataset_repo: str, output_repo: str, steps: int = 2000, batch_size: int = 8) -> None:
    repo = train.remote(dataset_repo, output_repo, steps=steps, batch_size=batch_size)
    print(f"SFT checkpoint: {repo}")
