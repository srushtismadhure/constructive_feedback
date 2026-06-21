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
_LEROBOT = "lerobot[training,pi] @ git+https://github.com/huggingface/lerobot.git@b8ad81bf397d59dda69ccfc7e74e847f0a9d4fbf"

# Run lerobot_train but force FRESH processors. pi05_base's saved processor config uses a
# step name (`relative_actions_processor`) that this pinned lerobot renamed
# (`delta_actions_processor`), so loading it KeyErrors. We still want pi05_base's WEIGHTS
# (loaded separately by make_policy), just not its processor — so we patch
# make_pre_post_processors to ignore the pretrained processor path and build fresh from
# the policy config + our dataset stats. The relative-action step is inert at
# use_relative_actions=false, so our delta actions pass through unchanged.
_TRAIN_WRAPPER = '''\
import lerobot.policies.factory as F
import lerobot.scripts.lerobot_train as T
_orig = F.make_pre_post_processors
def _fresh(policy_cfg, pretrained_path=None, **kw):
    return _orig(policy_cfg, pretrained_path=None, **kw)
F.make_pre_post_processors = _fresh
T.make_pre_post_processors = _fresh
T.main()
'''

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg")
    .pip_install(
        _LEROBOT,
        "torch", "transformers", "accelerate", "safetensors", "huggingface_hub",
        "wandb", "pillow", "scipy", "einops",
    )
    .env({"HF_HOME": CACHE, "WANDB_MODE": "disabled",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
)

app = modal.App("mars-swarm-sft")
cache_vol = modal.Volume.from_name("mars-swarm-pi05-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")


@app.function(image=image, gpu="A100-80GB", timeout=6 * 3600,
              volumes={CACHE: cache_vol}, secrets=[hf_secret])
def train(dataset_repo: str, output_repo: str, steps: int = 2000,
          batch_size: int = 4, base: str = "lerobot/pi05_base") -> str:
    """Finetune pi0.5 → local dir → push to HF. Returns the output repo id."""
    import shutil
    import subprocess
    import sys

    out_dir = f"{CACHE}/ckpt/{output_repo.replace('/', '__')}"
    # lerobot refuses to train into an existing output_dir (unless --resume). The Modal
    # cache Volume persists it across runs, so clear it first — the durable copy is the
    # HF upload below.
    shutil.rmtree(out_dir, ignore_errors=True)

    # lerobot's trainer (draccus config CLI), via the fresh-processor wrapper.
    wrapper = f"{CACHE}/lerobot_train_fresh.py"
    with open(wrapper, "w") as f:
        f.write(_TRAIN_WRAPPER)
    cmd = [
        sys.executable, wrapper,
        f"--dataset.repo_id={dataset_repo}",
        # New embodiment: derive features (1 cam, 9-dim state, 5-dim action) from OUR
        # dataset via --policy.type, and load pi05_base weights via --pretrained_path.
        # (--policy.path would import pi05_base's rigid 3-camera LIBERO feature config
        # and fail validate_visual_features_consistency.)
        "--policy.type=pi05",
        f"--policy.pretrained_path={base}",
        f"--output_dir={out_dir}",
        f"--steps={steps}",
        f"--batch_size={batch_size}",
        f"--save_freq={steps}",
        "--save_checkpoint=true",
        "--policy.device=cuda",
        # pi05 is 4B params — a full finetune's optimizer states alone exceed 40GB.
        # Freeze the PaliGemma VLM and train only the action expert + projections (the
        # right way to finetune pi05 on a new embodiment), with gradient checkpointing.
        "--policy.train_expert_only=true",
        "--policy.gradient_checkpointing=true",
        "--policy.push_to_hub=false",  # we upload the checkpoint ourselves (below)
        "--wandb.enable=false",
    ]
    print(f"[sft] {' '.join(cmd)}", flush=True)
    # Stream the child's output live AND keep the tail, so on failure the real lerobot
    # traceback is re-raised at the top of the Modal error (not buried/lost).
    import collections
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    tail: collections.deque[str] = collections.deque(maxlen=80)
    for line in proc.stdout:
        print(line, end="", flush=True)
        tail.append(line)
    if proc.wait() != 0:
        raise RuntimeError("lerobot train failed. Last 80 lines:\n" + "".join(tail))

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
def main(dataset_repo: str, output_repo: str, steps: int = 2000, batch_size: int = 4) -> None:
    repo = train.remote(dataset_repo, output_repo, steps=steps, batch_size=batch_size)
    print(f"SFT checkpoint: {repo}")
