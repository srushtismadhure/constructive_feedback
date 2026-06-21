"""Record the scripted pick-and-place oracle into a LeRobot v3 dataset (and push to HF).

The scripted policy (`_scripted_actions`) is the expert; this driver replays it
against `MarsArmPickPlaceBridge`, captures `(image, proprioceptive state, action)`
at every tick, and writes a LeRobot v3 dataset that lerobot can fine-tune a VLA on
(e.g. pi0.5).

Run from `core/`:
    # validate the data pipeline with NO heavy deps (no lerobot/torch needed):
    uv run python robot_env/record_dataset.py --dry-run

    # write + push a real dataset (needs the `record` extra: uv sync --extra record,
    # and `huggingface-cli login` or HF_TOKEN):
    uv run python robot_env/record_dataset.py --repo-id <hf-user>/mars-dome-pickplace --push

Key design choice — DON'T leak the answer. The bridge's 16-dim state includes the
target cube/goal coordinates; a policy trained on those ignores the camera. So the
recorded `observation.state` is PROPRIOCEPTION ONLY (arm joints, gripper, end-effector
pose, holding flag). The oracle still *uses* the privileged coords to act; the student
just never sees them. Widen `PROPRIO_IDX` if you want more in the observation.

The privileged cube/target coordinates ARE recorded, but in a separate `godmode` column
(deliberately NOT under the `observation.*` prefix, so no standard training config feeds
it to the policy). Use it for debugging, analysis, reward, or a privileged critic.
"""

from __future__ import annotations

import argparse
import asyncio

import numpy as np

from hud_arm_bridge import MarsArmPickPlaceBridge, _scripted_actions

# Slice of the bridge's 16-dim state that the policy is allowed to see.
# layout: [yaw, shoulder, elbow, wrist, gripper, ee_x, ee_y, ee_z,
#          cube_x, cube_y, cube_z, target_x, target_y, target_z, holding, placed_count]
PROPRIO_IDX = [0, 1, 2, 3, 4, 5, 6, 7, 14]  # joints(4) + gripper + ee_xyz + holding
PROPRIO_NAMES = ["arm_yaw", "arm_shoulder", "arm_elbow", "arm_wrist",
                 "gripper", "ee_x", "ee_y", "ee_z", "holding"]
ACTION_NAMES = ["d_yaw", "d_shoulder", "d_elbow", "d_wrist", "gripper"]
IMAGE_SHAPE = (256, 256, 3)

# Privileged ground truth — the cube + target coordinates. Stored in a SEPARATE
# `godmode` column (not `observation.*`), so it never feeds the policy unless you
# explicitly map it. Useful for debugging, analysis, reward, or a privileged critic.
GODMODE_IDX = [8, 9, 10, 11, 12, 13]
GODMODE_NAMES = ["cube_x", "cube_y", "cube_z", "target_x", "target_y", "target_z"]


def _features() -> dict:
    return {
        "observation.image": {"dtype": "image", "shape": IMAGE_SHAPE,
                              "names": ["height", "width", "channels"]},
        "observation.state": {"dtype": "float32", "shape": (len(PROPRIO_IDX),),
                              "names": PROPRIO_NAMES},
        "action": {"dtype": "float32", "shape": (len(ACTION_NAMES),), "names": ACTION_NAMES},
        # privileged ground truth — intentionally NOT under the observation.* prefix
        "godmode": {"dtype": "float32", "shape": (len(GODMODE_IDX),), "names": GODMODE_NAMES},
    }


async def _collect_episode(bridge: MarsArmPickPlaceBridge, seed: int, on_frame) -> dict:
    """Replay the oracle once; call on_frame(image, state, godmode, action) per tick (obs BEFORE action)."""
    prompt = await bridge.reset(task_id="record", seed=seed)
    actions = _scripted_actions()
    for action in actions:
        obs, terminated = bridge.get_observation()
        full = obs["observation/state"]
        image = obs["observation/image"]
        state = full[PROPRIO_IDX].astype(np.float32)          # what the policy sees
        godmode = full[GODMODE_IDX].astype(np.float32)        # privileged side channel
        on_frame(image, state, godmode, np.asarray(action, dtype=np.float32), prompt)
        bridge.step(action)                                   # then the expert acts
        if terminated:
            break
    return bridge.result()


def _dry_run(episodes: int, seed0: int) -> None:
    """Collect frames and validate shapes WITHOUT importing lerobot (no torch needed)."""
    bridge = MarsArmPickPlaceBridge(render=True)
    total = {"frames": 0}
    first = {}

    def on_frame(image, state, godmode, action, prompt):
        total["frames"] += 1
        if not first:
            first.update(img=image.shape, img_dtype=str(image.dtype),
                         img_mean=round(float(image.mean()), 1),
                         state=state.shape, godmode=godmode.shape, action=action.shape, task=prompt)

    for ep in range(episodes):
        res = asyncio.run(_collect_episode(bridge, seed0 + ep, on_frame))
        print(f"  episode {ep}: placed={res['placed_count']}/20 success={res['success']}")
    bridge.close()
    print(f"\nfirst frame: {first}")
    print(f"total frames: {total['frames']}  (features: {list(_features())})")
    print("dry-run OK — data pipeline produces valid (image, proprio, action) frames.")


def _record(repo_id: str, episodes: int, seed0: int, fps: int, root: str | None,
            push: bool, overwrite: bool) -> None:
    import shutil
    from pathlib import Path
    try:  # import path moved across lerobot versions
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        try:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:
            raise SystemExit(
                "lerobot not installed. From core/: `uv sync --extra record` "
                "(pulls lerobot + torch), then re-run."
            ) from exc

    # Local copy defaults to /tmp (ephemeral staging) instead of lerobot's ~/.cache —
    # the durable copy is the HF Hub after --push. Pass --root to choose another path.
    target = Path(root) if root else Path("/tmp/lerobot") / repo_id

    # `create` refuses to write into an existing dir; a prior (even failed) run leaves one.
    if target.exists():
        if overwrite:
            shutil.rmtree(target)
        else:
            raise SystemExit(
                f"{target} already exists (a prior run left it). Re-run with --overwrite, "
                f"or delete it manually."
            )

    ds = LeRobotDataset.create(
        repo_id=repo_id, fps=fps, features=_features(),
        robot_type="mars_pick_place_arm", root=str(target), use_videos=True,
    )
    bridge = MarsArmPickPlaceBridge(render=True)
    for ep in range(episodes):
        def on_frame(image, state, godmode, action, prompt):
            # lerobot >=0.4: `task` is a reserved key inside the frame dict (no kwarg).
            ds.add_frame({"observation.image": image,
                          "observation.state": state,
                          "godmode": godmode,
                          "action": action,
                          "task": prompt})
        res = asyncio.run(_collect_episode(bridge, seed0 + ep, on_frame))
        ds.save_episode()
        print(f"  episode {ep}: placed={res['placed_count']}/20 saved")
    bridge.close()

    # REQUIRED before push: flushes the buffered episode metadata (meta/episodes/*.parquet)
    # and consolidates. Skipping it uploads an incomplete dataset — HF's viewer then fails
    # with "Parquet magic bytes not found" — and leaves the flush to a failing __del__.
    ds.finalize()

    print(f"\nLeRobot v3 dataset written: {ds.root}")
    if push:
        print(f"pushing to HF Hub: {repo_id} …")
        ds.push_to_hub()
        print(f"done → https://huggingface.co/datasets/{repo_id}")
    else:
        print(f"to publish: huggingface-cli login  &&  re-run with --push "
              f"(or `huggingface-cli upload {repo_id} {ds.root} --repo-type dataset`)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Record the scripted oracle into a LeRobot v3 dataset.")
    ap.add_argument("--repo-id", help="HF dataset id, e.g. your-user/mars-dome-pickplace")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0, help="first episode seed")
    ap.add_argument("--fps", type=int, default=10, help="contract control_rate is 10 Hz")
    ap.add_argument("--root", default=None, help="local dataset dir (default: /tmp/lerobot/<repo-id>)")
    ap.add_argument("--push", action="store_true", help="push to the HF Hub after writing")
    ap.add_argument("--overwrite", action="store_true", help="delete an existing local dataset dir first")
    ap.add_argument("--dry-run", action="store_true", help="validate frames without lerobot")
    args = ap.parse_args()

    if args.episodes > 1:
        print("NOTE: the scene is currently deterministic — every episode is identical. "
              "Add scene randomization (and make the oracle read live cube poses) for real "
              "VLA diversity. Recording multiple identical episodes adds no information.")

    if args.dry_run:
        _dry_run(args.episodes, args.seed)
        return
    if not args.repo_id:
        ap.error("--repo-id is required unless --dry-run")
    _record(args.repo_id, args.episodes, args.seed, args.fps, args.root, args.push, args.overwrite)


if __name__ == "__main__":
    main()
