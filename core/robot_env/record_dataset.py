"""Record a scripted oracle into a LeRobot v3 dataset (and push to HF).

Each robot ships a scripted expert; this driver replays it against the bridge,
captures `(image, proprioceptive state, action)` per tick, and writes a LeRobot v3
dataset that lerobot can fine-tune a VLA on (e.g. pi0.5).

Supported robots (`--robot`):
  - arm      MarsArmPickPlaceBridge — pick 20 cubes into a dome
  - printer  MarsPrinterBridge      — print a structure (--structure dome|wall|tower)

Run from `core/`:
    # validate the data pipeline with NO heavy deps (no lerobot/torch needed):
    uv run python robot_env/record_dataset.py --robot arm --dry-run
    uv run python robot_env/record_dataset.py --robot printer --structure tower --dry-run

    # write + push a real dataset (needs the `record` extra: uv sync --extra record,
    # and `huggingface-cli login` or HF_TOKEN):
    uv run python robot_env/record_dataset.py --robot printer --structure tower \
        --repo-id <hf-user>/mars-print-tower --push

Key design choice — DON'T leak the answer. Each bridge's state includes the target
coordinates; a policy trained on those ignores the camera. So the recorded
`observation.state` is PROPRIOCEPTION ONLY (joints, end-effector pose, tool state). The
privileged target coords ARE recorded, but in a separate `godmode` column (deliberately
NOT under the `observation.*` prefix, so no standard training config feeds it to the
policy). Use it for debugging, analysis, reward, or a privileged critic.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

IMAGE_SHAPE = (256, 256, 3)


@dataclass
class RecordSpec:
    """Everything the recorder needs to drive + slice one robot's scripted oracle."""
    name: str
    robot_type: str
    make_bridge: Callable[[bool], Any]      # render -> bridge instance
    reset_task_id: str                       # passed to bridge.reset(task_id=...)
    make_actions: Callable[[], list]         # the scripted expert's action sequence
    proprio_idx: list[int]                   # state slice the POLICY sees
    proprio_names: list[str]
    godmode_idx: list[int]                    # privileged target coords (side channel)
    godmode_names: list[str]
    action_names: list[str]


def _build_spec(robot: str, structure: str) -> RecordSpec:
    if robot == "arm":
        from hud_arm_bridge import MarsArmPickPlaceBridge, _scripted_actions
        # state[16]: joints(0-3), gripper(4), ee_xyz(5-7), cube_xyz(8-10),
        #            target_xyz(11-13), holding(14), placed(15)
        return RecordSpec(
            name="arm", robot_type="mars_pick_place_arm",
            make_bridge=lambda render: MarsArmPickPlaceBridge(render=render),
            reset_task_id="record",  # the arm ignores task_id
            make_actions=_scripted_actions,
            proprio_idx=[0, 1, 2, 3, 4, 5, 6, 7, 14],
            proprio_names=["arm_yaw", "arm_shoulder", "arm_elbow", "arm_wrist",
                           "gripper", "ee_x", "ee_y", "ee_z", "holding"],
            godmode_idx=[8, 9, 10, 11, 12, 13],
            godmode_names=["cube_x", "cube_y", "cube_z", "target_x", "target_y", "target_z"],
            action_names=["d_yaw", "d_shoulder", "d_elbow", "d_wrist", "gripper"],
        )
    if robot == "printer":
        from hud_printer_bridge import (
            PRINT_STRUCTURES,
            MarsPrinterBridge,
            _scripted_printer_actions,
        )
        if structure not in PRINT_STRUCTURES:
            raise SystemExit(f"unknown structure '{structure}'. choices: {list(PRINT_STRUCTURES)}")
        # state[16]: joints(0-3), extruder(4), ee_xyz(5-7), target_xyz(8-10),
        #            printed/total/pct/at_target/wp_idx(11-15)
        return RecordSpec(
            name=f"printer-{structure}", robot_type="mars_3d_printer_arm",
            make_bridge=lambda render: MarsPrinterBridge(render=render),
            reset_task_id=structure,  # the printer reads structure from task_id
            make_actions=lambda: _scripted_printer_actions(structure),
            proprio_idx=[0, 1, 2, 3, 4, 5, 6, 7],
            proprio_names=["arm_yaw", "shoulder", "elbow", "wrist",
                           "extruder", "ee_x", "ee_y", "ee_z"],
            godmode_idx=[8, 9, 10],
            godmode_names=["target_x", "target_y", "target_z"],
            action_names=["d_yaw", "d_shoulder", "d_elbow", "d_wrist", "extrude"],
        )
    raise SystemExit(f"unknown robot '{robot}'. choices: arm, printer")


def _features(spec: RecordSpec) -> dict:
    return {
        "observation.image": {"dtype": "image", "shape": IMAGE_SHAPE,
                              "names": ["height", "width", "channels"]},
        "observation.state": {"dtype": "float32", "shape": (len(spec.proprio_idx),),
                              "names": spec.proprio_names},
        "action": {"dtype": "float32", "shape": (len(spec.action_names),),
                   "names": spec.action_names},
        # privileged ground truth — intentionally NOT under the observation.* prefix
        "godmode": {"dtype": "float32", "shape": (len(spec.godmode_idx),),
                    "names": spec.godmode_names},
    }


def _count_str(res: dict) -> str:
    if "placed_count" in res:
        return f"placed={res['placed_count']}/20"
    if "printed_count" in res:
        return f"printed={res['printed_count']}/{res.get('total_waypoints', '?')}"
    return f"score={res.get('score')}"


async def _collect_episode(bridge: Any, spec: RecordSpec, seed: int, on_frame) -> dict:
    """Replay the oracle once; on_frame(image, state, godmode, action) per tick (obs before action)."""  # noqa: E501
    prompt = await bridge.reset(task_id=spec.reset_task_id, seed=seed)
    actions = spec.make_actions()
    for action in actions:
        obs, terminated = bridge.get_observation()
        full = obs["observation/state"]
        image = obs["observation/image"]
        state = full[spec.proprio_idx].astype(np.float32)     # what the policy sees
        godmode = full[spec.godmode_idx].astype(np.float32)   # privileged side channel
        on_frame(image, state, godmode, np.asarray(action, dtype=np.float32), prompt)
        bridge.step(action)                                   # then the expert acts
        if terminated:
            break
    return bridge.result()


def _dry_run(spec: RecordSpec, episodes: int, seed0: int) -> None:
    """Collect frames and validate shapes WITHOUT importing lerobot (no torch needed)."""
    bridge = spec.make_bridge(True)
    total = {"frames": 0}
    first: dict = {}

    def on_frame(image, state, godmode, action, prompt):
        total["frames"] += 1
        if not first:
            first.update(img=image.shape, img_dtype=str(image.dtype),
                         img_mean=round(float(image.mean()), 1),
                         state=state.shape, godmode=godmode.shape, action=action.shape, task=prompt)

    for ep in range(episodes):
        res = asyncio.run(_collect_episode(bridge, spec, seed0 + ep, on_frame))
        print(f"  episode {ep}: {_count_str(res)} success={res['success']}")
    bridge.close()
    print(f"\nfirst frame: {first}")
    print(f"total frames: {total['frames']}  (features: {list(_features(spec))})")
    print(f"dry-run OK [{spec.name}] — valid (image, proprio, godmode, action) frames.")


def _record(spec: RecordSpec, repo_id: str, episodes: int, seed0: int, fps: int,
            root: str | None, push: bool, overwrite: bool) -> None:
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
        repo_id=repo_id, fps=fps, features=_features(spec),
        robot_type=spec.robot_type, root=str(target), use_videos=True,
    )
    bridge = spec.make_bridge(True)
    for ep in range(episodes):
        def on_frame(image, state, godmode, action, prompt):
            # lerobot >=0.4: `task` is a reserved key inside the frame dict (no kwarg).
            ds.add_frame({"observation.image": image,
                          "observation.state": state,
                          "godmode": godmode,
                          "action": action,
                          "task": prompt})
        res = asyncio.run(_collect_episode(bridge, spec, seed0 + ep, on_frame))
        ds.save_episode()
        print(f"  episode {ep}: {_count_str(res)} saved")
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
    ap = argparse.ArgumentParser(description="Record a scripted oracle into a LeRobot v3 dataset.")
    ap.add_argument("--robot", choices=["arm", "printer"], default="arm")
    ap.add_argument("--structure", default="dome", help="printer only: dome|wall|tower")
    ap.add_argument("--repo-id", help="HF dataset id, e.g. your-user/mars-print-tower")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0, help="first episode seed")
    ap.add_argument("--fps", type=int, default=10, help="contract control_rate is 10 Hz")
    ap.add_argument("--root", default=None, help="local dataset dir (default: /tmp/lerobot/<repo>)")
    ap.add_argument("--push", action="store_true", help="push to the HF Hub after writing")
    ap.add_argument("--overwrite", action="store_true", help="delete existing local dataset dir")
    ap.add_argument("--dry-run", action="store_true", help="validate frames without lerobot")
    args = ap.parse_args()

    spec = _build_spec(args.robot, args.structure)

    if args.episodes > 1:
        print("NOTE: the scene is currently deterministic — every episode is identical. "
              "Add scene randomization (and a live-pose oracle) for real VLA diversity; "
              "recording multiple identical episodes adds no information.")

    if args.dry_run:
        _dry_run(spec, args.episodes, args.seed)
        return
    if not args.repo_id:
        ap.error("--repo-id is required unless --dry-run")
    _record(spec, args.repo_id, args.episodes, args.seed, args.fps,
            args.root, args.push, args.overwrite)


if __name__ == "__main__":
    main()
