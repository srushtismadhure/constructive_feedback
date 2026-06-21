"""Mars 3-D Printer Bridge — same arm as hud_arm_bridge, different task.

The arm EE traces a printing path. When the 'extruder' action is engaged
near a waypoint the position is marked printed. No cubes, no gripper.

Action: [d_yaw, d_shoulder, d_elbow, d_wrist, extrude]
  extrude ∈ [-1, 1] — integrates into self.extruder [0, 1]; > 0.5 = printing

Supported structures (pass structure_id to reset()):
  "dome"  — 20-position concentric-circle dome (default)
  Add new entries to PRINT_STRUCTURES to extend.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import os
import sys
from dataclasses import dataclass
from typing import Callable

import mujoco
import numpy as np

try:
    from hud.environment.robot import RobotBridge
except ImportError:
    class RobotBridge:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            pass

# Ensure robot_env/ is on the path so hud_arm_bridge resolves regardless of
# the caller's working directory (e.g. when run from core/ via uv).
_ROBOT_ENV_DIR = os.path.dirname(os.path.abspath(__file__))
if _ROBOT_ENV_DIR not in sys.path:
    sys.path.insert(0, _ROBOT_ENV_DIR)

# ---------------------------------------------------------------------------
# Re-use constants and helpers from the arm bridge (same hardware).
# ---------------------------------------------------------------------------
from hud_arm_bridge import (
    ARM_ACTUATORS,
    ARM_JOINTS,
    ARM_YAW_BODY_WORLD,
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    DELTA_SCALE,
    GRIPPER_ACTUATORS,
    GRIPPER_CLOSED_LEFT,
    GRIPPER_CLOSED_RIGHT,
    GRIPPER_JOINTS,
    L1,
    L2,
    L3,
    PHYSICS_STEPS_PER_ACTION,
    SCENE_DIR,
    _dome_positions,
    _ik_top_down,
    _move_to,
    _repeat,
)

# ---------------------------------------------------------------------------
# Printer-specific constants
# ---------------------------------------------------------------------------
# Tracked-printer scene (same arm kinematics as mars_scene.xml; restyled base
# with tank treads + nozzle extruder). Falls back to mars_scene.xml if absent.
SCENE_FILE = "printer_scene.xml"
PRINT_RADIUS = 0.10          # EE must be within this distance to mark printed
EXTRUDE_THRESHOLD = 0.5      # self.extruder above this → printing active
EXTRUDE_RATE = 0.2           # change per step per unit action (matches gripper rate)

CONTRACT = {
    "robot_type": "mars_3d_printer_arm_mujoco",
    "control_rate": 10,
    "features": {
        "observation/image": {
            "role": "observation",
            "type": "rgb",
            "dtype": "uint8",
            "shape": [CAMERA_HEIGHT, CAMERA_WIDTH, 3],
            "names": ["height", "width", "channel"],
        },
        "observation/state": {
            "role": "observation",
            "dtype": "float32",
            "shape": [16],
            "names": [
                "arm_yaw",
                "shoulder",
                "elbow",
                "wrist",
                "extruder",
                "ee_x",
                "ee_y",
                "ee_z",
                "target_x",
                "target_y",
                "target_z",
                "printed_count",
                "total_waypoints",
                "completion_pct",
                "at_target",
                "waypoint_idx",
            ],
        },
        "action": {
            "role": "action",
            "dtype": "float32",
            "shape": [5],
            "names": ["d_yaw", "d_shoulder", "d_elbow", "d_wrist", "extrude"],
        },
    },
}

def _wall_positions() -> list[np.ndarray]:
    """4-row × 5-column vertical wall (20 waypoints), printed row-by-row from base.

    Layout: fixed x=1.64, y spans ±0.10 around centre (−0.11), z rises from
    3.39 to 3.54 in 0.05 m steps.  All points verified reachable by _ik_top_down.
    """
    cx, cy = 1.64, -0.11
    z_levels = [3.39, 3.44, 3.49, 3.54]
    y_offsets = [-0.10, -0.05, 0.00, 0.05, 0.10]
    pts: list[np.ndarray] = []
    for z in z_levels:
        for dy in y_offsets:
            pts.append(np.array([cx, cy + dy, z], dtype=np.float64))
    return pts


def _tower_positions() -> list[np.ndarray]:
    """4 concentric rings stacked in z (20 waypoints), tapering inward like a tower.

    Ring radii shrink from 0.14 → 0.05 m as z rises from 3.39 → 3.54.  Five
    equally-spaced waypoints per ring, printed in angle order.
    """
    cx, cy = 1.64, -0.11
    z_levels = [3.39, 3.44, 3.49, 3.54]
    radii    = [0.14, 0.11, 0.08, 0.05]
    n_per_ring = 5
    pts: list[np.ndarray] = []
    for z, r in zip(z_levels, radii):
        for k in range(n_per_ring):
            angle = 2 * np.pi * k / n_per_ring
            pts.append(np.array([cx + r * np.cos(angle), cy + r * np.sin(angle), z], dtype=np.float64))
    return pts


# ---------------------------------------------------------------------------
# Structure registry — add new structures here.
# ---------------------------------------------------------------------------
PRINT_STRUCTURES: dict[str, Callable[[], list[np.ndarray]]] = {
    "dome":  _dome_positions,
    "wall":  _wall_positions,
    "tower": _tower_positions,
}


@dataclass
class EpisodeResult:
    score: float = 0.0
    success: bool = False
    total_reward: float = 0.0
    printed_count: int = 0
    total_waypoints: int = 0


class MarsPrinterBridge(RobotBridge):
    """HUD RobotBridge for the Mars 3-D printing arm.

    Uses the same arm hardware as MarsArmPickPlaceBridge but replaces the
    pick-and-place task with a print-path-following task.  The fifth action
    dimension controls the extruder instead of the gripper.
    """

    def __init__(self, render: bool = True):
        super().__init__()
        self.render = render
        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None
        self.renderer: mujoco.Renderer | None = None

        self.arm_qposadrs: list[int] = []
        self.arm_dofadrs: list[int] = []
        self.arm_actuator_ids: list[int] = []
        self.gripper_actuator_ids: list[int] = []
        self.gripper_qposadrs: list[int] = []
        self.gripper_dofadrs: list[int] = []
        self.ee_site_id = 0

        self.waypoints: list[np.ndarray] = []
        self.waypoint_printed: list[bool] = []
        self.current_waypoint_idx = 0

        self.targets = np.zeros(4, dtype=np.float32)
        self.extruder = 0.0       # [0, 1]; > EXTRUDE_THRESHOLD = printing
        self.success = False
        self.terminated = False
        self.total_reward = 0.0
        self.steps = 0
        self.max_steps = 3000

    async def reset(self, task_id: str = "dome", seed: int = 0, **kwargs) -> str:
        structure_id = task_id if task_id in PRINT_STRUCTURES else "dome"
        np.random.seed(seed)

        cwd = os.getcwd()
        try:
            os.chdir(SCENE_DIR)
            scene = SCENE_FILE if os.path.exists(SCENE_FILE) else "mars_scene.xml"
            self.model = mujoco.MjModel.from_xml_path(scene)
        finally:
            os.chdir(cwd)
        self.data = mujoco.MjData(self.model)

        if self.render:
            try:
                self.renderer = mujoco.Renderer(
                    self.model, height=CAMERA_HEIGHT, width=CAMERA_WIDTH
                )
            except Exception as exc:
                self.renderer = None
                print(f"[warn] renderer unavailable; returning black frames: {exc}")

        self.arm_qposadrs = [self.model.jnt_qposadr[self.model.joint(j).id] for j in ARM_JOINTS]
        self.arm_dofadrs  = [self.model.jnt_dofadr[self.model.joint(j).id]  for j in ARM_JOINTS]
        self.arm_actuator_ids = [self.model.actuator(a).id for a in ARM_ACTUATORS]
        self.gripper_actuator_ids = [self.model.actuator(a).id for a in GRIPPER_ACTUATORS]
        self.gripper_qposadrs = [self.model.jnt_qposadr[self.model.joint(j).id] for j in GRIPPER_JOINTS]
        self.gripper_dofadrs  = [self.model.jnt_dofadr[self.model.joint(j).id]  for j in GRIPPER_JOINTS]
        self.ee_site_id = self.model.site("ee_site").id

        self.waypoints = PRINT_STRUCTURES[structure_id]()
        self.waypoint_printed = [False] * len(self.waypoints)
        self.current_waypoint_idx = 0

        # Home pose (matches arm bridge home).
        self.targets[:] = np.array([0.0, -0.35, 0.65, -0.25], dtype=np.float32)
        self.extruder = 0.0
        self.success = False
        self.terminated = False
        self.total_reward = 0.0
        self.steps = 0

        self._apply_targets()
        mujoco.mj_forward(self.model, self.data)
        return (
            f"Print a {structure_id} structure using {len(self.waypoints)} waypoints. "
            f"Move the arm EE to each waypoint and activate the extruder to deposit material."
        )

    def step(self, action) -> None:
        self._require_ready()
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size < 5:
            raise ValueError(
                f"Expected 5-D action [d_yaw,d_shoulder,d_elbow,d_wrist,extrude], got {action!r}"
            )

        deltas = np.clip(action[:4], -1.0, 1.0) * DELTA_SCALE
        self.targets = np.clip(self.targets + deltas, self._arm_low(), self._arm_high())
        self.extruder = float(np.clip(self.extruder + EXTRUDE_RATE * float(action[4]), 0.0, 1.0))

        self._apply_targets()
        self._update_print()
        for _ in range(PHYSICS_STEPS_PER_ACTION):
            mujoco.mj_step(self.model, self.data)
            self._update_print()

        self.steps += 1
        reward = self._compute_reward()
        self.total_reward += reward
        self.success = self._is_success()
        self.terminated = self.success or self.steps >= self.max_steps

    def get_observation(self):
        self._require_ready()
        assert self.data is not None
        if self.renderer is None:
            rgb = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
        else:
            self.renderer.update_scene(self.data)
            rgb = self.renderer.render().astype(np.uint8)
        return {"observation/image": rgb, "observation/state": self._state()}, self.terminated

    def result(self) -> dict:
        n = len(self.waypoints)
        return EpisodeResult(
            score=sum(self.waypoint_printed) / max(1, n),
            success=self.success,
            total_reward=float(self.total_reward),
            printed_count=int(sum(self.waypoint_printed)),
            total_waypoints=n,
        ).__dict__

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
        self.renderer = None
        self.data = None
        self.model = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _require_ready(self) -> None:
        if self.model is None or self.data is None:
            raise RuntimeError("Bridge not reset. Call reset() first.")

    def _arm_low(self) -> np.ndarray:
        assert self.model is not None
        return np.array([self.model.jnt_range[self.model.joint(j).id, 0] for j in ARM_JOINTS], dtype=np.float32)

    def _arm_high(self) -> np.ndarray:
        assert self.model is not None
        return np.array([self.model.jnt_range[self.model.joint(j).id, 1] for j in ARM_JOINTS], dtype=np.float32)

    def _apply_targets(self) -> None:
        assert self.data is not None
        for act_id, target in zip(self.arm_actuator_ids, self.targets):
            self.data.ctrl[act_id] = float(target)
        # Hold arm qpos directly (same as arm bridge — no fighting actuator dynamics).
        for qposadr, dofadr, target in zip(self.arm_qposadrs, self.arm_dofadrs, self.targets):
            self.data.qpos[qposadr] = float(target)
            self.data.qvel[dofadr] = 0.0
        # Keep gripper fully open; it's unused in the printing task.
        left, right = self.gripper_actuator_ids
        self.data.ctrl[left] = GRIPPER_CLOSED_LEFT * 0.0   # i.e. 0 → open
        self.data.ctrl[right] = GRIPPER_CLOSED_RIGHT * 0.0
        for qposadr, dofadr in zip(self.gripper_qposadrs, self.gripper_dofadrs):
            self.data.qpos[qposadr] = 0.0
            self.data.qvel[dofadr] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _update_print(self) -> None:
        """Mark current waypoint printed when extruder is active and EE is close."""
        if self.current_waypoint_idx >= len(self.waypoints):
            return
        assert self.data is not None
        mujoco.mj_forward(self.model, self.data)
        ee = self.data.site_xpos[self.ee_site_id]
        target = self.waypoints[self.current_waypoint_idx]
        if self.extruder > EXTRUDE_THRESHOLD and np.linalg.norm(ee - target) <= PRINT_RADIUS:
            self.waypoint_printed[self.current_waypoint_idx] = True
            self.current_waypoint_idx += 1

    def _current_target(self) -> np.ndarray:
        idx = min(self.current_waypoint_idx, len(self.waypoints) - 1)
        return self.waypoints[idx]

    def _state(self) -> np.ndarray:
        assert self.data is not None
        joints = np.array([self.data.qpos[q] for q in self.arm_qposadrs], dtype=np.float32)
        ee = self.data.site_xpos[self.ee_site_id].astype(np.float32)
        target = self._current_target().astype(np.float32)
        n = len(self.waypoints)
        printed = int(sum(self.waypoint_printed))
        at_target = float(np.linalg.norm(ee - target) <= PRINT_RADIUS)
        return np.concatenate([
            joints,                                              # 4
            np.array([self.extruder], dtype=np.float32),        # 1
            ee,                                                  # 3
            target,                                              # 3
            np.array([                                           # 5
                float(printed),
                float(n),
                float(printed) / max(1, n),
                at_target,
                float(self.current_waypoint_idx),
            ], dtype=np.float32),
        ])

    def _compute_reward(self) -> float:
        printed = sum(self.waypoint_printed)
        bonus = 5.0 if self._is_success() else 0.0
        # Dense: reward each new printed waypoint; small living penalty.
        return float(printed - 0.01 + bonus)

    def _is_success(self) -> bool:
        return all(self.waypoint_printed)


# ---------------------------------------------------------------------------
# Scripted policy
# ---------------------------------------------------------------------------

def _scripted_printer_actions(structure_id: str = "dome") -> list[np.ndarray]:
    """Generate the full print sequence for the given structure.

    Per waypoint: approach → descend → extrude → stop → lift.
    Each move is sized so DELTA_SCALE-clipped per-step deltas can actually reach the
    next IK waypoint.
    """
    if structure_id not in PRINT_STRUCTURES:
        raise ValueError(f"Unknown structure '{structure_id}'. Add it to PRINT_STRUCTURES.")
    waypoints = PRINT_STRUCTURES[structure_id]()

    current = np.array([0.0, -0.35, 0.65, -0.25], dtype=np.float32)
    actions: list[np.ndarray] = []

    approach_dz = 0.20  # EE this far above waypoint for transit
    print_dz = 0.05     # EE this far above waypoint to deposit material

    for i, waypoint in enumerate(waypoints):
        waypoint_xyz = waypoint

        waypoint_high = _ik_top_down(waypoint_xyz + np.array([0.0, 0.0, approach_dz]))
        waypoint_low = _ik_top_down(waypoint_xyz + np.array([0.0, 0.0, print_dz]))

        if any(p is None for p in (waypoint_high, waypoint_low)):
            raise RuntimeError(f"Unreachable waypoint {i}: {waypoint_xyz}")

        move, current = _move_to(current, waypoint_high, 0.0)
        actions += move
        move, current = _move_to(current, waypoint_low, 0.0)
        actions += move
        # 5 extrude ticks: extruder 0.0 → 1.0, crossing the 0.5 print threshold by tick 3.
        actions += _repeat([0.0, 0.0, 0.0, 0.0, 1.0], 5)
        # 3 stop ticks: extruder 1.0 → 0.4, back below threshold before transit.
        actions += _repeat([0.0, 0.0, 0.0, 0.0, -1.0], 3)
        move, current = _move_to(current, waypoint_high, 0.0)
        actions += move

    return actions


# ---------------------------------------------------------------------------
# Smoke test / demo runner
# ---------------------------------------------------------------------------

async def _smoke_test(steps: int, scripted: bool, render: bool, structure_id: str) -> None:
    bridge = MarsPrinterBridge(render=render)
    prompt = await bridge.reset(task_id=structure_id, seed=0)
    print(f"prompt: {prompt}")

    actions = _scripted_printer_actions(structure_id) if scripted else [np.zeros(5, dtype=np.float32)] * steps
    print(f"[scripted actions] {len(actions)} total" if scripted else f"[zero actions] {steps} steps")

    for i, action in enumerate(actions[:steps]):
        bridge.step(action)
        obs, terminated = bridge.get_observation()
        state = obs["observation/state"]
        if (i + 1) % 25 == 0 or terminated:
            print(
                f"step={i + 1:4d}  printed={int(state[11])}/{int(state[12])}  "
                f"extruder={state[4]:.2f}  at_target={state[14]:.0f}  "
                f"terminated={terminated}"
            )
        if terminated:
            break

    print(f"result: {bridge.result()}")
    bridge.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the Mars 3-D printer bridge.")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--scripted", action="store_true", help="Run full scripted print sequence.")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--structure", default="dome", choices=list(PRINT_STRUCTURES))
    args = parser.parse_args()
    asyncio.run(_smoke_test(args.steps, args.scripted, args.render, args.structure))


if __name__ == "__main__":
    main()
