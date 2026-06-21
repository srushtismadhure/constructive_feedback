"""Single-rover pick-and-place manipulation task for VLA eval / RL finetuning.

Wraps ``MarsSwarmBridge`` (one rover, one cube) as a HUD ``RobotBridge`` so a VLA
policy can drive the 5-DoF arm over the ``robot`` (openpi/0) capability and be graded
on placement. The split mirrors the swarm demo: **the policy controls only the arm;
navigation stays scripted inside the bridge.** The rover is driven (deterministically)
to pile staging before the episode hands control to the policy, and again to the dome
staging point between the pick and the place — so every agent step the policy sees is a
manipulation step (matching the Path-B dataset, which records manipulation frames only).

Grasp and release use the bridge's NATURAL detection (gripper closes within
``PICK_DISTANCE`` of the reserved cube → grasp; gripper opens while holding → release
snaps the cube onto its dome target), so success genuinely depends on the policy
positioning the arm and actuating the gripper — the signal RL improves.

Reward is shaped for partial credit so the demo's per-iteration curve is smooth:
approach progress < grasped < placed.
"""
from __future__ import annotations

import asyncio

import numpy as np

try:
    from hud.environment.robot import RobotBridge
except ImportError:  # allow importing without hud installed (e.g. shape checks)
    class RobotBridge:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            pass

from hud_arm_bridge import CAMERA_HEIGHT, CAMERA_WIDTH
from rover_nav import RoverDriveController
from run_swarm_demo import (
    STAGING_TOL,  # track the demo's nav tolerance (loosened + paired with stall recovery)
    STOW_CLOSED,
    STOW_OPEN,
    _dome_staging_for_target,
    _pile_staging_for,
)
from swarm_bridge import MarsSwarmBridge

# Proprioception the policy sees (must match record_dataset.py's swarm proprio_idx):
# joints(0-3), gripper(4), ee_xyz(5-7), holding(14). cube/target stay godmode-only.
PROPRIO_IDX = [0, 1, 2, 3, 4, 5, 6, 7, 14]
PROPRIO_NAMES = ["arm_yaw", "arm_shoulder", "arm_elbow", "arm_wrist",
                 "gripper", "ee_x", "ee_y", "ee_z", "holding"]
ACTION_NAMES = ["d_yaw", "d_shoulder", "d_elbow", "d_wrist", "gripper"]

# Rover 0 owns the south wedge; the prompt MUST match the dataset's rover-0 prompt
# (record_dataset.SWARM_SECTOR_NAMES[0]) so train/infer instructions are identical.
PROMPT = "Pick a crimson block from the pile and place it on the south wedge of the dome."

APPROACH_REF = 0.30         # m; EE-to-cube distance that maps to zero approach credit
PICK_CAP = 90               # max policy steps to achieve a grasp before the pick fails
PLACE_CAP = 90              # max policy steps to release after a grasp
DRIVE_CAP = 4000            # max physics ticks for one scripted nav leg
STALL_THRESHOLD = 800       # ticks before snapping the chassis to goal (matches RoverAgent)

# The HUD per-rover contract: 5-DoF arm action, ONE camera, 9-dim proprio state.
# No baked stats — the policy self-normalizes from its dataset (like hud_arm_bridge).
CONTRACT = {
    "robot_type": "mars_swarm_rover_arm",
    "control_rate": 10,
    "features": {
        "observation/image": {
            "role": "observation", "type": "rgb", "dtype": "uint8",
            "shape": [CAMERA_HEIGHT, CAMERA_WIDTH, 3],
            "names": ["height", "width", "channel"],
        },
        "observation/state": {
            "role": "observation", "dtype": "float32",
            "shape": [len(PROPRIO_NAMES)], "names": PROPRIO_NAMES,
        },
        "action": {
            "role": "action", "dtype": "float32",
            "shape": [len(ACTION_NAMES)], "names": ACTION_NAMES,
        },
    },
}


class SwarmManipBridge(RobotBridge):
    """One rover's pick→place as a gradeable HUD robot episode."""

    ROVER_IDX = 0  # south rover

    def __init__(self, render: bool = True):
        super().__init__()
        self.render = render
        self.bridge: MarsSwarmBridge | None = None
        self.nav: RoverDriveController | None = None
        self.phase = "pick"
        self.terminated = False
        self._reserved_cube: int | None = None
        self._target_idx: int | None = None
        self._pick_steps = 0
        self._place_steps = 0
        self._best_pick_dist = float("inf")
        self._score = 0.0
        self._success = False

    # ---- HUD RobotBridge interface ----
    async def reset(self, task_id: str = "south", seed: int = 0) -> str:
        del task_id
        bridge = MarsSwarmBridge(render=self.render)
        bridge.reset(seed=seed)
        # Isolate a single one-cube task on rover 0; idle the others.
        for other in (1, 2):
            bridge.rovers[other].dome_queue.clear()
        r0 = bridge.rovers[self.ROVER_IDX]
        while len(r0.dome_queue) > 1:
            r0.dome_queue.pop()
        # The dome slot is assigned at grasp time (claim_pile_cube pops dome_queue[0]),
        # so capture the upcoming slot now for grading.
        self._target_idx = r0.dome_queue[0]
        self._reserved_cube = bridge.claim_pile_cube_for(self.ROVER_IDX)
        if self._reserved_cube is None:
            raise RuntimeError("swarm reset produced no pickable cube for rover 0")

        self.bridge = bridge
        self.nav = RoverDriveController(bridge.model, bridge.data, prefix="r0_")
        # Scripted leg 1: drive (gripper open) from spawn to pile staging.
        self._drive_to(_pile_staging_for(self.ROVER_IDX), STOW_OPEN)

        self.phase = "pick"
        self.terminated = False
        self._pick_steps = self._place_steps = 0
        self._best_pick_dist = float("inf")
        self._score, self._success = 0.0, False
        return PROMPT

    def get_observation(self):
        obs, _ = self.bridge.get_observation(self.ROVER_IDX)
        full = obs["observation/state"]
        return {
            "observation/image": obs["observation/image"],
            "observation/state": full[PROPRIO_IDX].astype(np.float32),
        }, self.terminated

    def step(self, action) -> None:
        if self.terminated:
            return
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        rover = self.bridge.rovers[self.ROVER_IDX]

        if self.phase == "pick":
            self._arm_step(action)
            self._pick_steps += 1
            full = self.bridge._rover_state(rover)
            self._best_pick_dist = min(self._best_pick_dist,
                                       float(np.linalg.norm(full[5:8] - full[8:11])))
            if rover.holding:
                # Grasp achieved → scripted leg 2: carry the cube to dome staging.
                tgt = np.asarray(self.bridge.dome_targets[self._target_idx])
                self._drive_to(_dome_staging_for_target(tgt), STOW_CLOSED)
                self.phase = "place"
            elif self._pick_steps >= PICK_CAP:
                self.terminated = True  # never grasped → pick failed
        elif self.phase == "place":
            self._arm_step(action)
            self._place_steps += 1
            if self.bridge.cube_placed[self._target_idx]:
                self.terminated = True  # placed
            elif self._place_steps >= PLACE_CAP:
                self.terminated = True

        self._grade()

    def result(self) -> dict:
        return {"score": float(self._score), "success": bool(self._success)}

    def close(self) -> None:
        if self.bridge is not None:
            self.bridge.close()
        self.bridge = None
        self.nav = None

    # ---- internals ----
    def _arm_step(self, arm_action: np.ndarray) -> None:
        """One physics tick: policy drives rover 0's arm, the other rovers idle."""
        actions = [STOW_OPEN, STOW_OPEN, STOW_OPEN]
        actions[self.ROVER_IDX] = arm_action
        self.bridge.step(actions)

    def _drive_to(self, goal_xy, arm_action: np.ndarray) -> bool:
        """Scripted nav leg: drive rover 0 to goal_xy holding ``arm_action`` on the
        arm (open before grasp, closed while carrying). Physics advances via the
        bridge so grasp/carry bookkeeping stays consistent. Mirrors RoverAgent's
        stall recovery: if the rover can't reach the goal (wheel/contact edge case),
        snap its chassis there so the arm always starts from a stable, in-reach pose."""
        for tick in range(DRIVE_CAP):
            if self.nav.at_goal(goal_xy, tol=STAGING_TOL):
                self.nav.stop()
                self._arm_step(arm_action)  # one settle tick
                return True
            if tick and tick % STALL_THRESHOLD == 0:
                qa, qv = self.nav.qposadr, self.nav.qveladr
                self.bridge.data.qpos[qa] = goal_xy[0]
                self.bridge.data.qpos[qa + 1] = goal_xy[1]
                self.bridge.data.qvel[qv:qv + 6] = 0.0
            self.nav.step_drive(goal_xy)
            self._arm_step(arm_action)
        self.nav.stop()
        return self.nav.at_goal(goal_xy, tol=STAGING_TOL)

    def _grade(self) -> None:
        """Shaped partial credit: approach (<0.5) < grasped (0.5) < placed (1.0)."""
        rover = self.bridge.rovers[self.ROVER_IDX]
        if self.bridge.cube_placed[self._target_idx]:
            self._score, self._success = 1.0, True
        elif rover.holding or self.phase == "place":
            self._score, self._success = 0.5, False
        else:
            approach = max(0.0, 1.0 - self._best_pick_dist / APPROACH_REF)
            self._score, self._success = 0.5 * approach, False


def _smoke() -> None:
    """Scripted-policy sanity check: replay rover 0's oracle arm actions through this
    bridge and confirm it reaches placement (score 1.0). Run from core/:
        uv run python robot_env/swarm_manip_bridge.py
    """
    from run_swarm_demo import RoverAgent

    b = SwarmManipBridge(render=False)
    asyncio.run(b.reset(seed=0))
    agent = RoverAgent(SwarmManipBridge.ROVER_IDX, b.bridge)  # reuse its IK action builders
    agent.current_pile_cube_idx = b._reserved_cube              # what tick() normally sets

    for a in agent._build_pick_actions():      # close on the cube → natural grasp
        b.step(a)
        if b.phase == "place" or b.terminated:
            break
    if b.phase == "place":                      # bridge auto-drove to dome staging
        for a in agent._build_place_actions():  # open over the target → release
            b.step(a)
            if b.terminated:
                break

    print(f"scripted smoke: {b.result()} phase={b.phase} terminated={b.terminated} "
          f"pick_steps={b._pick_steps} best_dist={b._best_pick_dist:.3f}")
    b.close()


if __name__ == "__main__":
    _smoke()
