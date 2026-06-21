import argparse
import asyncio
import math
import os
from dataclasses import dataclass

import mujoco
import numpy as np

try:
    from hud.environment.robot import RobotBridge
except ImportError:
    class RobotBridge:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            pass


SCENE_DIR = os.path.dirname(os.path.abspath(__file__))
CAMERA_HEIGHT = 256
CAMERA_WIDTH = 256
PHYSICS_STEPS_PER_ACTION = 20
DELTA_SCALE = np.array([0.07, 0.07, 0.07, 0.08], dtype=np.float32)
GRIPPER_CLOSED_LEFT = -0.020
GRIPPER_CLOSED_RIGHT = 0.020
PICK_DISTANCE = 0.16
TARGET_RADIUS = 0.16

NUM_CUBES = 36

# Pile: 3 cols x 3 rows x 4 layers, stacked at (1.68, -0.92). Order is top-down so the
# arm always picks an unsupported cube first.
PILE_CENTER_XY = (1.68, -0.92)
PILE_GRID = (3, 3, 4)  # cols, rows, layers
PILE_XY_SPACING = 0.12
PILE_Z_SPACING = 0.115
PILE_BASE_Z = 3.39

# Dome: 3 tiers of decreasing radius at increasing z. 16+12+8 = 36.
DOME_CENTER = (1.671, -0.146, 3.39)
DOME_TIERS = [
    # (count, radius, z, angle_offset_rad)
    (16, 0.50, 3.39, 0.0),
    (12, 0.36, 3.50, math.pi / 12),  # half-step offset so the ring lands between tier-0 cubes
    (8, 0.20, 3.61, 0.0),
]

# The arm's yaw_body sits at this world position (rover at 1.25,-0.5,3.55, +0.09 chassis-top,
# +0.09 pedestal, +0.09 yaw_body offset).
ARM_YAW_BODY_WORLD = np.array([1.25, -0.5, 3.73], dtype=np.float64)

# Arm link lengths used for analytic IK.
L1 = 0.42  # upper arm
L2 = 0.34  # forearm
L3 = 0.12  # wrist body origin to EE site

# Held-cube offset below the EE site (world frame).
HELD_OFFSET = np.array([0.0, 0.0, -0.075], dtype=np.float64)

ARM_JOINTS = ["arm_yaw", "arm_shoulder", "arm_elbow", "arm_wrist"]
ARM_ACTUATORS = ["arm_yaw_pos", "arm_shoulder_pos", "arm_elbow_pos", "arm_wrist_pos"]
GRIPPER_ACTUATORS = ["left_gripper_pos", "right_gripper_pos"]
GRIPPER_JOINTS = ["left_gripper_slide", "right_gripper_slide"]

CONTRACT = {
    "robot_type": "mars_pick_place_arm_mujoco",
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
                "gripper",
                "ee_x",
                "ee_y",
                "ee_z",
                "cube_x",
                "cube_y",
                "cube_z",
                "target_x",
                "target_y",
                "target_z",
                "holding",
                "placed_count",
            ],
        },
        "action": {
            "role": "action",
            "dtype": "float32",
            "shape": [5],
            "names": ["d_yaw", "d_shoulder", "d_elbow", "d_wrist", "gripper"],
        },
    },
}


@dataclass
class EpisodeResult:
    score: float = 0.0
    success: bool = False
    total_reward: float = 0.0
    placed_count: int = 0


def _pile_positions(pile_center: tuple[float, float] = PILE_CENTER_XY) -> list[np.ndarray]:
    """36 cube positions in a 3x3x4 stacked pile centered at ``pile_center``.
    Top layer first so picking always starts from an unsupported cube.
    """
    positions = []
    cols, rows, layers = PILE_GRID
    cx, cy = pile_center
    for i in range(NUM_CUBES):
        # i=0..8 → top layer; i=9..17 → second; etc.
        layer = (layers - 1) - (i // (cols * rows))
        in_layer = i % (cols * rows)
        col = in_layer % cols
        row = in_layer // cols
        x = cx + (col - (cols - 1) / 2.0) * PILE_XY_SPACING
        y = cy + (row - (rows - 1) / 2.0) * PILE_XY_SPACING
        z = PILE_BASE_Z + layer * PILE_Z_SPACING
        positions.append(np.array([x, y, z], dtype=np.float64))
    return positions


def _dome_positions(dome_center: tuple[float, float, float] = DOME_CENTER) -> list[np.ndarray]:
    """36 dome target positions forming a 3-tier dome (16+12+8) with decreasing
    radius at increasing height, centered at ``dome_center`` (xy used; z from tiers).
    """
    cx, cy = dome_center[0], dome_center[1]
    positions: list[np.ndarray] = []
    for count, radius, z, offset in DOME_TIERS:
        for i in range(count):
            angle = offset + 2 * math.pi * i / count
            positions.append(np.array([cx + radius * math.cos(angle),
                                       cy + radius * math.sin(angle),
                                       z], dtype=np.float64))
    return positions


def _ik_top_down(ee_world: np.ndarray,
                 arm_yaw_body_world: np.ndarray = ARM_YAW_BODY_WORLD,
                 rover_yaw: float = 0.0) -> np.ndarray | None:
    """Analytic IK: place the EE at ee_world with the gripper pointing straight down
    (shoulder+elbow+wrist = 0). Returns [yaw, shoulder, elbow, wrist] or None if unreachable.

    ``arm_yaw_body_world`` is the world position of the arm's yaw joint.
    ``rover_yaw`` is the rover's heading in world frame; the target is rotated into
    the arm's local frame before solving. The returned ``yaw`` is the joint angle in
    rover-local frame (which is what the joint actuator consumes).
    """
    rel_world = ee_world - arm_yaw_body_world
    # Rotate world-relative target into rover-local frame.
    cos_t = math.cos(-rover_yaw)
    sin_t = math.sin(-rover_yaw)
    x_rel = cos_t * float(rel_world[0]) - sin_t * float(rel_world[1])
    y_rel = sin_t * float(rel_world[0]) + cos_t * float(rel_world[1])
    z_rel = float(rel_world[2])
    yaw = math.atan2(y_rel, x_rel)
    r_xy = math.hypot(x_rel, y_rel)
    # With sum=0, the wrist body sits L3 behind the EE in arm-plane +x; same z as EE.
    r_w = r_xy - L3
    z_w = z_rel
    d_sq = r_w * r_w + z_w * z_w
    cos_e = (d_sq - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    if cos_e > 1.0 or cos_e < -1.0:
        return None
    elbow_geom = math.acos(cos_e)
    # Elbow-up branch: positive theta_e (forearm bends "down" relative to upper arm).
    theta_e = elbow_geom
    theta_s = (-math.atan2(z_w, r_w)
               + math.atan2(-L2 * math.sin(theta_e), L1 + L2 * math.cos(theta_e)))
    theta_w = -theta_s - theta_e
    return np.array([yaw, theta_s, theta_e, theta_w], dtype=np.float32)


class MarsArmPickPlaceBridge(RobotBridge):
    """HUD RobotBridge for a 20-cube pick-and-place dome construction task.

    Cubes start in a pile; the arm grasps each one in order and places it at its
    assigned dome target. The bridge teleports the currently-held cube along the
    EE so the demo stays deterministic regardless of contact dynamics.
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
        self.gripper_qposadrs: list[int] = []
        self.gripper_dofadrs: list[int] = []
        self.gripper_actuator_ids: list[int] = []
        self.cube_qposadrs: list[int] = []
        self.cube_qveladrs: list[int] = []
        self.cube_body_ids: list[int] = []
        self.cube_geom_ids: list[int] = []
        self.cube_default_contype: list[int] = []
        self.cube_default_conaffinity: list[int] = []
        self.cube_targets: list[np.ndarray] = []
        self.cube_placed: list[bool] = []
        self.current_cube_idx = 0
        self.ee_site_id = 0
        self.target_site_id = 0
        self.targets = np.zeros(4, dtype=np.float32)
        self.gripper = 1.0
        self.holding = False
        self.success = False
        self.terminated = False
        self.total_reward = 0.0
        self.steps = 0
        self.max_steps = 4000

    async def reset(self, task_id: str, seed: int = 0,
                    scene_path: str = "mars_scene.xml",
                    pile_center: tuple[float, float] | None = None,
                    dome_center: tuple[float, float, float] | None = None) -> str:
        del task_id
        np.random.seed(seed)
        self._scene_path = scene_path
        self._pile_center = pile_center if pile_center is not None else PILE_CENTER_XY
        self._dome_center = dome_center if dome_center is not None else DOME_CENTER

        cwd = os.getcwd()
        try:
            os.chdir(SCENE_DIR)
            self.model = mujoco.MjModel.from_xml_path(scene_path)
        finally:
            os.chdir(cwd)
        self.data = mujoco.MjData(self.model)
        if self.render:
            try:
                self.renderer = mujoco.Renderer(
                    self.model,
                    height=CAMERA_HEIGHT,
                    width=CAMERA_WIDTH,
                )
            except Exception as exc:
                self.renderer = None
                print(f"[warn] MuJoCo renderer unavailable; returning black frames: {exc}")

        self.arm_qposadrs = [self.model.jnt_qposadr[self.model.joint(j).id] for j in ARM_JOINTS]
        self.arm_dofadrs = [self.model.jnt_dofadr[self.model.joint(j).id] for j in ARM_JOINTS]
        self.arm_actuator_ids = [self.model.actuator(a).id for a in ARM_ACTUATORS]
        self.gripper_qposadrs = [self.model.jnt_qposadr[self.model.joint(j).id] for j in GRIPPER_JOINTS]
        self.gripper_dofadrs = [self.model.jnt_dofadr[self.model.joint(j).id] for j in GRIPPER_JOINTS]
        self.gripper_actuator_ids = [self.model.actuator(a).id for a in GRIPPER_ACTUATORS]

        self.cube_qposadrs = []
        self.cube_qveladrs = []
        self.cube_body_ids = []
        self.cube_geom_ids = []
        self.cube_default_contype = []
        self.cube_default_conaffinity = []
        for i in range(NUM_CUBES):
            joint = self.model.joint(f"pick_cube_{i}_free").id
            self.cube_qposadrs.append(self.model.jnt_qposadr[joint])
            self.cube_qveladrs.append(self.model.jnt_dofadr[joint])
            self.cube_body_ids.append(self.model.body(f"pick_cube_{i}").id)
            geom_id = self.model.geom(f"pick_cube_{i}_geom").id
            self.cube_geom_ids.append(geom_id)
            self.cube_default_contype.append(int(self.model.geom_contype[geom_id]))
            self.cube_default_conaffinity.append(int(self.model.geom_conaffinity[geom_id]))

        self.cube_targets = _dome_positions(self._dome_center)
        self.cube_placed = [False] * NUM_CUBES
        self.current_cube_idx = 0
        self.arm_yaw_body_id = int(self.model.body("arm_yaw_body").id)

        self.ee_site_id = self.model.site("ee_site").id
        self.target_site_id = self.model.site("target_site").id

        self.targets[:] = np.array([0.0, -0.35, 0.65, -0.25], dtype=np.float32)
        self.gripper = 1.0
        self.holding = False
        self.success = False
        self.terminated = False
        self.total_reward = 0.0
        self.steps = 0

        pile = _pile_positions(self._pile_center)
        for i, pos in enumerate(pile):
            self._set_cube_pose_idx(i, pos)
        self._apply_targets()
        # Brief settle to resolve any tiny contact penetration, then re-pin cubes to
        # their commanded pile positions so the scripted IK targets match reality.
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)
        for i, pos in enumerate(pile):
            self._set_cube_pose_idx(i, pos)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        return f"Pick all {NUM_CUBES} crimson blocks from the pile and arrange them in a dome at the target pad."

    def step(self, action) -> None:
        self._require_ready()
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size < 5:
            raise ValueError(f"Expected 5D action [d_yaw,d_shoulder,d_elbow,d_wrist,gripper], got {action!r}")

        deltas = np.clip(action[:4], -1.0, 1.0) * DELTA_SCALE
        self.targets = np.clip(self.targets + deltas, self._arm_low(), self._arm_high())
        self.gripper = float(np.clip(self.gripper + 0.2 * float(action[4]), 0.0, 1.0))

        self._apply_targets()
        self._update_grasp()
        for _ in range(PHYSICS_STEPS_PER_ACTION):
            mujoco.mj_step(self.model, self.data)
            self._update_grasp()

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
        return {
            "observation/image": rgb,
            "observation/state": self._state().astype(np.float32),
        }, self.terminated

    def result(self) -> dict:
        return EpisodeResult(
            score=sum(self.cube_placed) / float(NUM_CUBES),
            success=self.success,
            total_reward=float(self.total_reward),
            placed_count=int(sum(self.cube_placed)),
        ).__dict__

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
        self.renderer = None
        self.data = None
        self.model = None

    def _require_ready(self) -> None:
        if self.model is None or self.data is None:
            raise RuntimeError("Bridge has not been reset. Call reset() first.")

    def _arm_low(self) -> np.ndarray:
        assert self.model is not None
        return np.array([self.model.jnt_range[self.model.joint(j).id, 0] for j in ARM_JOINTS], dtype=np.float32)

    def _arm_high(self) -> np.ndarray:
        assert self.model is not None
        return np.array([self.model.jnt_range[self.model.joint(j).id, 1] for j in ARM_JOINTS], dtype=np.float32)

    def _apply_targets(self) -> None:
        assert self.data is not None
        for actuator_id, target in zip(self.arm_actuator_ids, self.targets):
            self.data.ctrl[actuator_id] = float(target)
        left, right = self.gripper_actuator_ids
        opening = self.gripper
        left_pos = GRIPPER_CLOSED_LEFT * (1.0 - opening)
        right_pos = GRIPPER_CLOSED_RIGHT * (1.0 - opening)
        self.data.ctrl[left] = left_pos
        self.data.ctrl[right] = right_pos

        # Hold joint qpos directly so the scripted policy doesn't fight the actuator dynamics.
        for qposadr, dofadr, target in zip(self.arm_qposadrs, self.arm_dofadrs, self.targets):
            self.data.qpos[qposadr] = float(target)
            self.data.qvel[dofadr] = 0.0
        for qposadr, dofadr, pos in zip(self.gripper_qposadrs, self.gripper_dofadrs, [left_pos, right_pos]):
            self.data.qpos[qposadr] = float(pos)
            self.data.qvel[dofadr] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _current_cube_position(self) -> np.ndarray:
        assert self.data is not None
        idx = min(self.current_cube_idx, NUM_CUBES - 1)
        return np.array(self.data.xpos[self.cube_body_ids[idx]], dtype=np.float64)

    def arm_yaw_body_world(self) -> np.ndarray:
        """Current world position of the arm's yaw joint. Tracks the rover when mobile."""
        assert self.data is not None
        mujoco.mj_forward(self.model, self.data)
        return np.array(self.data.xpos[self.arm_yaw_body_id], dtype=np.float64)

    def _update_grasp(self) -> None:
        assert self.data is not None
        if self.current_cube_idx >= NUM_CUBES:
            return
        mujoco.mj_forward(self.model, self.data)
        ee = self.data.site_xpos[self.ee_site_id]
        idx = self.current_cube_idx
        cube = self.data.xpos[self.cube_body_ids[idx]]
        if self.holding:
            if self.gripper > 0.45:
                # Release: re-enable cube collision so it can rest on others, zero
                # velocity, and mark placed if it landed within target tolerance.
                self.holding = False
                self._set_cube_collidable(idx, True)
                qvel0 = self.cube_qveladrs[idx]
                self.data.qvel[qvel0:qvel0 + 6] = 0.0
                target_xy = self.cube_targets[idx][:2]
                if np.linalg.norm(cube[:2] - target_xy) <= TARGET_RADIUS:
                    self.cube_placed[idx] = True
                # Advance regardless: if the release missed, looping again would just
                # try to grasp empty air.
                self.current_cube_idx += 1
            else:
                self._set_cube_pose_idx(idx, ee + HELD_OFFSET)
            return
        if self.gripper < 0.25 and np.linalg.norm(ee - cube) <= PICK_DISTANCE:
            # Grasp: disable the held cube's collision so it can travel through the
            # pile / over the dome without bulldozing other cubes.
            self.holding = True
            self._set_cube_collidable(idx, False)
            self._set_cube_pose_idx(idx, ee + HELD_OFFSET)

    def _set_cube_collidable(self, idx: int, collidable: bool) -> None:
        assert self.model is not None
        geom_id = self.cube_geom_ids[idx]
        if collidable:
            self.model.geom_contype[geom_id] = self.cube_default_contype[idx]
            self.model.geom_conaffinity[geom_id] = self.cube_default_conaffinity[idx]
        else:
            self.model.geom_contype[geom_id] = 0
            self.model.geom_conaffinity[geom_id] = 0

    def _set_cube_pose_idx(self, idx: int, pos: np.ndarray) -> None:
        assert self.data is not None
        qpos0 = self.cube_qposadrs[idx]
        qvel0 = self.cube_qveladrs[idx]
        self.data.qpos[qpos0:qpos0 + 3] = pos
        self.data.qpos[qpos0 + 3:qpos0 + 7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[qvel0:qvel0 + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _state(self) -> np.ndarray:
        assert self.data is not None
        joints = np.array([self.data.qpos[q] for q in self.arm_qposadrs], dtype=np.float32)
        ee = self.data.site_xpos[self.ee_site_id]
        cube = self._current_cube_position()
        target = self.data.site_xpos[self.target_site_id]
        return np.concatenate([
            joints,
            np.array([self.gripper], dtype=np.float32),
            ee.astype(np.float32),
            cube.astype(np.float32),
            target.astype(np.float32),
            np.array([float(self.holding), float(sum(self.cube_placed))], dtype=np.float32),
        ])

    def _compute_reward(self) -> float:
        placed = sum(self.cube_placed)
        bonus = 5.0 if self._is_success() else 0.0
        return float(placed - 0.01 + (0.1 if self.holding else 0.0) + bonus)

    def _is_success(self) -> bool:
        return all(self.cube_placed)


async def _smoke_test(steps: int, scripted: bool, render: bool) -> None:
    bridge = MarsArmPickPlaceBridge(render=render)
    prompt = await bridge.reset(task_id="arm-smoke", seed=0)
    print(f"prompt: {prompt}")

    actions = _scripted_actions() if scripted else [np.zeros(5, dtype=np.float32)] * steps
    print(f"[scripted actions] {len(actions)} total")
    for i, action in enumerate(actions[:steps]):
        bridge.step(action)
        obs, terminated = bridge.get_observation()
        state = obs["observation/state"]
        if (i + 1) % 25 == 0 or terminated:
            print(
                f"step={i + 1} placed={int(state[15])}/{NUM_CUBES} "
                f"holding={state[14]:.0f} terminated={terminated}"
            )
        if terminated:
            break

    print(f"result: {bridge.result()}")
    bridge.close()


def _repeat(action: list[float], count: int) -> list[np.ndarray]:
    return [np.array(action, dtype=np.float32) for _ in range(count)]


def _move_to_pose_actions(
    start: np.ndarray,
    goal: np.ndarray,
    steps: int,
    gripper: float,
    delta_scale: np.ndarray = DELTA_SCALE,
) -> tuple[list[np.ndarray], np.ndarray]:
    actions: list[np.ndarray] = []
    current = start.astype(np.float32).copy()
    for _ in range(steps):
        remaining = goal.astype(np.float32) - current
        delta = remaining / max(1, steps - len(actions))
        action = np.zeros(5, dtype=np.float32)
        action[:4] = np.clip(delta / delta_scale, -1.0, 1.0)
        action[4] = gripper
        current += action[:4] * delta_scale
        actions.append(action)
    return actions, current


def _move_to(
    current: np.ndarray,
    goal: np.ndarray,
    gripper: float,
    pad: int = 2,
    delta_scale: np.ndarray = DELTA_SCALE,
):
    """Wrap _move_to_pose_actions, sizing the move so DELTA_SCALE clipping never starves it."""
    diff = np.abs(goal[:4] - current[:4]) / delta_scale
    steps = max(3, int(math.ceil(float(np.max(diff)))) + pad)
    return _move_to_pose_actions(current, goal, steps, gripper, delta_scale=delta_scale)


def _scripted_actions() -> list[np.ndarray]:
    """Generate the full 20-cube pick-and-place sequence.

    Per cube: approach → descend → close → lift → translate → descend → open → lift.
    Each move is sized so DELTA_SCALE-clipped per-step deltas can actually reach the
    next IK waypoint.
    """
    pile = _pile_positions()
    targets = _dome_positions()

    current = np.array([0.0, -0.35, 0.65, -0.25], dtype=np.float32)
    actions: list[np.ndarray] = []

    approach_dz = 0.28  # EE this far above cube center for transit (raised for visual clearance)
    grasp_dz = 0.08     # EE this far above cube center to grasp / release

    for i in range(NUM_CUBES):
        pickup_xyz = pile[i]
        target_xyz = targets[i]

        pickup_high = _ik_top_down(pickup_xyz + np.array([0.0, 0.0, approach_dz]))
        pickup_low = _ik_top_down(pickup_xyz + np.array([0.0, 0.0, grasp_dz]))
        target_high = _ik_top_down(target_xyz + np.array([0.0, 0.0, approach_dz]))
        target_low = _ik_top_down(target_xyz + np.array([0.0, 0.0, grasp_dz]))

        if any(p is None for p in (pickup_high, pickup_low, target_high, target_low)):
            raise RuntimeError(f"Unreachable waypoint for cube {i}: pile={pickup_xyz}, target={target_xyz}")

        move, current = _move_to(current, pickup_high, 1.0)
        actions += move
        move, current = _move_to(current, pickup_low, 1.0)
        actions += move
        # 5 close ticks: gripper 1.0 → 0.0, crossing the 0.25 grasp threshold by tick 4.
        actions += _repeat([0.0, 0.0, 0.0, 0.0, -1.0], 5)
        move, current = _move_to(current, pickup_high, -1.0)
        actions += move
        move, current = _move_to(current, target_high, -1.0)
        actions += move
        move, current = _move_to(current, target_low, -1.0)
        actions += move
        # 5 open ticks: gripper 0.0 → 1.0, crossing the 0.45 release threshold by tick 3.
        actions += _repeat([0.0, 0.0, 0.0, 0.0, 1.0], 5)
        move, current = _move_to(current, target_high, 1.0)
        actions += move

    return actions


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the HUD pick/place arm bridge.")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--scripted-pick-place", action="store_true")
    parser.add_argument("--render", action="store_true", help="Attempt real MuJoCo offscreen render.")
    args = parser.parse_args()
    asyncio.run(_smoke_test(args.steps, args.scripted_pick_place, args.render))


if __name__ == "__main__":
    main()
