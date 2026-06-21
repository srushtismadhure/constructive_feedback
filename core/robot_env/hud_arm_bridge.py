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
GRIPPER_CLOSED_LEFT = -0.032
GRIPPER_CLOSED_RIGHT = 0.032
PICK_DISTANCE = 0.16
TARGET_RADIUS = 0.12

NUM_CUBES = 20

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


def _pile_positions() -> list[np.ndarray]:
    """20 cube positions in a 4-column x 5-row grid centered at (1.65, -0.70, 3.39)."""
    positions = []
    for i in range(NUM_CUBES):
        col = i // 5
        row = i % 5
        x = 1.47 + col * 0.12
        y = -0.94 + row * 0.12
        positions.append(np.array([x, y, 3.39], dtype=np.float64))
    return positions


def _dome_positions() -> list[np.ndarray]:
    """20 dome positions forming concentric circles at z=3.39.

    12 outer (r=0.20) + 6 middle (r=0.10) + 2 center. Ordered outer-first so the
    arm builds the foundation ring before filling inward.
    """
    cx, cy, cz = 1.64, -0.11, 3.39
    positions: list[np.ndarray] = []
    for i in range(12):
        angle = 2 * math.pi * i / 12
        positions.append(np.array([cx + 0.20 * math.cos(angle),
                                   cy + 0.20 * math.sin(angle),
                                   cz], dtype=np.float64))
    for i in range(6):
        angle = math.pi / 6 + 2 * math.pi * i / 6
        positions.append(np.array([cx + 0.10 * math.cos(angle),
                                   cy + 0.10 * math.sin(angle),
                                   cz], dtype=np.float64))
    positions.append(np.array([cx - 0.05, cy, cz], dtype=np.float64))
    positions.append(np.array([cx + 0.05, cy, cz], dtype=np.float64))
    return positions


def _ik_top_down(ee_world: np.ndarray) -> np.ndarray | None:
    """Analytic IK: place the EE at ee_world with the gripper pointing straight down
    (shoulder+elbow+wrist = 0). Returns [yaw, shoulder, elbow, wrist] or None if unreachable.
    """
    rel = ee_world - ARM_YAW_BODY_WORLD
    x_rel, y_rel, z_rel = float(rel[0]), float(rel[1]), float(rel[2])
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
        self.max_steps = 3000

    async def reset(self, task_id: str, seed: int = 0) -> str:
        del task_id
        np.random.seed(seed)

        cwd = os.getcwd()
        try:
            os.chdir(SCENE_DIR)
            self.model = mujoco.MjModel.from_xml_path("mars_scene.xml")
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
        for i in range(NUM_CUBES):
            joint = self.model.joint(f"pick_cube_{i}_free").id
            self.cube_qposadrs.append(self.model.jnt_qposadr[joint])
            self.cube_qveladrs.append(self.model.jnt_dofadr[joint])
            self.cube_body_ids.append(self.model.body(f"pick_cube_{i}").id)

        self.cube_targets = _dome_positions()
        self.cube_placed = [False] * NUM_CUBES
        self.current_cube_idx = 0

        self.ee_site_id = self.model.site("ee_site").id
        self.target_site_id = self.model.site("target_site").id

        self.targets[:] = np.array([0.0, -0.35, 0.65, -0.25], dtype=np.float32)
        self.gripper = 1.0
        self.holding = False
        self.success = False
        self.terminated = False
        self.total_reward = 0.0
        self.steps = 0

        pile = _pile_positions()
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
                # Release: zero velocity, mark placed if within target tolerance.
                self.holding = False
                qvel0 = self.cube_qveladrs[idx]
                self.data.qvel[qvel0:qvel0 + 6] = 0.0
                target_xy = self.cube_targets[idx][:2]
                if np.linalg.norm(cube[:2] - target_xy) <= TARGET_RADIUS:
                    self.cube_placed[idx] = True
                # Advance to the next cube regardless: if the release missed, we'd
                # otherwise loop forever trying to grasp empty air.
                self.current_cube_idx += 1
            else:
                self._set_cube_pose_idx(idx, ee + HELD_OFFSET)
            return
        if self.gripper < 0.25 and np.linalg.norm(ee - cube) <= PICK_DISTANCE:
            self.holding = True
            self._set_cube_pose_idx(idx, ee + HELD_OFFSET)

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
) -> tuple[list[np.ndarray], np.ndarray]:
    actions: list[np.ndarray] = []
    current = start.astype(np.float32).copy()
    for _ in range(steps):
        remaining = goal.astype(np.float32) - current
        delta = remaining / max(1, steps - len(actions))
        action = np.zeros(5, dtype=np.float32)
        action[:4] = np.clip(delta / DELTA_SCALE, -1.0, 1.0)
        action[4] = gripper
        current += action[:4] * DELTA_SCALE
        actions.append(action)
    return actions, current


def _move_to(current: np.ndarray, goal: np.ndarray, gripper: float, pad: int = 2):
    """Wrap _move_to_pose_actions, sizing the move so DELTA_SCALE clipping never starves it."""
    diff = np.abs(goal[:4] - current[:4]) / DELTA_SCALE
    steps = max(3, int(math.ceil(float(np.max(diff)))) + pad)
    return _move_to_pose_actions(current, goal, steps, gripper)


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

    approach_dz = 0.20  # EE this far above cube center for transit
    grasp_dz = 0.05     # EE this far above cube center to grasp / release

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
