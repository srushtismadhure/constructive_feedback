import argparse
import asyncio
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
GRIPPER_OPEN = 0.0
GRIPPER_CLOSED_LEFT = -0.032
GRIPPER_CLOSED_RIGHT = 0.032
PICK_DISTANCE = 0.16
TARGET_RADIUS = 0.16

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
                "success",
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


class MarsArmPickPlaceBridge(RobotBridge):
    """HUD RobotBridge for a simple fixed-base pick/place arm.

    action = [d_yaw, d_shoulder, d_elbow, d_wrist, gripper]
    The first four values are joint deltas in [-1, 1]. The gripper value closes
    when negative and opens when positive. A scripted grasp helper attaches the
    cube when closed near the end-effector, which keeps the task learnable.
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
        self.cube_qposadr = 0
        self.cube_qveladr = 0
        self.ee_site_id = 0
        self.target_site_id = 0
        self.cube_body_id = 0
        self.targets = np.zeros(4, dtype=np.float32)
        self.gripper = 1.0
        self.holding = False
        self.success = False
        self.terminated = False
        self.total_reward = 0.0
        self.steps = 0
        self.max_steps = 300

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
        cube_joint = self.model.joint("pick_cube_free").id
        self.cube_qposadr = self.model.jnt_qposadr[cube_joint]
        self.cube_qveladr = self.model.jnt_dofadr[cube_joint]
        self.ee_site_id = self.model.site("ee_site").id
        self.target_site_id = self.model.site("target_site").id
        self.cube_body_id = self.model.body("pick_cube").id

        self.targets[:] = np.array([0.0, -0.35, 0.65, -0.25], dtype=np.float32)
        self.gripper = 1.0
        self.holding = False
        self.success = False
        self.terminated = False
        self.total_reward = 0.0
        self.steps = 0

        self._set_cube_pose(np.array([1.80, -0.5, 3.39], dtype=np.float64))
        self._apply_targets()
        for _ in range(200):
            mujoco.mj_step(self.model, self.data)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        return "Pick the blue cube and place it on the green target pad."

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
            score=1.0 if self.success else 0.0,
            success=self.success,
            total_reward=float(self.total_reward),
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

        # Keep this primitive HUD task stable and deterministic. The XML still
        # exposes real MuJoCo position actuators, but the bridge holds their
        # target qpos directly so a learned/scripted policy can focus on the
        # pick/place action rather than fighting a tiny arm's settling dynamics.
        for qposadr, dofadr, target in zip(self.arm_qposadrs, self.arm_dofadrs, self.targets):
            self.data.qpos[qposadr] = float(target)
            self.data.qvel[dofadr] = 0.0
        for qposadr, dofadr, pos in zip(self.gripper_qposadrs, self.gripper_dofadrs, [left_pos, right_pos]):
            self.data.qpos[qposadr] = float(pos)
            self.data.qvel[dofadr] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _update_grasp(self) -> None:
        assert self.data is not None
        mujoco.mj_forward(self.model, self.data)
        ee = self.data.site_xpos[self.ee_site_id]
        cube = self.data.xpos[self.cube_body_id]
        if self.holding:
            if self.gripper > 0.45:
                self.holding = False
                self.data.qvel[self.cube_qveladr : self.cube_qveladr + 6] = 0.0
            else:
                self._set_cube_pose(ee + np.array([0.0, 0.0, -0.075]))
            return
        if self.gripper < 0.25 and np.linalg.norm(ee - cube) <= PICK_DISTANCE:
            self.holding = True
            self._set_cube_pose(ee + np.array([0.0, 0.0, -0.075]))

    def _set_cube_pose(self, pos: np.ndarray) -> None:
        assert self.data is not None
        self.data.qpos[self.cube_qposadr : self.cube_qposadr + 3] = pos
        self.data.qpos[self.cube_qposadr + 3 : self.cube_qposadr + 7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[self.cube_qveladr : self.cube_qveladr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _state(self) -> np.ndarray:
        assert self.data is not None
        joints = np.array([self.data.qpos[q] for q in self.arm_qposadrs], dtype=np.float32)
        ee = self.data.site_xpos[self.ee_site_id]
        cube = self.data.xpos[self.cube_body_id]
        target = self.data.site_xpos[self.target_site_id]
        return np.concatenate([
            joints,
            np.array([self.gripper], dtype=np.float32),
            ee.astype(np.float32),
            cube.astype(np.float32),
            target.astype(np.float32),
            np.array([float(self.holding), float(self.success)], dtype=np.float32),
        ])

    def _compute_reward(self) -> float:
        assert self.data is not None
        cube = self.data.xpos[self.cube_body_id]
        target = self.data.site_xpos[self.target_site_id]
        dist = np.linalg.norm(cube[:2] - target[:2])
        return float(-dist + (0.25 if self.holding else 0.0) + (5.0 if self._is_success() else 0.0))

    def _is_success(self) -> bool:
        assert self.data is not None
        cube = self.data.xpos[self.cube_body_id]
        target = self.data.site_xpos[self.target_site_id]
        xy_ok = np.linalg.norm(cube[:2] - target[:2]) <= TARGET_RADIUS
        z_ok = abs(float(cube[2] - target[2])) <= 0.18
        return bool((not self.holding) and xy_ok and z_ok)


async def _smoke_test(steps: int, scripted: bool, render: bool) -> None:
    bridge = MarsArmPickPlaceBridge(render=render)
    prompt = await bridge.reset(task_id="arm-smoke", seed=0)
    print(f"prompt: {prompt}")

    actions = _scripted_actions() if scripted else [np.zeros(5, dtype=np.float32)] * steps
    for i, action in enumerate(actions[:steps]):
        bridge.step(action)
        obs, terminated = bridge.get_observation()
        state = obs["observation/state"]
        print(
            f"step={i + 1} cube={np.round(state[8:11], 3).tolist()} "
            f"target={np.round(state[11:14], 3).tolist()} holding={state[14]:.0f} "
            f"success={state[15]:.0f} terminated={terminated}"
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


def _scripted_actions() -> list[np.ndarray]:
    # Top-down grasp: shoulder+elbow+wrist sum to 0 keeps the gripper vertical.
    start = np.array([0.0, -0.35, 0.65, -0.25], dtype=np.float32)
    cube_pose = np.array([0.0, -0.166, 1.690, -1.524], dtype=np.float32)
    target_pose = np.array([0.70, -0.166, 1.690, -1.524], dtype=np.float32)

    actions: list[np.ndarray] = []
    move, current = _move_to_pose_actions(start, cube_pose, 34, 1.0)
    actions += move
    actions += _repeat([0.0, 0.0, 0.0, 0.0, -1.0], 8)
    move, current = _move_to_pose_actions(current, target_pose, 28, -1.0)
    actions += move
    actions += _repeat([0.0, 0.0, 0.0, 0.0, 1.0], 8)
    actions += _repeat([0.0, 0.0, 0.0, 0.0, 1.0], 4)
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
