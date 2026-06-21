import argparse
import asyncio
import math
import os
from dataclasses import dataclass

import mujoco
import numpy as np

try:
    from hud.environment.robot import RobotBridge
except ImportError:  # Allows local smoke tests before installing hud-python[robot].
    class RobotBridge:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            pass


SCENE_DIR = os.path.dirname(os.path.abspath(__file__))
SCENE_XML = os.path.join(SCENE_DIR, "mars_scene.xml")

ROBOT_BODY = "simple_rover"
ROBOT_JOINT = "rover_free"
TERRAIN_GEOM_GROUP = 1
SURFACE_CLEARANCE = 0.24
MAX_WHEEL_TORQUE = 1.0
PHYSICS_STEPS_PER_ACTION = 20
CAMERA_HEIGHT = 256
CAMERA_WIDTH = 256

# Drive-to-goal task: dense progress reward + a bonus when the rover arrives.
GOAL_TOLERANCE = 0.4          # metres; within this counts as "reached"
DEFAULT_GOAL = (0.0, 1.5)     # standalone HUD/demo goal when none is supplied
SUCCESS_BONUS = 1.0           # one-off reward added on arrival

CONTRACT = {
    "robot_type": "mars_wheel_rover_mujoco",
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
            "shape": [8],
            "names": ["x", "y", "z", "yaw", "vx", "vy", "vz", "yaw_rate"],
        },
        "action": {
            "role": "action",
            "dtype": "float32",
            "shape": [2],
            "names": ["forward_speed", "turn_speed"],
        },
    },
}


def yaw_to_quat(yaw: float) -> np.ndarray:
    half = 0.5 * yaw
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float64)


def quat_to_yaw(quat: np.ndarray) -> float:
    w, x, y, z = quat
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


@dataclass
class EpisodeResult:
    score: float = 0.0
    success: bool = False
    total_reward: float = 0.0


class MarsMujocoBridge(RobotBridge):
    """HUD RobotBridge for the Mars MuJoCo scene.

    Actions are [forward_speed, turn_speed] in [-1, 1]. The bridge maps them
    to left/right wheel velocity actuator targets on a primitive rover.
    """

    def __init__(self, render: bool = True):
        super().__init__()
        self.render = render
        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None
        self.renderer: mujoco.Renderer | None = None
        self.robot_qposadr = 0
        self.robot_qveladr = 0
        self.wheel_actuator_ids: list[int] = []
        self.terrain_geomgroup = np.zeros(6, dtype=np.uint8)
        self.terrain_geomgroup[TERRAIN_GEOM_GROUP] = 1
        self.ray_down = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        self.terminated = False
        self.success = False
        self.total_reward = 0.0
        self.steps = 0
        self.max_steps = 500
        # Drive-to-goal task state.
        self.goal_xy: tuple[float, float] = DEFAULT_GOAL
        self.goal_tol = GOAL_TOLERANCE
        self.prev_dist = 0.0

    async def reset(self, task_id: str, seed: int = 0, goal_xy: tuple[float, float] | None = None) -> str:
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

        joint_id = self.model.joint(ROBOT_JOINT).id
        self.robot_qposadr = self.model.jnt_qposadr[joint_id]
        self.robot_qveladr = self.model.jnt_dofadr[joint_id]
        self.wheel_actuator_ids = [
            self.model.actuator(name).id
            for name in ("wheel_fl_motor", "wheel_fr_motor", "wheel_rl_motor", "wheel_rr_motor")
        ]

        self.terminated = False
        self.success = False
        self.total_reward = 0.0
        self.steps = 0

        self._set_robot_pose(1.25, -0.5, 0.0)
        for _ in range(200):
            mujoco.mj_step(self.model, self.data)
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self.goal_xy = goal_xy if goal_xy is not None else DEFAULT_GOAL
        self.prev_dist = self._dist_to_goal()
        return (
            f"Drive the Mars rover to ({self.goal_xy[0]:.2f}, {self.goal_xy[1]:.2f}) "
            f"using actions [forward_speed, turn_speed]."
        )

    def step(self, action) -> None:
        self._require_ready()
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size < 2:
            raise ValueError(
                f"Expected action with 2 values [forward_speed, turn_speed], got {action!r}"
            )
        assert self.model is not None
        assert self.data is not None

        forward = float(np.clip(action[0], -1.0, 1.0))
        turn = float(np.clip(action[1], -1.0, 1.0))
        left_torque = np.clip(forward - turn, -1.0, 1.0) * MAX_WHEEL_TORQUE
        right_torque = np.clip(forward + turn, -1.0, 1.0) * MAX_WHEEL_TORQUE

        fl, fr, rl, rr = self.wheel_actuator_ids
        self.data.ctrl[fl] = left_torque
        self.data.ctrl[rl] = left_torque
        self.data.ctrl[fr] = right_torque
        self.data.ctrl[rr] = right_torque

        for _ in range(PHYSICS_STEPS_PER_ACTION):
            mujoco.mj_step(self.model, self.data)

        self.steps += 1

        # Dense progress reward: how much closer to the goal this action got us.
        dist = self._dist_to_goal()
        reward = self.prev_dist - dist
        self.prev_dist = dist
        if dist <= self.goal_tol and not self.success:
            reward += SUCCESS_BONUS
            self.success = True
            self.terminated = True
        self.total_reward += reward

        if self.steps >= self.max_steps:
            self.terminated = True

    def get_observation(self):
        self._require_ready()
        assert self.model is not None
        assert self.data is not None
        if self.renderer is None:
            rgb = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
        else:
            self.renderer.update_scene(self.data)
            rgb = self.renderer.render().astype(np.uint8)
        state = np.array(self._robot_state(), dtype=np.float32)

        return {
            "observation/image": rgb,
            "observation/state": state,
        }, self.terminated

    def result(self) -> dict:
        return EpisodeResult(
            score=1.0 if self.success else 0.0,
            success=self.success,
            total_reward=self.total_reward,
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

    def _robot_pose(self) -> tuple[float, float, float, float]:
        assert self.data is not None
        pos = self.data.qpos[self.robot_qposadr : self.robot_qposadr + 3]
        quat = self.data.qpos[self.robot_qposadr + 3 : self.robot_qposadr + 7]
        return float(pos[0]), float(pos[1]), float(pos[2]), quat_to_yaw(quat)

    def set_goal(self, x: float, y: float) -> None:
        """Point the drive-to-goal task at a new (x, y) without re-loading the scene.

        Clears the per-goal success/terminated flags and re-baselines the progress
        distance, so one bridge can serve many sequential goals (e.g. one per
        orchestration task). Cumulative ``total_reward`` and ``steps`` are left intact.
        """
        self.goal_xy = (float(x), float(y))
        self.success = False
        self.terminated = False
        self.prev_dist = self._dist_to_goal()

    def _dist_to_goal(self) -> float:
        x, y, _, _ = self._robot_pose()
        return math.hypot(x - self.goal_xy[0], y - self.goal_xy[1])

    def _robot_state(self) -> tuple[float, float, float, float, float, float, float, float]:
        assert self.data is not None
        x, y, z, yaw = self._robot_pose()
        vel = self.data.qvel[self.robot_qveladr : self.robot_qveladr + 6]
        return x, y, z, yaw, float(vel[0]), float(vel[1]), float(vel[2]), float(vel[5])

    def _set_robot_pose(self, x: float, y: float, yaw: float) -> None:
        assert self.model is not None
        assert self.data is not None
        z = self._surface_z(x, y) + SURFACE_CLEARANCE
        self.data.qpos[self.robot_qposadr : self.robot_qposadr + 3] = [x, y, z]
        self.data.qpos[self.robot_qposadr + 3 : self.robot_qposadr + 7] = yaw_to_quat(yaw)
        self.data.qvel[self.robot_qveladr : self.robot_qveladr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _surface_z(self, x: float, y: float) -> float:
        assert self.model is not None
        assert self.data is not None
        geomid = np.array([-1], dtype=np.int32)
        origin = np.array([x, y, 20.0], dtype=np.float64)
        dist = mujoco.mj_ray(
            self.model,
            self.data,
            origin,
            self.ray_down,
            self.terrain_geomgroup,
            1,
            -1,
            geomid,
        )
        if dist < 0:
            return self.data.qpos[self.robot_qposadr + 2] - SURFACE_CLEARANCE
        return float(origin[2] - dist)


async def _smoke_test(steps: int, render: bool) -> None:
    bridge = MarsMujocoBridge(render=render)
    prompt = await bridge.reset(task_id="smoke", seed=0)
    print(f"prompt: {prompt}")

    for i in range(steps):
        bridge.step(np.array([0.6, 0.15], dtype=np.float32))
        obs, terminated = bridge.get_observation()
        print(
            f"step={i + 1} state={np.round(obs['observation/state'], 3).tolist()} "
            f"image_shape={obs['observation/image'].shape} terminated={terminated}"
        )
        if terminated:
            break

    print(f"result: {bridge.result()}")
    bridge.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the HUD MuJoCo robot bridge.")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--render", action="store_true", help="Attempt real MuJoCo offscreen render.")
    args = parser.parse_args()
    asyncio.run(_smoke_test(args.steps, args.render))


if __name__ == "__main__":
    main()
