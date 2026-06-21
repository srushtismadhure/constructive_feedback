"""Rover navigation primitives for the arm-on-rover pick-and-place demo.

This is the scaffolding for task #3 (rover drives pile → arm work zone → dome).
The dome-building demo in ``hud_arm_bridge.py`` still uses a stationary rover
(``simple_rover`` welded to world). To drive the rover, the XML needs a
``freejoint`` re-added under ``simple_rover``; until then, ``RoverDriveController``
operates on a model that has one.

Usage sketch (next iteration will integrate this with the arm bridge)::

    # Scene XML needs:
    #   <body name="simple_rover" pos="0 0 3.55">
    #     <freejoint name="rover_free"/>
    #     ...
    # Wheel geoms need contype=2 conaffinity=2 (so they grip the ground but
    # pass through cubes), and the drive plane needs conaffinity=3.

    controller = RoverDriveController(model, data)
    while not controller.at_goal(goal_xy, tol=0.2):
        controller.step_drive(goal_xy)
        mujoco.mj_step(model, data)

The choreographed pile→pick→dome→place loop will then be::

    1. ``controller.drive_to(pile_staging_xy)``     # near, not on, the pile
    2. arm bridge picks the next cube (current behaviour)
    3. ``controller.drive_to(dome_staging_xy)``
    4. arm bridge places the cube at its assigned dome target
    5. repeat for all 36 cubes

Tipping risk: the arm currently teleports its joint qpos, which can produce
non-physical impulses on the rover when collisions are enabled. The
integration step will either (a) freeze the arm at a "stowed" pose during
driving, or (b) drop the qpos teleport in favor of true actuator-driven motion.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

# Tuning constants for the pure-pursuit-ish differential drive controller.
MAX_FORWARD = 1.0     # normalized forward command (consumed by both wheels)
MAX_TURN = 1.0        # normalized differential turn command
HEADING_TOL = 0.14    # rad — start moving sooner instead of over-pivoting
PIVOT_TURN_GAIN = 2.0
FORWARD_GAIN = 2.5
WHEEL_TORQUE_CAP = 10.0  # swarm XML permits this higher cap for responsive travel


def _quat_to_yaw(quat: np.ndarray) -> float:
    """Extract yaw (rotation around +z) from a (w, x, y, z) quaternion."""
    w, x, y, z = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class RoverDriveController:
    """Differential-drive controller for the Mars rover.

    Expects a model whose rover body has a freejoint named ``rover_free`` and
    four wheel motors named ``wheel_{fl,fr,rl,rr}_motor``.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, prefix: str = ""):
        self.model = model
        self.data = data
        self.prefix = prefix
        joint_id = model.joint(f"{prefix}rover_free").id
        self.qposadr = int(model.jnt_qposadr[joint_id])
        self.wheel_actuator_ids = [
            int(model.actuator(n).id)
            for n in (
                f"{prefix}wheel_fl_motor",
                f"{prefix}wheel_fr_motor",
                f"{prefix}wheel_rl_motor",
                f"{prefix}wheel_rr_motor",
            )
        ]

    def rover_pose(self) -> tuple[float, float, float]:
        """Return (x, y, yaw) of the rover in the world frame."""
        x = float(self.data.qpos[self.qposadr])
        y = float(self.data.qpos[self.qposadr + 1])
        quat = self.data.qpos[self.qposadr + 3 : self.qposadr + 7]
        return x, y, _quat_to_yaw(quat)

    def distance_to(self, goal_xy: tuple[float, float]) -> float:
        x, y, _ = self.rover_pose()
        return float(math.hypot(goal_xy[0] - x, goal_xy[1] - y))

    def at_goal(self, goal_xy: tuple[float, float], tol: float = 0.2) -> bool:
        return self.distance_to(goal_xy) <= tol

    def step_drive(self, goal_xy: tuple[float, float]) -> tuple[float, float]:
        """Compute and apply wheel torques to drive toward goal_xy. Returns the
        (forward, turn) command actually issued (clipped, for diagnostics).
        """
        x, y, yaw = self.rover_pose()
        dx, dy = goal_xy[0] - x, goal_xy[1] - y
        target_heading = math.atan2(dy, dx)
        heading_err = ((target_heading - yaw + math.pi) % (2 * math.pi)) - math.pi

        if abs(heading_err) > HEADING_TOL:
            # Pivot in place. Use bang-bang (full torque) until aligned; proportional
            # control's diminishing returns near setpoint kills pivot rate against the
            # rover's mass + friction.
            forward = 0.0
            turn = MAX_TURN if heading_err > 0 else -MAX_TURN
        else:
            dist = math.hypot(dx, dy)
            forward = float(np.clip(FORWARD_GAIN * dist, -MAX_FORWARD, MAX_FORWARD))
            turn = float(np.clip(PIVOT_TURN_GAIN * heading_err, -MAX_TURN, MAX_TURN))

        # Differential drive: left wheels = forward - turn, right wheels = forward + turn.
        left = float(np.clip(forward - turn, -1.0, 1.0)) * WHEEL_TORQUE_CAP
        right = float(np.clip(forward + turn, -1.0, 1.0)) * WHEEL_TORQUE_CAP
        fl, fr, rl, rr = self.wheel_actuator_ids
        self.data.ctrl[fl] = left
        self.data.ctrl[rl] = left
        self.data.ctrl[fr] = right
        self.data.ctrl[rr] = right
        return forward, turn

    def stop(self) -> None:
        for aid in self.wheel_actuator_ids:
            self.data.ctrl[aid] = 0.0


def drive_to(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    goal_xy: tuple[float, float],
    tol: float = 0.2,
    max_steps: int = 2000,
    physics_substeps: int = 4,
) -> bool:
    """Drive the rover to ``goal_xy``. Returns True if it arrived within ``tol``.

    Standalone helper for use outside the arm bridge — runs physics itself.
    """
    controller = RoverDriveController(model, data)
    for _ in range(max_steps):
        if controller.at_goal(goal_xy, tol):
            controller.stop()
            for _ in range(physics_substeps * 5):
                mujoco.mj_step(model, data)
            return controller.at_goal(goal_xy, tol)
        controller.step_drive(goal_xy)
        for _ in range(physics_substeps):
            mujoco.mj_step(model, data)
    controller.stop()
    return controller.at_goal(goal_xy, tol)
