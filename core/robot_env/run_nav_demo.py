"""Mobile-rover construction demo.

The rover starts at the world origin, drives to a pile of crimson cubes at (3, -2),
picks one up with its top-mounted arm, drives across the map to the dome build site
at (-3, 2), places it, and repeats. Demonstrates rover navigation choreographed with
arm pick-and-place.

Run headless::

    python3 core/robot_env/run_nav_demo.py --cubes 4

With the passive viewer::

    mjpython core/robot_env/run_nav_demo.py --cubes 4 --viewer
"""
from __future__ import annotations

import argparse
import asyncio
import math
import time

import mujoco
import numpy as np

from hud_arm_bridge import (
    MarsArmPickPlaceBridge,
    _ik_top_down,
    _move_to,
    _pile_positions,
    _repeat,
)
from rover_nav import RoverDriveController, HEADING_TOL


PILE_CENTER = (3.0, -2.0)
DOME_CENTER = (-3.0, 2.0, 3.39)
# Rover sits 0.45m back from the site center; arm reaches +x. Closer staging
# keeps the demo's first few dome cubes inside the arm's outer reach annulus
# (even with the 0.06m drive tolerance slack).
PILE_STAGING = (PILE_CENTER[0] - 0.45, PILE_CENTER[1])
DOME_STAGING = (DOME_CENTER[0] - 0.45, DOME_CENTER[1])
# Approach waypoints: drive through these first so the rover ends the final segment
# heading in +x (toward the pile / dome). Each approach is ~2m back along -x from
# the staging point, so the final leg is a long straight in +x.
PILE_APPROACH = (PILE_STAGING[0] - 2.0, PILE_STAGING[1])
DOME_APPROACH = (DOME_STAGING[0] - 2.0, DOME_STAGING[1])
DRIVE_TOL = 0.18
ALIGN_TARGET_YAW = 0.0  # arm reaches in rover-local +x; we always finish driving facing +x

STOW_OPEN = np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # empty hand, gripper open
STOW_CLOSED = np.array([0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)  # holding, gripper closed


def _wrap_pi(angle: float) -> float:
    return ((angle + math.pi) % (2 * math.pi)) - math.pi


def drive_to_with_stowed_arm(bridge: MarsArmPickPlaceBridge,
                             nav: RoverDriveController,
                             goal_xy: tuple[float, float],
                             stow_action: np.ndarray,
                             tol: float = DRIVE_TOL,
                             max_steps: int = 2000,
                             brake_steps: int = 80,
                             tick=None) -> bool:
    """Drive the rover toward goal_xy while the arm holds its stow pose. After
    arrival (or timeout), runs a brake/settle period with zero wheel torque so
    rolling momentum decays before the arm operates. ``tick`` (if given) is
    invoked once per ``bridge.step`` so the viewer can sync and pace."""
    arrived = False
    for _ in range(max_steps):
        if nav.at_goal(goal_xy, tol):
            arrived = True
            break
        nav.step_drive(goal_xy)
        bridge.step(stow_action)
        if tick is not None:
            tick()
    nav.stop()
    # Brake & settle: zero ctrl + run physics so rover stops drifting.
    for _ in range(brake_steps):
        bridge.step(stow_action)
        if tick is not None:
            tick()
    return arrived or nav.at_goal(goal_xy, tol)


def align_yaw_with_stowed_arm(bridge: MarsArmPickPlaceBridge,
                              nav: RoverDriveController,
                              target_yaw: float,
                              stow_action: np.ndarray,
                              tol: float = 0.05,
                              max_steps: int = 1200) -> bool:
    """Pivot in place until the rover's yaw matches ``target_yaw`` (radians).

    Uses a low-gain proportional pivot with velocity damping (not bang-bang)
    because at full torque the wheels skid and the rover translates instead of
    rotating in place. The pivot torque is also capped to about half of the
    drive cap so the lateral skid stays small.
    """
    from rover_nav import WHEEL_TORQUE_CAP
    pivot_cap = WHEEL_TORQUE_CAP * 0.45
    fl, fr, rl, rr = nav.wheel_actuator_ids
    yaw_vel_dofadr = nav.qposadr + 5  # freejoint angular vel z is the 6th dof
    yaw_vel_dofadr = int(bridge.model.jnt_dofadr[bridge.model.joint("rover_free").id]) + 5
    for _ in range(max_steps):
        _, _, yaw = nav.rover_pose()
        err = _wrap_pi(target_yaw - yaw)
        yaw_vel = float(bridge.data.qvel[yaw_vel_dofadr])
        if abs(err) <= tol and abs(yaw_vel) < 0.05:
            nav.stop()
            return True
        # P + D: drive toward err, brake against current yaw velocity.
        turn = 4.0 * err - 0.6 * yaw_vel
        turn = float(np.clip(turn, -1.0, 1.0))
        torque = turn * pivot_cap
        bridge.data.ctrl[fl] = -torque
        bridge.data.ctrl[rl] = -torque
        bridge.data.ctrl[fr] = torque
        bridge.data.ctrl[rr] = torque
        bridge.step(stow_action)
    nav.stop()
    _, _, yaw = nav.rover_pose()
    return abs(_wrap_pi(target_yaw - yaw)) <= tol


def execute_arm_sequence(bridge: MarsArmPickPlaceBridge, actions: list[np.ndarray],
                         tick=None) -> None:
    """Run a list of arm actions through bridge.step. Wheel ctrl is zeroed first so
    the rover stays put. ``tick`` (if given) is invoked after each ``bridge.step``
    so the viewer can sync and pace."""
    fl, fr, rl, rr = (bridge.model.actuator(n).id for n in
                     ("wheel_fl_motor", "wheel_fr_motor", "wheel_rl_motor", "wheel_rr_motor"))
    for aid in (fl, fr, rl, rr):
        bridge.data.ctrl[aid] = 0.0
    for action in actions:
        bridge.step(action)
        if tick is not None:
            tick()


def pick_actions(bridge: MarsArmPickPlaceBridge, cube_idx: int,
                 nav: RoverDriveController) -> list[np.ndarray]:
    pile = _pile_positions(bridge._pile_center)
    pickup_xyz = pile[cube_idx]
    arm_yaw = bridge.arm_yaw_body_world()
    _, _, rover_yaw = nav.rover_pose()
    pickup_high = _ik_top_down(pickup_xyz + np.array([0.0, 0.0, 0.28]), arm_yaw, rover_yaw)
    pickup_low = _ik_top_down(pickup_xyz + np.array([0.0, 0.0, 0.08]), arm_yaw, rover_yaw)
    if pickup_high is None or pickup_low is None:
        raise RuntimeError(f"cube {cube_idx} unreachable from rover pose {arm_yaw} yaw={rover_yaw}")
    current = bridge.targets.copy()
    actions: list[np.ndarray] = []
    move, current = _move_to(current, pickup_high, 1.0)
    actions += move
    move, current = _move_to(current, pickup_low, 1.0)
    actions += move
    actions += _repeat([0.0, 0.0, 0.0, 0.0, -1.0], 5)
    move, current = _move_to(current, pickup_high, -1.0)
    actions += move
    return actions


def place_actions(bridge: MarsArmPickPlaceBridge, cube_idx: int,
                  nav: RoverDriveController) -> list[np.ndarray]:
    target_xyz = bridge.cube_targets[cube_idx]
    arm_yaw = bridge.arm_yaw_body_world()
    _, _, rover_yaw = nav.rover_pose()
    target_high = _ik_top_down(target_xyz + np.array([0.0, 0.0, 0.28]), arm_yaw, rover_yaw)
    target_low = _ik_top_down(target_xyz + np.array([0.0, 0.0, 0.08]), arm_yaw, rover_yaw)
    if target_high is None or target_low is None:
        raise RuntimeError(f"dome slot {cube_idx} unreachable from rover pose {arm_yaw} yaw={rover_yaw}")
    current = bridge.targets.copy()
    actions: list[np.ndarray] = []
    move, current = _move_to(current, target_high, -1.0)
    actions += move
    move, current = _move_to(current, target_low, -1.0)
    actions += move
    actions += _repeat([0.0, 0.0, 0.0, 0.0, 1.0], 5)
    move, current = _move_to(current, target_high, 1.0)
    actions += move
    return actions


async def run(num_cubes: int, use_viewer: bool, speed: float) -> None:
    bridge = MarsArmPickPlaceBridge(render=False)
    await bridge.reset(
        task_id="nav-demo",
        seed=0,
        scene_path="mars_scene_mobile.xml",
        pile_center=PILE_CENTER,
        dome_center=DOME_CENTER,
    )
    nav = RoverDriveController(bridge.model, bridge.data)
    bridge.max_steps = 100000  # disable the per-cube time bound during nav

    viewer = None
    if use_viewer:
        viewer = mujoco.viewer.launch_passive(bridge.model, bridge.data)
        viewer.cam.distance = 8.0
        viewer.cam.elevation = -28
        viewer.cam.azimuth = 145
        viewer.cam.lookat[:] = [0.0, 0.0, 3.5]

    # Each bridge.step advances PHYSICS_STEPS_PER_ACTION * timestep = 0.04 s of sim
    # time. For 1.0x playback the wall-clock gap between ticks should match. ``speed``
    # >1 plays the sim back faster than real-time; <1 slower.
    sim_dt_per_step = 0.040
    wall_dt = sim_dt_per_step / max(speed, 1e-6)

    last_tick = [time.time()]

    def tick() -> None:
        if viewer is None or not viewer.is_running():
            return
        viewer.sync()
        now = time.time()
        sleep_for = wall_dt - (now - last_tick[0])
        if sleep_for > 0:
            time.sleep(sleep_for)
        last_tick[0] = time.time()

    try:
        for cube_idx in range(num_cubes):
            print(f"\n=== cube {cube_idx + 1}/{num_cubes} ===")
            print("driving to pile (via approach)…")
            drive_to_with_stowed_arm(bridge, nav, PILE_APPROACH, STOW_OPEN,
                                     brake_steps=10, tick=tick)
            drive_to_with_stowed_arm(bridge, nav, PILE_STAGING, STOW_OPEN,
                                     tol=0.06, tick=tick)
            print(f"  arrived: {nav.rover_pose()}")
            print("  picking…")
            execute_arm_sequence(bridge, pick_actions(bridge, cube_idx, nav), tick=tick)

            print("driving to dome (via approach)…")
            drive_to_with_stowed_arm(bridge, nav, DOME_APPROACH, STOW_CLOSED,
                                     brake_steps=10, tick=tick)
            drive_to_with_stowed_arm(bridge, nav, DOME_STAGING, STOW_CLOSED,
                                     tol=0.06, tick=tick)
            print(f"  arrived: {nav.rover_pose()}")
            print("  placing…")
            execute_arm_sequence(bridge, place_actions(bridge, cube_idx, nav), tick=tick)

            placed = sum(bridge.cube_placed)
            print(f"  placed so far: {placed}/{cube_idx + 1}")

        print(f"\nDone. placed={sum(bridge.cube_placed)}/{num_cubes}, holding={bridge.holding}")
    finally:
        if viewer is not None:
            viewer.close()
        bridge.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Drive-and-build demo for the mobile rover.")
    parser.add_argument("--cubes", type=int, default=4)
    parser.add_argument("--viewer", action="store_true",
                        help="Launch the MuJoCo passive viewer (use with mjpython).")
    parser.add_argument("--speed", type=float, default=1.5,
                        help="Sim playback speed multiplier (only used with --viewer). "
                             "1.0 = real-time; 2.0 = 2x fast-forward.")
    args = parser.parse_args()
    asyncio.run(run(args.cubes, args.viewer, args.speed))


if __name__ == "__main__":
    main()
