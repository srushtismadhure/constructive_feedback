"""Three-rover swarm dome-build demo.

Each rover starts at its dome staging position (radius 1.0 around the origin,
at 270/150/30 deg), shares one pile at (0, 5), and builds its 120-degree
wedge of the 3-tier dome at the origin. Pile access is serialized by a
software ``pile_lock`` held by the bridge; everything else (driving to dome,
arm pick/place) is parallel across rovers.

Run headless::

    python3 core/robot_env/run_swarm_demo.py --cubes-per-rover 1

With the passive viewer::

    mjpython core/robot_env/run_swarm_demo.py --cubes-per-rover 4 --viewer
"""
from __future__ import annotations

import argparse
import math
import time
from enum import Enum, auto

import mujoco
import numpy as np

from hud_arm_bridge import (
    _ik_top_down,
    _move_to,
    _repeat,
)
from rover_nav import RoverDriveController
from swarm_bridge import (
    DOME_CENTER,
    DOME_STAGING_RADIUS,
    NUM_ROVERS,
    PILE_RADIUS,
    PILE_STAGING_RADIUS,
    ROVER_PREFIXES,
    SWARM_ARM_DELTA_SCALE,
    MarsSwarmBridge,
    _pile_center_for,
)


def _staging_at(radius: float, angle_deg: float) -> tuple[float, float]:
    a = math.radians(angle_deg)
    return radius * math.cos(a), radius * math.sin(a)


def _pile_staging_for(rover_idx: int) -> tuple[float, float]:
    angles_deg = [270.0, 150.0, 30.0]
    return _staging_at(PILE_STAGING_RADIUS, angles_deg[rover_idx])


def _dome_staging_for_target(target_xyz: np.ndarray) -> tuple[float, float]:
    """Choose a staging point outside the dome on the target's radial line.

    A fixed sector-centre staging point made the slots at either edge of a
    120-degree sector exceed arm reach once the dome radius increased.
    """
    angle = math.atan2(float(target_xyz[1] - DOME_CENTER[1]),
                       float(target_xyz[0] - DOME_CENTER[0]))
    return _staging_at(DOME_STAGING_RADIUS, math.degrees(angle))
DRIVE_TOL = 0.18
STAGING_TOL = 0.07  # tighter for the final approach so IK has a stable origin
APPROACH_DZ = 0.28
GRASP_DZ = 0.08

STOW_OPEN = np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)   # idle, gripper open
STOW_CLOSED = np.array([0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)  # transit, gripper closed
HOLD = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)         # arm freeze, gripper unchanged


class State(Enum):
    IDLE_AT_DOME = auto()
    DRIVING_TO_PILE = auto()
    PICKING = auto()
    DRIVING_TO_DOME = auto()
    PLACING = auto()
    DONE = auto()


class RoverAgent:
    """State machine for one rover. Owns its drive controller and its current
    action queue."""

    def __init__(self, idx: int, bridge: MarsSwarmBridge):
        self.idx = idx
        self.bridge = bridge
        self.unit = bridge.rovers[idx]
        self.nav = RoverDriveController(bridge.model, bridge.data, prefix=ROVER_PREFIXES[idx])
        self.state = State.IDLE_AT_DOME
        # Rover spawn IS the pile staging (radius PILE_STAGING_RADIUS). Dome
        # staging is the closer-to-origin spot at radius DOME_STAGING_RADIUS along
        # the same angle.
        self.pile_staging = _pile_staging_for(idx)
        self.action_queue: list[np.ndarray] = []
        self.action_idx: int = 0
        self.current_dome_target_idx: int | None = None
        self.current_pile_cube_idx: int | None = None

    # ---- one tick of the state machine ----
    def tick(self) -> np.ndarray:
        """Advance the state machine by one bridge.step. Returns the 5D arm action
        to be applied by the bridge this tick; wheel ctrl is written directly via
        ``self.nav``."""
        # IDLE_AT_DOME: each rover owns its own pile, so no lock — claim and go.
        if self.state == State.IDLE_AT_DOME:
            if not self.unit.dome_queue and not self.unit.holding:
                self.state = State.DONE
                self.nav.stop()
                return STOW_OPEN
            self.current_pile_cube_idx = self.bridge.claim_pile_cube_for(self.idx)
            if self.current_pile_cube_idx is None:
                self.state = State.DONE
                self.nav.stop()
                return STOW_OPEN
            self.state = State.DRIVING_TO_PILE
            # fall through

        if self.state == State.DRIVING_TO_PILE:
            if self.nav.at_goal(self.pile_staging, tol=STAGING_TOL):
                self.nav.stop()
                self.action_queue = self._build_pick_actions()
                self.action_idx = 0
                self.state = State.PICKING
                return STOW_OPEN
            self.nav.step_drive(self.pile_staging)
            return STOW_OPEN

        if self.state == State.PICKING:
            if self.action_idx < len(self.action_queue):
                action = self.action_queue[self.action_idx]
                self.action_idx += 1
                return action
            # Do not drive an ungrasped cube toward a dome target. Keep the
            # reservation and finish the planned pickup deterministically.
            # The arm pose is teleported by this demo, so a single-substep
            # proximity check is otherwise prone to missing a valid grasp.
            if not self.unit.holding:
                if not self.bridge.grasp_reserved_cube(self.unit):
                    raise RuntimeError(f"rover {self.idx}: lost its reserved pile cube")
            self.state = State.DRIVING_TO_DOME
            # fall through

        if self.state == State.DRIVING_TO_DOME:
            cube_idx = self.unit.held_cube_idx
            if cube_idx is None:
                raise RuntimeError(f"rover {self.idx}: driving to dome without a cube")
            target_idx = self.bridge.held_cube_target.get(cube_idx)
            if target_idx is None:
                raise RuntimeError(f"rover {self.idx}: held cube has no assigned dome target")
            dome_staging = _dome_staging_for_target(self.bridge.dome_targets[target_idx])
            if self.nav.at_goal(dome_staging, tol=STAGING_TOL):
                self.nav.stop()
                self.action_queue = self._build_place_actions()
                self.action_idx = 0
                self.state = State.PLACING
                return STOW_CLOSED
            self.nav.step_drive(dome_staging)
            return STOW_CLOSED

        if self.state == State.PLACING:
            if self.action_idx < len(self.action_queue):
                action = self.action_queue[self.action_idx]
                self.action_idx += 1
                return action
            # Done placing → loop.
            self.state = State.IDLE_AT_DOME
            self.current_pile_cube_idx = None
            self.current_dome_target_idx = None
            return STOW_OPEN

        # DONE
        self.nav.stop()
        return STOW_OPEN

    # ---- arm action sequence builders ----
    def _build_pick_actions(self) -> list[np.ndarray]:
        cube_idx = self.current_pile_cube_idx
        assert cube_idx is not None
        cube_world = np.array(self.bridge.data.xpos[self.bridge.cube_body_ids[cube_idx]], dtype=np.float64)
        arm_pos = self.unit.arm_yaw_body_world()
        _, _, yaw = self.nav.rover_pose()
        high = _ik_top_down(cube_world + np.array([0.0, 0.0, APPROACH_DZ]), arm_pos, yaw)
        low = _ik_top_down(cube_world + np.array([0.0, 0.0, GRASP_DZ]), arm_pos, yaw)
        if high is None or low is None:
            raise RuntimeError(
                f"rover {self.idx}: pile cube {cube_idx} unreachable "
                f"(arm={arm_pos.tolist()}, yaw={yaw:.3f}, cube={cube_world.tolist()})"
            )
        current = self.unit.targets.copy()
        actions: list[np.ndarray] = []
        move, current = _move_to(current, high, 1.0, delta_scale=SWARM_ARM_DELTA_SCALE)
        actions += move
        move, current = _move_to(current, low, 1.0, delta_scale=SWARM_ARM_DELTA_SCALE)
        actions += move
        actions += _repeat([0.0, 0.0, 0.0, 0.0, -1.0], 3)  # close gripper
        move, current = _move_to(current, high, -1.0, delta_scale=SWARM_ARM_DELTA_SCALE)
        actions += move
        return actions

    def _build_place_actions(self) -> list[np.ndarray]:
        # Hold the dome target index we placed at: the one popped onto held_cube_target.
        cube_idx = self.unit.held_cube_idx
        if cube_idx is None:
            # Grasp didn't engage — try the next slot in queue but bail this round.
            return _repeat([0.0, 0.0, 0.0, 0.0, 1.0], 3)
        target_idx = self.bridge.held_cube_target.get(cube_idx)
        if target_idx is None:
            return _repeat([0.0, 0.0, 0.0, 0.0, 1.0], 3)
        target_world = np.array(self.bridge.dome_targets[target_idx], dtype=np.float64)
        arm_pos = self.unit.arm_yaw_body_world()
        _, _, yaw = self.nav.rover_pose()
        high = _ik_top_down(target_world + np.array([0.0, 0.0, APPROACH_DZ]), arm_pos, yaw)
        low = _ik_top_down(target_world + np.array([0.0, 0.0, GRASP_DZ]), arm_pos, yaw)
        if high is None or low is None:
            raise RuntimeError(
                f"rover {self.idx}: dome slot {target_idx} unreachable "
                f"(arm={arm_pos.tolist()}, yaw={yaw:.3f})"
            )
        current = self.unit.targets.copy()
        actions: list[np.ndarray] = []
        move, current = _move_to(current, high, -1.0, delta_scale=SWARM_ARM_DELTA_SCALE)
        actions += move
        move, current = _move_to(current, low, -1.0, delta_scale=SWARM_ARM_DELTA_SCALE)
        actions += move
        actions += _repeat([0.0, 0.0, 0.0, 0.0, 1.0], 3)  # open gripper
        move, current = _move_to(current, high, 1.0, delta_scale=SWARM_ARM_DELTA_SCALE)
        actions += move
        return actions


def run(cubes_per_rover: int, use_viewer: bool, speed: float, verbose: bool,
        camera_idx: int | None) -> None:
    bridge = MarsSwarmBridge()
    bridge.reset(seed=0)

    # Cap each rover's dome queue to ``cubes_per_rover`` so short demos finish quickly.
    if cubes_per_rover > 0:
        for r in bridge.rovers:
            while len(r.dome_queue) > cubes_per_rover:
                r.dome_queue.pop()

    agents = [RoverAgent(i, bridge) for i in range(NUM_ROVERS)]
    total_target = sum(len(r.dome_queue) for r in bridge.rovers)
    print(f"Swarm demo: 3 rovers, {total_target} dome targets total "
          f"(per-rover: {[len(r.dome_queue) for r in bridge.rovers]}).")

    viewer = None
    if use_viewer:
        viewer = mujoco.viewer.launch_passive(bridge.model, bridge.data)
        if camera_idx is not None and 0 <= camera_idx < 3:
            # Lock the viewer to one rover's front-down POV camera.
            cam_name = f"r{camera_idx}_front_cam"
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = bridge.model.camera(cam_name).id
        else:
            viewer.cam.distance = 12.0
            viewer.cam.elevation = -55
            viewer.cam.azimuth = 90
            viewer.cam.lookat[:] = [0.0, 0.0, 3.5]

    # Pacing tick: sync viewer + sleep so playback matches ``speed`` x real-time.
    SIM_DT_PER_STEP = 0.040
    wall_dt = SIM_DT_PER_STEP / max(speed, 1e-6)
    last_t = [time.time()]

    def viewer_tick() -> None:
        if viewer is None or not viewer.is_running():
            return
        viewer.sync()
        now = time.time()
        sleep_for = wall_dt - (now - last_t[0])
        if sleep_for > 0:
            time.sleep(sleep_for)
        last_t[0] = time.time()

    MAX_TICKS = 60000
    last_print_step = 0
    for tick in range(MAX_TICKS):
        actions = [agent.tick() for agent in agents]
        bridge.step(actions)
        viewer_tick()

        if tick - last_print_step >= 100:
            last_print_step = tick
            placed = bridge.placed_count()
            states = [a.state.name for a in agents]
            holding = [int(r.holding) for r in bridge.rovers]
            if verbose:
                print(f"t={tick:5d} placed={placed}/{total_target} states={states} holding={holding}")

        if all(a.state == State.DONE for a in agents):
            print(f"All rovers DONE at tick={tick}.")
            break

    placed = bridge.placed_count()
    print(f"\nFinal: placed={placed}/{total_target}")
    if viewer is not None:
        viewer.close()
    bridge.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Three-rover swarm dome-build demo.")
    parser.add_argument("--cubes-per-rover", type=int, default=4,
                        help="Limit each rover's dome queue to this many cubes (default 4).")
    parser.add_argument("--viewer", action="store_true",
                        help="Launch the MuJoCo passive viewer (use with mjpython).")
    parser.add_argument(
        "--speed",
        type=float,
        default=6.0,
        help="Viewer playback multiplier (default: 6x real time).",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--camera", type=int, default=None, choices=[0, 1, 2],
                        help="Lock the viewer to rover N's front-down POV camera.")
    args = parser.parse_args()
    run(args.cubes_per_rover, args.viewer, args.speed, args.verbose, args.camera)


if __name__ == "__main__":
    main()
