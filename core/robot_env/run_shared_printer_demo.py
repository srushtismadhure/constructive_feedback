"""Shared construction demo: swarm rovers plus five mobile printers.

Run with:
    mjpython robot_env/run_shared_printer_demo.py
"""
from __future__ import annotations

import argparse
import math
import select
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_swarm_demo import RoverAgent, State
from swarm_bridge import MarsSwarmBridge, NUM_ROVERS, PRINTER_DOME_CENTER


PRINTER_COUNT = 5
PRINTER_LAYERS = 20
PRINTER_DOME_HEIGHT = 1.28
PRINTER_ELLIPSE_RADII = (1.95, 1.35)
PRINTER_TOP_SCALE = 0.14
PRINTER_APPROACH_STEPS = 600
PRINTER_MOTION_TICKS = 3
PRINTER_LAYER_TRANSITION_TICKS = 12
PRINTER_DEPARTURE_STEPS = 600
# Shoulder and elbow fold the nozzle back over the chassis while parked.
PRINTER_PARKED_JOINTS = np.array([0.0, 0.0, 2.80], dtype=np.float64)
PRINTER_START_POSITIONS = np.array(
    [
        [7.80, -2.40, 3.49],
        [6.90, 0.70, 3.49],
        [3.00, 0.80, 3.49],
        [2.60, -7.10, 3.49],
        [6.70, -7.00, 3.49],
    ],
    dtype=np.float64,
)
PRINTER_DEPARTURE_POSITIONS = np.array(
    [
        [7.85, -0.60, 3.49],
        [7.45, 1.75, 3.49],
        [1.10, 1.45, 3.49],
        [1.10, -7.75, 3.49],
        [7.70, -7.70, 3.49],
    ],
    dtype=np.float64,
)
# Deliberate overlap produces a continuous print skin instead of a dotted path.
BEAD_RADIUS = 0.070
BEAD_SPACING = BEAD_RADIUS * 0.45
BEAD_RGBA = np.array([0.70, 0.26, 0.16, 1.0])
IDENTITY = np.eye(3).ravel()


@dataclass
class PrinterRig:
    body_id: int
    free_qposadr: int
    free_dofadr: int
    qposadrs: np.ndarray
    dofadrs: np.ndarray
    low: np.ndarray
    high: np.ndarray
    tip_site_id: int
    sector_start: float
    sector_end: float
    start_pos: np.ndarray
    final_pos: np.ndarray
    departure_pos: np.ndarray
    final_yaw: float
    travel_yaw: float
    departure_yaw: float


def _ellipse_circumference(rx: float, ry: float) -> float:
    h = ((rx - ry) ** 2) / max((rx + ry) ** 2, 1e-9)
    return math.pi * (rx + ry) * (1.0 + (3.0 * h) / (10.0 + math.sqrt(max(4.0 - 3.0 * h, 1e-9))))


def _layer_scale(layer_idx: int, layers: int) -> float:
    u = layer_idx / max(layers - 1, 1)
    return PRINTER_TOP_SCALE + (1.0 - PRINTER_TOP_SCALE) * math.sqrt(max(0.0, 1.0 - u * u))


def _sector_points(rx: float, ry: float) -> int:
    sector_arc = _ellipse_circumference(rx, ry) / PRINTER_COUNT
    return max(14, int(math.ceil(sector_arc / BEAD_SPACING)))


def _smoothstep(u: float) -> float:
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * (3.0 - 2.0 * u)


def _yaw_from_quat(quat: np.ndarray) -> float:
    return math.atan2(2.0 * (quat[0] * quat[3] + quat[1] * quat[2]), 1.0 - 2.0 * (quat[2] ** 2 + quat[3] ** 2))


def _quat_from_yaw(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw * 0.5), 0.0, 0.0, math.sin(yaw * 0.5)], dtype=np.float64)


def _lerp_angle(start: float, end: float, alpha: float) -> float:
    return start + math.atan2(math.sin(end - start), math.cos(end - start)) * alpha


def _lookup_printers(model: mujoco.MjModel, data: mujoco.MjData) -> list[PrinterRig]:
    rigs: list[PrinterRig] = []
    for idx in range(PRINTER_COUNT):
        free_joint_id = model.joint(f"printer_{idx}_free").id
        joint_names = [
            f"printer_{idx}_arm_yaw_joint",
            f"printer_{idx}_arm_shoulder_joint",
            f"printer_{idx}_arm_elbow_joint",
        ]
        joint_ids = np.array([model.joint(name).id for name in joint_names], dtype=np.int32)
        # The folded working posture needs the full mechanical travel of the two-link arm.
        model.jnt_range[joint_ids[1], :] = np.deg2rad([-120.0, 120.0])
        model.jnt_range[joint_ids[2], :] = np.deg2rad([-170.0, 170.0])
        free_qposadr = int(model.jnt_qposadr[free_joint_id])
        free_dofadr = int(model.jnt_dofadr[free_joint_id])
        body_id = model.body(f"printer_{idx}_base").id
        final_pos = np.array(model.body_pos[body_id], dtype=np.float64)
        final_yaw = _yaw_from_quat(np.array(model.body_quat[body_id], dtype=np.float64))
        travel_xy = final_pos[:2] - PRINTER_START_POSITIONS[idx, :2]
        departure_xy = PRINTER_DEPARTURE_POSITIONS[idx, :2] - final_pos[:2]
        rigs.append(
            PrinterRig(
                body_id=body_id,
                free_qposadr=free_qposadr,
                free_dofadr=free_dofadr,
                qposadrs=np.array([model.jnt_qposadr[jid] for jid in joint_ids], dtype=np.int32),
                dofadrs=np.array([model.jnt_dofadr[jid] for jid in joint_ids], dtype=np.int32),
                low=model.jnt_range[joint_ids, 0].astype(np.float64),
                high=model.jnt_range[joint_ids, 1].astype(np.float64),
                tip_site_id=model.site(f"printer_{idx}_tip").id,
                sector_start=(2.0 * math.pi * idx) / PRINTER_COUNT,
                sector_end=(2.0 * math.pi * (idx + 1)) / PRINTER_COUNT,
                start_pos=PRINTER_START_POSITIONS[idx].copy(),
                final_pos=final_pos,
                departure_pos=PRINTER_DEPARTURE_POSITIONS[idx].copy(),
                final_yaw=final_yaw,
                travel_yaw=math.atan2(travel_xy[1], travel_xy[0]),
                departure_yaw=math.atan2(departure_xy[1], departure_xy[0]),
            )
        )
    return rigs


def _printer_target(rig: PrinterRig, layer_idx: int, point_idx: int, point_count: int) -> np.ndarray:
    layer_z = PRINTER_DOME_CENTER[2] + (layer_idx * PRINTER_DOME_HEIGHT / max(PRINTER_LAYERS - 1, 1))
    scale = _layer_scale(layer_idx, PRINTER_LAYERS)
    rx = PRINTER_ELLIPSE_RADII[0] * scale
    ry = PRINTER_ELLIPSE_RADII[1] * scale
    u = (point_idx + 0.5) / point_count
    # Serpentine layer ordering: finish one layer at an edge, rise, then
    # immediately continue the next layer from that same edge.
    if layer_idx % 2:
        u = 1.0 - u
    angle = rig.sector_start + (rig.sector_end - rig.sector_start) * u
    return np.array(
        [
            PRINTER_DOME_CENTER[0] + rx * math.cos(angle),
            PRINTER_DOME_CENTER[1] + ry * math.sin(angle),
            layer_z,
        ],
        dtype=np.float64,
    )


def _set_printer_tip(model: mujoco.MjModel, data: mujoco.MjData, rig: PrinterRig, target_xyz: np.ndarray) -> None:
    """Solve the attached yaw/shoulder/elbow arm, keeping the nozzle on the robot."""
    jacp = np.empty((3, model.nv), dtype=np.float64)
    jacr = np.empty((3, model.nv), dtype=np.float64)
    for _ in range(24):
        mujoco.mj_forward(model, data)
        error = target_xyz - data.site_xpos[rig.tip_site_id]
        if np.linalg.norm(error) < 1e-4:
            break
        mujoco.mj_jacSite(model, data, jacp, jacr, rig.tip_site_id)
        arm_jacobian = jacp[:, rig.dofadrs]
        delta = arm_jacobian.T @ np.linalg.solve(
            arm_jacobian @ arm_jacobian.T + 0.003 * np.eye(3), error
        )
        delta = np.clip(delta, -0.14, 0.14)
        data.qpos[rig.qposadrs] = np.clip(data.qpos[rig.qposadrs] + delta, rig.low, rig.high)
    data.qvel[rig.dofadrs] = 0.0


def _set_printer_base_pose(data: mujoco.MjData, rig: PrinterRig, pos_xyz: np.ndarray, yaw: float) -> None:
    qa = rig.free_qposadr
    va = rig.free_dofadr
    data.qpos[qa:qa + 3] = pos_xyz
    data.qpos[qa + 3:qa + 7] = _quat_from_yaw(yaw)
    data.qvel[va:va + 6] = 0.0


def _render_beads(viewer: mujoco.viewer.Handle, beads: list[np.ndarray]) -> None:
    if viewer.user_scn is None or not beads:
        return
    scn = viewer.user_scn
    if len(beads) <= scn.maxgeom:
        render_beads = beads
    else:
        sample_idx = np.linspace(0, len(beads) - 1, scn.maxgeom).round().astype(np.int32)
        render_beads = [beads[i] for i in sample_idx]
    scn.ngeom = len(render_beads)
    for idx, bead in enumerate(render_beads):
        mujoco.mjv_initGeom(
            scn.geoms[idx],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([BEAD_RADIUS, 0.0, 0.0]),
            bead,
            IDENTITY,
            BEAD_RGBA,
        )


def _trim_swarm_queues(bridge: MarsSwarmBridge, cubes_per_rover: int) -> None:
    if cubes_per_rover <= 0:
        return
    for rover in bridge.rovers:
        while len(rover.dome_queue) > cubes_per_rover:
            rover.dome_queue.pop()


def _run_live_commands(
    viewer: mujoco.viewer.Handle,
    bridge: MarsSwarmBridge,
    agents: list[RoverAgent],
    printers: list[PrinterRig],
    beads: list[np.ndarray],
    wall_dt: float,
) -> None:
    """Advance swarm and printer state machines independently from live stdin commands."""
    model, data = bridge.model, bridge.data
    assert model is not None and data is not None
    swarm_active = False
    printer_phase = "idle"
    phase_step = 0
    layer_idx = point_idx = motion_idx = 0
    previous_targets: list[np.ndarray] = []
    last_sync = time.time()
    print("Simulation idle. Enter 1 for swarm, 2 for printers, or q to quit. Commands may be entered while either runs.")

    while viewer.is_running():
        ready, _, _ = select.select([sys.stdin], [], [], 0.0)
        for _ in ready:
            command = sys.stdin.readline().strip()
            if command == "1" and not all(agent.state == State.DONE for agent in agents):
                swarm_active = True
                print("Swarm started.")
            elif command == "2" and printer_phase == "idle":
                printer_phase, phase_step = "approach", 0
                print("Printers started.")
            elif command == "q":
                return

        actions = [agent.tick() for agent in agents] if swarm_active else [np.zeros(5, dtype=np.float32) for _ in agents]
        bridge.step(actions)
        if swarm_active and all(agent.state == State.DONE for agent in agents):
            swarm_active = False
            print("Swarm complete.")

        if printer_phase == "idle":
            for rig in printers:
                _set_printer_base_pose(data, rig, rig.start_pos, rig.travel_yaw)
                data.qpos[rig.qposadrs] = PRINTER_PARKED_JOINTS
        elif printer_phase == "approach":
            phase_step += 1
            alpha = _smoothstep(phase_step / PRINTER_APPROACH_STEPS)
            for rig in printers:
                pos = (1.0 - alpha) * rig.start_pos + alpha * rig.final_pos
                yaw = _lerp_angle(rig.travel_yaw, rig.final_yaw, _smoothstep(max(0.0, 3.0 * alpha - 2.0)))
                _set_printer_base_pose(data, rig, pos, yaw)
                data.qpos[rig.qposadrs] = PRINTER_PARKED_JOINTS
            if phase_step >= PRINTER_APPROACH_STEPS:
                printer_phase, layer_idx, point_idx, motion_idx = "printing", 0, 0, 0
                mujoco.mj_forward(model, data)
                previous_targets = [np.array(data.site_xpos[rig.tip_site_id], dtype=np.float64) for rig in printers]
        elif printer_phase == "printing":
            scale = _layer_scale(layer_idx, PRINTER_LAYERS)
            count = _sector_points(PRINTER_ELLIPSE_RADII[0] * scale, PRINTER_ELLIPSE_RADII[1] * scale)
            target_points = [_printer_target(rig, layer_idx, point_idx, count) for rig in printers]
            ticks = PRINTER_LAYER_TRANSITION_TICKS if point_idx == 0 and layer_idx else PRINTER_MOTION_TICKS
            for rig in printers:
                _set_printer_base_pose(data, rig, rig.final_pos, rig.final_yaw)
            mujoco.mj_forward(model, data)
            alpha = (motion_idx + 1) / ticks
            for rig, previous, target in zip(printers, previous_targets, target_points):
                _set_printer_tip(model, data, rig, (1.0 - alpha) * previous + alpha * target)
            if motion_idx + 1 >= ticks:
                previous_targets, motion_idx = target_points, 0
                point_idx += 1
                if point_idx >= count:
                    layer_idx, point_idx = layer_idx + 1, 0
                    if layer_idx >= PRINTER_LAYERS:
                        printer_phase, phase_step = "departure", 0
            else:
                motion_idx += 1
        elif printer_phase == "departure":
            phase_step += 1
            alpha = _smoothstep(phase_step / PRINTER_DEPARTURE_STEPS)
            for rig in printers:
                pos = (1.0 - alpha) * rig.final_pos + alpha * rig.departure_pos
                yaw = _lerp_angle(rig.final_yaw, rig.departure_yaw, alpha)
                _set_printer_base_pose(data, rig, pos, yaw)
                data.qpos[rig.qposadrs] = PRINTER_PARKED_JOINTS
            if phase_step >= PRINTER_DEPARTURE_STEPS:
                printer_phase = "complete"

        mujoco.mj_forward(model, data)
        if printer_phase == "printing":
            for rig in printers:
                beads.append(np.array(data.site_xpos[rig.tip_site_id], dtype=np.float64))
        _render_beads(viewer, beads)
        viewer.sync()
        now = time.time()
        if (sleep_for := wall_dt - (now - last_sync)) > 0.0:
            time.sleep(sleep_for)
        last_sync = time.time()


def main() -> None:
    parser = argparse.ArgumentParser(description="Swarm + five-printer shared construction demo.")
    parser.add_argument("--cubes-per-rover", type=int, default=12)
    parser.add_argument("--speed", type=float, default=2.8)
    args = parser.parse_args()

    bridge = MarsSwarmBridge()
    bridge.reset(seed=0)
    _trim_swarm_queues(bridge, args.cubes_per_rover)

    model, data = bridge.model, bridge.data
    assert model is not None and data is not None
    agents = [RoverAgent(i, bridge) for i in range(NUM_ROVERS)]
    printers = _lookup_printers(model, data)
    beads: list[np.ndarray] = []

    for rig in printers:
        _set_printer_base_pose(data, rig, rig.start_pos, rig.travel_yaw)
        data.qpos[rig.qposadrs] = PRINTER_PARKED_JOINTS
    mujoco.mj_forward(model, data)

    sim_dt = 0.040
    wall_dt = sim_dt / max(args.speed, 1e-6)

    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        viewer.cam.lookat[:] = [2.4, -1.9, 3.8]
        viewer.cam.distance = 13.8
        viewer.cam.elevation = -26
        viewer.cam.azimuth = 110

        _run_live_commands(viewer, bridge, agents, printers, beads, wall_dt)
        bridge.close()
        return

        viewer.sync()
        print("Simulation idle. Select: 1 = swarm dome, 2 = printed dome, 3 = swarm then printed dome.")
        mode = ""
        while mode not in {"1", "2", "3"}:
            mode = input("> ").strip()
            if mode not in {"1", "2", "3"}:
                print("Enter 1, 2, or 3.")
        swarm_enabled = mode in {"1", "3"}

        def swarm_actions() -> list[np.ndarray]:
            if swarm_enabled:
                return [agent.tick() for agent in agents]
            return [np.zeros(5, dtype=np.float32) for _ in agents]

        last_sync = time.time()
        printing_done = False
        # Modes 1 and 3 complete the swarm first; printer bases stay parked.
        if swarm_enabled:
            while not all(agent.state == State.DONE for agent in agents):
                if not viewer.is_running():
                    bridge.close()
                    return
                bridge.step(swarm_actions())
                for rig in printers:
                    _set_printer_base_pose(data, rig, rig.start_pos, rig.travel_yaw)
                    data.qpos[rig.qposadrs] = PRINTER_PARKED_JOINTS
                    data.qvel[rig.dofadrs] = 0.0
                mujoco.mj_forward(model, data)
                viewer.sync()
                now = time.time()
                sleep_for = wall_dt - (now - last_sync)
                if sleep_for > 0.0:
                    time.sleep(sleep_for)
                last_sync = time.time()
            if mode == "1":
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(wall_dt)
                bridge.close()
                return

        for step_idx in range(PRINTER_APPROACH_STEPS):
            if not viewer.is_running():
                bridge.close()
                return
            bridge.step(swarm_actions())
            alpha = _smoothstep((step_idx + 1) / PRINTER_APPROACH_STEPS)
            for rig in printers:
                base_pos = (1.0 - alpha) * rig.start_pos + alpha * rig.final_pos
                yaw = _lerp_angle(rig.travel_yaw, rig.final_yaw, _smoothstep(max(0.0, 3.0 * alpha - 2.0)))
                _set_printer_base_pose(data, rig, base_pos, yaw)
                data.qpos[rig.qposadrs] = PRINTER_PARKED_JOINTS
                data.qvel[rig.dofadrs] = 0.0
            mujoco.mj_forward(model, data)
            viewer.sync()
            now = time.time()
            sleep_for = wall_dt - (now - last_sync)
            if sleep_for > 0.0:
                time.sleep(sleep_for)
            last_sync = time.time()

        previous_targets = [np.array(data.site_xpos[rig.tip_site_id], dtype=np.float64) for rig in printers]
        for layer_idx in range(PRINTER_LAYERS):
            scale = _layer_scale(layer_idx, PRINTER_LAYERS)
            rx = PRINTER_ELLIPSE_RADII[0] * scale
            ry = PRINTER_ELLIPSE_RADII[1] * scale
            point_count = _sector_points(rx, ry)
            for point_idx in range(point_count):
                target_points = [_printer_target(rig, layer_idx, point_idx, point_count) for rig in printers]
                # Several normal swarm ticks per bead waypoint: printer motion is slow,
                # while swarm actions retain their original tick rate.
                motion_ticks = PRINTER_LAYER_TRANSITION_TICKS if point_idx == 0 and layer_idx else PRINTER_MOTION_TICKS
                for motion_tick in range(motion_ticks):
                    if not viewer.is_running():
                        bridge.close()
                        return
                    bridge.step(swarm_actions())
                    for rig in printers:
                        _set_printer_base_pose(data, rig, rig.final_pos, rig.final_yaw)
                    mujoco.mj_forward(model, data)
                    alpha = (motion_tick + 1) / motion_ticks
                    for rig, previous, target in zip(printers, previous_targets, target_points):
                        _set_printer_tip(model, data, rig, (1.0 - alpha) * previous + alpha * target)
                    mujoco.mj_forward(model, data)
                    for rig in printers:
                        beads.append(np.array(data.site_xpos[rig.tip_site_id], dtype=np.float64))
                    _render_beads(viewer, beads)
                    viewer.sync()
                    now = time.time()
                    sleep_for = wall_dt - (now - last_sync)
                    if sleep_for > 0.0:
                        time.sleep(sleep_for)
                    last_sync = time.time()
                previous_targets = target_points
        printing_done = True

        # Retract, then drive each printer away from the shared build area.
        for step_idx in range(PRINTER_DEPARTURE_STEPS):
            if not viewer.is_running():
                bridge.close()
                return
            bridge.step(swarm_actions())
            alpha = _smoothstep((step_idx + 1) / PRINTER_DEPARTURE_STEPS)
            for rig in printers:
                base_pos = (1.0 - alpha) * rig.final_pos + alpha * rig.departure_pos
                yaw = _lerp_angle(rig.final_yaw, rig.departure_yaw, _smoothstep(min(1.0, 3.0 * alpha)))
                _set_printer_base_pose(data, rig, base_pos, yaw)
                data.qpos[rig.qposadrs] = PRINTER_PARKED_JOINTS
                data.qvel[rig.dofadrs] = 0.0
            mujoco.mj_forward(model, data)
            _render_beads(viewer, beads)
            viewer.sync()
            now = time.time()
            sleep_for = wall_dt - (now - last_sync)
            if sleep_for > 0.0:
                time.sleep(sleep_for)
            last_sync = time.time()

        while viewer.is_running():
            if swarm_enabled and not all(agent.state == State.DONE for agent in agents):
                bridge.step(swarm_actions())
                for rig in printers:
                    _set_printer_base_pose(data, rig, rig.departure_pos, rig.departure_yaw)
                mujoco.mj_forward(model, data)
            _render_beads(viewer, beads)
            viewer.sync()
            now = time.time()
            sleep_for = wall_dt - (now - last_sync)
            if sleep_for > 0.0:
                time.sleep(sleep_for)
            last_sync = time.time()
            if printing_done and all(agent.state == State.DONE for agent in agents):
                time.sleep(0.4)
                break

    bridge.close()


if __name__ == "__main__":
    main()
