"""Headless, deterministic construction runners used by the Atomz live feed.

This module intentionally has no viewer or terminal dependencies.  A caller owns
the pacing; each ``step`` advances one control action and ``render`` returns an
RGB frame from a fixed scene camera.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Literal

# Must be set before MuJoCo creates an OpenGL context in a Modal worker.
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

HabitatType = Literal["regolith_dome", "ellipsoid_habitat"]
HABITAT_TYPES = frozenset(("regolith_dome", "ellipsoid_habitat"))


@dataclass
class RunnerStatus:
    progress: int
    complete: bool


class BaseRunner:
    width = 1280
    height = 720

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.model, self.data = model, data
        # The authored scenes use MuJoCo's default 640px offscreen buffer.
        # Raise it before Renderer allocation so the live 1280×720 feed fits.
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, self.width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, self.height)
        self.renderer = mujoco.Renderer(model, height=self.height, width=self.width)
        self.complete = False

    def render(self) -> np.ndarray:
        self.renderer.update_scene(self.data, camera=self.camera)
        self._add_overlay_geometry()
        return self.renderer.render()

    def _add_overlay_geometry(self) -> None:
        """Subclasses can add persistent construction material to the scene."""

    def close(self) -> None:
        self.renderer.close()


class RegolithDomeRunner(BaseRunner):
    """Three-rover regolith pick-and-place dome construction."""

    def __init__(self) -> None:
        from run_swarm_demo import NUM_ROVERS, RoverAgent
        from swarm_bridge import MarsSwarmBridge

        self.bridge = MarsSwarmBridge()
        self.bridge.reset(seed=0)
        # Keep the live session bounded while retaining the full choreography.
        for rover in self.bridge.rovers:
            while len(rover.dome_queue) > 4:
                rover.dome_queue.pop()
        self.agents = [RoverAgent(i, self.bridge) for i in range(NUM_ROVERS)]
        self.total_targets = sum(len(rover.dome_queue) for rover in self.bridge.rovers)
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.lookat[:] = [0.0, 0.0, 3.5]
        self.camera.distance = 12.0
        self.camera.elevation = -55.0
        self.camera.azimuth = 90.0
        super().__init__(self.bridge.model, self.bridge.data)

    def step(self) -> RunnerStatus:
        if not self.complete:
            self.bridge.step([agent.tick() for agent in self.agents])
            self.complete = all(agent.state.name == "DONE" for agent in self.agents)
        progress = round(100 * self.bridge.placed_count() / max(self.total_targets, 1))
        return RunnerStatus(progress=progress, complete=self.complete)

    def close(self) -> None:
        super().close()
        self.bridge.close()


class EllipsoidHabitatRunner(BaseRunner):
    """Headless version of ``run_shared_printer_demo.py``'s five-printer build.

    It deliberately reuses the shared-scene helper functions and constants so
    the browser feed has the same terrain, mobile printer choreography, and
    ellipsoidal print geometry as the desktop MuJoCo demo.
    """

    def __init__(self) -> None:
        from run_shared_printer_demo import (
            BEAD_RADIUS,
            BEAD_RGBA,
            IDENTITY,
            PRINTER_APPROACH_STEPS,
            PRINTER_DEPARTURE_STEPS,
            PRINTER_ELLIPSE_RADII,
            PRINTER_LAYER_TRANSITION_TICKS,
            PRINTER_LAYERS,
            PRINTER_MOTION_TICKS,
            PRINTER_PARKED_JOINTS,
            _layer_scale,
            _lerp_angle,
            _lookup_printers,
            _printer_target,
            _sector_points,
            _set_printer_base_pose,
            _set_printer_tip,
            _smoothstep,
        )
        from swarm_bridge import MarsSwarmBridge

        self.bridge = MarsSwarmBridge()
        self.bridge.reset(seed=0)
        self.printers = _lookup_printers(self.bridge.model, self.bridge.data)
        self.bead_radius, self.bead_rgba, self.identity = BEAD_RADIUS, BEAD_RGBA, IDENTITY
        self._layer_scale, self._lerp_angle = _layer_scale, _lerp_angle
        self._printer_target, self._sector_points = _printer_target, _sector_points
        self._set_base, self._set_tip, self._smoothstep = _set_printer_base_pose, _set_printer_tip, _smoothstep
        self.approach_steps, self.departure_steps = PRINTER_APPROACH_STEPS, PRINTER_DEPARTURE_STEPS
        self.layers, self.motion_ticks, self.transition_ticks = PRINTER_LAYERS, PRINTER_MOTION_TICKS, PRINTER_LAYER_TRANSITION_TICKS
        self.radii = PRINTER_ELLIPSE_RADII
        self.parked_joints = PRINTER_PARKED_JOINTS
        self.phase = "approach"
        self.phase_step = self.layer_idx = self.point_idx = self.motion_idx = 0
        self.beads: list[np.ndarray] = []
        for rig in self.printers:
            self._set_base(self.bridge.data, rig, rig.start_pos, rig.travel_yaw)
            self.bridge.data.qpos[rig.qposadrs] = self.parked_joints
        mujoco.mj_forward(self.bridge.model, self.bridge.data)
        self.previous_targets = [np.array(self.bridge.data.site_xpos[rig.tip_site_id], dtype=np.float64) for rig in self.printers]
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.lookat[:] = [2.4, -1.9, 3.8]
        self.camera.distance = 13.8
        self.camera.elevation = -26.0
        self.camera.azimuth = 110.0
        super().__init__(self.bridge.model, self.bridge.data)

    def step(self) -> RunnerStatus:
        if not self.complete:
            self.bridge.step([np.zeros(5, dtype=np.float32) for _ in self.bridge.rovers])
            data, model = self.bridge.data, self.bridge.model
            if self.phase == "approach":
                self.phase_step += 1
                alpha = self._smoothstep(self.phase_step / self.approach_steps)
                for rig in self.printers:
                    pos = (1.0 - alpha) * rig.start_pos + alpha * rig.final_pos
                    yaw = self._lerp_angle(rig.travel_yaw, rig.final_yaw, self._smoothstep(max(0.0, 3.0 * alpha - 2.0)))
                    self._set_base(data, rig, pos, yaw)
                    data.qpos[rig.qposadrs] = self.parked_joints
                if self.phase_step >= self.approach_steps:
                    self.phase = "printing"
            elif self.phase == "printing":
                scale = self._layer_scale(self.layer_idx, self.layers)
                count = self._sector_points(self.radii[0] * scale, self.radii[1] * scale)
                targets = [self._printer_target(rig, self.layer_idx, self.point_idx, count) for rig in self.printers]
                ticks = self.transition_ticks if self.point_idx == 0 and self.layer_idx else self.motion_ticks
                for rig in self.printers:
                    self._set_base(data, rig, rig.final_pos, rig.final_yaw)
                mujoco.mj_forward(model, data)
                alpha = (self.motion_idx + 1) / ticks
                for rig, previous, target in zip(self.printers, self.previous_targets, targets):
                    self._set_tip(model, data, rig, (1.0 - alpha) * previous + alpha * target)
                if self.motion_idx + 1 >= ticks:
                    self.previous_targets, self.motion_idx = targets, 0
                    self.point_idx += 1
                    if self.point_idx >= count:
                        self.layer_idx, self.point_idx = self.layer_idx + 1, 0
                        if self.layer_idx >= self.layers:
                            self.phase, self.phase_step = "departure", 0
                else:
                    self.motion_idx += 1
                mujoco.mj_forward(model, data)
                self.beads.extend(np.array(data.site_xpos[rig.tip_site_id], dtype=np.float64) for rig in self.printers)
            elif self.phase == "departure":
                self.phase_step += 1
                alpha = self._smoothstep(self.phase_step / self.departure_steps)
                for rig in self.printers:
                    pos = (1.0 - alpha) * rig.final_pos + alpha * rig.departure_pos
                    self._set_base(data, rig, pos, self._lerp_angle(rig.final_yaw, rig.departure_yaw, alpha))
                    data.qpos[rig.qposadrs] = self.parked_joints
                if self.phase_step >= self.departure_steps:
                    self.phase, self.complete = "complete", True
            mujoco.mj_forward(model, data)
        if self.phase == "approach":
            progress = round(10 * self.phase_step / self.approach_steps)
        elif self.phase == "printing":
            progress = 10 + round(80 * self.layer_idx / self.layers)
        elif self.phase == "departure":
            progress = 90 + round(10 * self.phase_step / self.departure_steps)
        else:
            progress = 100
        return RunnerStatus(progress=progress, complete=self.complete)

    def _add_overlay_geometry(self) -> None:
        scene = self.renderer.scene
        available = scene.maxgeom - scene.ngeom
        if available <= 0 or not self.beads:
            return
        indices = np.linspace(0, len(self.beads) - 1, min(available, len(self.beads))).round().astype(np.int32)
        start = scene.ngeom
        for offset, bead_index in enumerate(indices):
            mujoco.mjv_initGeom(
                scene.geoms[start + offset],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([self.bead_radius, 0.0, 0.0]),
                self.beads[bead_index],
                self.identity,
                self.bead_rgba,
            )
        scene.ngeom += len(indices)

    def close(self) -> None:
        super().close()
        self.bridge.close()


def create_runner(habitat_type: HabitatType) -> BaseRunner:
    if habitat_type == "regolith_dome":
        return RegolithDomeRunner()
    if habitat_type == "ellipsoid_habitat":
        return EllipsoidHabitatRunner()
    raise ValueError(f"Unsupported habitat type: {habitat_type}")
