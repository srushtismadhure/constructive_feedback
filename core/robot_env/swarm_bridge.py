"""Multi-rover bridge for the Mars swarm dome-build demo.

Owns a single MjModel/MjData (loaded from ``mars_scene_swarm.xml``) and 3
``RoverUnit`` instances, each scoped to one rover via a name prefix
(``r0_``/``r1_``/``r2_``). Pile cubes are a shared pool; dome targets are
partitioned by sector and assigned per-rover. The bridge does NOT decide
when each rover acts — that's the orchestrator's job. The bridge just lets
each rover write its own arm/gripper/wheel ctrl into shared ``data`` and
advances physics once per tick.

Reuses ``_ik_top_down``, ``_pile_positions``, ``_dome_positions``,
``_move_to``, ``_repeat`` from ``hud_arm_bridge`` since they are pure
free functions.
"""
from __future__ import annotations

import math
import os
from collections import deque
from dataclasses import dataclass, field

import mujoco
import numpy as np

from hud_arm_bridge import (
    DELTA_SCALE,
    GRIPPER_CLOSED_LEFT,
    GRIPPER_CLOSED_RIGHT,
    HELD_OFFSET,
    NUM_CUBES,
    PHYSICS_STEPS_PER_ACTION,
    PICK_DISTANCE,
    TARGET_RADIUS,
    _dome_positions,
)


def _swarm_pile_positions(center_x: float, center_y: float) -> list[np.ndarray]:
    """12-cube positions in a 2x2x3 stack centered at (center_x, center_y).
    Top layer first (so picking peels from the top)."""
    positions = []
    cols, rows, layers = 2, 2, 3
    spacing_xy = 0.12
    spacing_z = 0.115
    base_z = 3.39
    for i in range(CUBES_PER_PILE):
        layer = (layers - 1) - (i // (cols * rows))
        in_layer = i % (cols * rows)
        col = in_layer % cols
        row = in_layer // cols
        x = center_x + (col - (cols - 1) / 2.0) * spacing_xy
        y = center_y + (row - (rows - 1) / 2.0) * spacing_xy
        z = base_z + layer * spacing_z
        positions.append(np.array([x, y, z], dtype=np.float64))
    return positions

SCENE_DIR = os.path.dirname(os.path.abspath(__file__))

DOME_CENTER = (0.0, 0.0, 3.39)
NUM_ROVERS = 3
PILE_RADIUS = 3.5
# The outer dome tier has a 0.50 m radius. Keep rover chassis outside that
# ring while retaining enough arm reach for the innermost tier.
DOME_STAGING_RADIUS = 1.00
# Keep rover centres outside the pile by more than the chassis half-length
# (0.36 m) plus a margin. The arm reaches into the remaining gap.
PILE_SAFETY_CLEARANCE = 0.70
PILE_STAGING_RADIUS = PILE_RADIUS - PILE_SAFETY_CLEARANCE
CUBES_PER_PILE = 12  # each rover owns a 2x2x3 = 12 cube pile
# Swarm choreography uses the same IK poses as the single rover but can safely
# take larger position increments because the bridge pins scripted cubes.
SWARM_ARM_DELTA_SCALE = DELTA_SCALE * 1.8
SWARM_GRIPPER_STEP = 0.35

# Per-rover pile centers (must match gen_swarm_xml.py): radius 2.0 at 270/150/30 deg.
def _pile_center_for(rover_idx: int) -> tuple[float, float]:
    angles_deg = [270.0, 150.0, 30.0]
    a = math.radians(angles_deg[rover_idx])
    return PILE_RADIUS * math.cos(a), PILE_RADIUS * math.sin(a)

# Rover prefixes in same order as XML/spawn angles (270, 150, 30 deg).
ROVER_PREFIXES = ["r0_", "r1_", "r2_"]
# Sector angle ranges (radians, atan2 output convention). Each rover owns dome
# slots whose azimuth lies in its 120-degree wedge:
#   rover 0 (south)    : [-2pi/3, -pi/3]   (270 deg ± 60)
#   rover 1 (NW)       : [+pi/3, +pi]      (150 deg ± 60)
#   rover 2 (NE)       : [-pi/3, +pi/3]    (30 deg ± 60)
SECTOR_RANGES = [
    (-2.0 * math.pi / 3.0, -math.pi / 3.0),
    (math.pi / 3.0, math.pi),
    (-math.pi / 3.0, math.pi / 3.0),
]


SECTOR_CENTERS = [-math.pi / 2.0, 5.0 * math.pi / 6.0, math.pi / 6.0]  # 270, 150, 30 deg


def _azimuth_to_rover(angle: float) -> int:
    """Map a dome-slot azimuth (radians) to the rover whose 120-degree wedge centre
    is closest (accounting for the +/- pi wrap)."""
    diffs = []
    for c in SECTOR_CENTERS:
        d = ((angle - c + math.pi) % (2.0 * math.pi)) - math.pi
        diffs.append(abs(d))
    return int(min(range(len(diffs)), key=lambda i: diffs[i]))


@dataclass
class RoverUnit:
    """Per-rover slice of the bridge: own joint addrs, actuator ids, IK targets,
    grasp state, dome queue, etc. Wraps the ``hud_arm_bridge`` semantics for one
    rover via a name prefix.
    """
    model: mujoco.MjModel
    data: mujoco.MjData
    prefix: str
    spawn_pose: tuple[float, float, float, float]  # x, y, z, yaw (radians)
    pile_center: tuple[float, float] = (0.0, 0.0)
    pile_queue: deque[int] = field(default_factory=deque)  # cube indices this rover may pick
    dome_queue: deque[int] = field(default_factory=deque)  # cube-index ordered list

    # Joint adrs (looked up in __post_init__)
    arm_qposadrs: list[int] = field(default_factory=list)
    arm_dofadrs: list[int] = field(default_factory=list)
    arm_actuator_ids: list[int] = field(default_factory=list)
    gripper_qposadrs: list[int] = field(default_factory=list)
    gripper_dofadrs: list[int] = field(default_factory=list)
    gripper_actuator_ids: list[int] = field(default_factory=list)
    ee_site_id: int = 0
    arm_yaw_body_id: int = 0
    rover_qposadr: int = 0
    wheel_actuator_ids: list[int] = field(default_factory=list)

    # Active state
    targets: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    gripper: float = 1.0
    holding: bool = False
    held_cube_idx: int | None = None
    placed_count: int = 0

    def __post_init__(self) -> None:
        p = self.prefix
        arm_joints = [f"{p}arm_yaw", f"{p}arm_shoulder", f"{p}arm_elbow", f"{p}arm_wrist"]
        arm_actuators = [f"{p}arm_yaw_pos", f"{p}arm_shoulder_pos",
                         f"{p}arm_elbow_pos", f"{p}arm_wrist_pos"]
        gripper_joints = [f"{p}left_gripper_slide", f"{p}right_gripper_slide"]
        gripper_actuators = [f"{p}left_gripper_pos", f"{p}right_gripper_pos"]
        wheel_actuators = [f"{p}wheel_fl_motor", f"{p}wheel_fr_motor",
                           f"{p}wheel_rl_motor", f"{p}wheel_rr_motor"]

        self.arm_qposadrs = [self.model.jnt_qposadr[self.model.joint(j).id] for j in arm_joints]
        self.arm_dofadrs = [self.model.jnt_dofadr[self.model.joint(j).id] for j in arm_joints]
        self.arm_actuator_ids = [self.model.actuator(a).id for a in arm_actuators]
        self.gripper_qposadrs = [self.model.jnt_qposadr[self.model.joint(j).id] for j in gripper_joints]
        self.gripper_dofadrs = [self.model.jnt_dofadr[self.model.joint(j).id] for j in gripper_joints]
        self.gripper_actuator_ids = [self.model.actuator(a).id for a in gripper_actuators]
        self.ee_site_id = self.model.site(f"{p}ee_site").id
        self.arm_yaw_body_id = self.model.body(f"{p}arm_yaw_body").id
        self.rover_qposadr = int(self.model.jnt_qposadr[self.model.joint(f"{p}rover_free").id])
        self.wheel_actuator_ids = [self.model.actuator(a).id for a in wheel_actuators]

        # Rest pose matches MarsArmPickPlaceBridge: arm up & forward, gripper open.
        self.targets[:] = np.array([0.0, -0.35, 0.65, -0.25], dtype=np.float32)
        self.gripper = 1.0

    # ---- pose / kinematics helpers ----
    def rover_pose(self) -> tuple[float, float, float]:
        qa = self.rover_qposadr
        x = float(self.data.qpos[qa])
        y = float(self.data.qpos[qa + 1])
        quat = self.data.qpos[qa + 3:qa + 7]
        w, _, _, qz = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
        yaw = math.atan2(2.0 * w * qz, 1.0 - 2.0 * qz * qz)
        return x, y, yaw

    def arm_yaw_body_world(self) -> np.ndarray:
        mujoco.mj_forward(self.model, self.data)
        return np.array(self.data.xpos[self.arm_yaw_body_id], dtype=np.float64)

    def ee_world(self) -> np.ndarray:
        mujoco.mj_forward(self.model, self.data)
        return np.array(self.data.site_xpos[self.ee_site_id], dtype=np.float64)

    def set_rover_pose(self, x: float, y: float, z: float, yaw: float) -> None:
        qa = self.rover_qposadr
        self.data.qpos[qa] = x
        self.data.qpos[qa + 1] = y
        self.data.qpos[qa + 2] = z
        self.data.qpos[qa + 3] = math.cos(yaw / 2.0)
        self.data.qpos[qa + 4] = 0.0
        self.data.qpos[qa + 5] = 0.0
        self.data.qpos[qa + 6] = math.sin(yaw / 2.0)
        # Zero rover velocity
        dofadr = int(self.model.jnt_dofadr[self.model.joint(f"{self.prefix}rover_free").id])
        self.data.qvel[dofadr:dofadr + 6] = 0.0

    def stop_wheels(self) -> None:
        for aid in self.wheel_actuator_ids:
            self.data.ctrl[aid] = 0.0

    def _arm_low(self) -> np.ndarray:
        return np.array([
            self.model.jnt_range[self.model.joint(f"{self.prefix}{j}").id, 0]
            for j in ("arm_yaw", "arm_shoulder", "arm_elbow", "arm_wrist")
        ], dtype=np.float32)

    def _arm_high(self) -> np.ndarray:
        return np.array([
            self.model.jnt_range[self.model.joint(f"{self.prefix}{j}").id, 1]
            for j in ("arm_yaw", "arm_shoulder", "arm_elbow", "arm_wrist")
        ], dtype=np.float32)

    # ---- step: write ctrl + teleport arm qpos for this rover ----
    def apply_action(self, action: np.ndarray) -> None:
        """Consume one 5D action [d_yaw, d_shoulder, d_elbow, d_wrist, gripper]
        and write this rover's arm + gripper ctrl. Does not touch wheel ctrl."""
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        deltas = np.clip(action[:4], -1.0, 1.0) * SWARM_ARM_DELTA_SCALE
        self.targets = np.clip(self.targets + deltas, self._arm_low(), self._arm_high())
        self.gripper = float(np.clip(
            self.gripper + SWARM_GRIPPER_STEP * float(action[4]), 0.0, 1.0
        ))

        for aid, target in zip(self.arm_actuator_ids, self.targets):
            self.data.ctrl[aid] = float(target)
        left, right = self.gripper_actuator_ids
        opening = self.gripper
        left_pos = GRIPPER_CLOSED_LEFT * (1.0 - opening)
        right_pos = GRIPPER_CLOSED_RIGHT * (1.0 - opening)
        self.data.ctrl[left] = left_pos
        self.data.ctrl[right] = right_pos

        # Teleport arm qpos to commanded targets (matches single-rover bridge behavior).
        for qa, da, t in zip(self.arm_qposadrs, self.arm_dofadrs, self.targets):
            self.data.qpos[qa] = float(t)
            self.data.qvel[da] = 0.0
        for qa, da, p in zip(self.gripper_qposadrs, self.gripper_dofadrs, [left_pos, right_pos]):
            self.data.qpos[qa] = float(p)
            self.data.qvel[da] = 0.0

    def update_grasp(self, swarm: "MarsSwarmBridge") -> None:
        """Track the held cube. If gripper open near a candidate pile cube, grasp it.
        If gripper opens while holding, release."""
        ee = self.ee_world()
        if self.holding:
            if self.gripper > 0.45:
                # Release: snap cube exactly onto its dome target so the structure
                # is correct regardless of where the EE happens to be hovering.
                # The placed cube is then pinned every substep via the bridge's
                # placed_cube_targets dict (cube-cube collisions are off, so without
                # pinning the cube would just drop to the ground).
                idx = self.held_cube_idx
                self.holding = False
                self.held_cube_idx = None
                if idx is not None:
                    swarm._set_cube_collidable(idx, True)
                    qvel0 = swarm.cube_qveladrs[idx]
                    self.data.qvel[qvel0:qvel0 + 6] = 0.0
                    target_idx = swarm.held_cube_target.get(idx)
                    if target_idx is not None:
                        tgt_pos = np.array(swarm.dome_targets[target_idx], dtype=np.float64)
                        swarm._set_cube_pose_idx(idx, tgt_pos)
                        swarm.placed_cube_targets[idx] = tgt_pos
                        swarm.cube_placed[target_idx] = True
                        self.placed_count += 1
                        swarm.held_cube_target.pop(idx, None)
            else:
                cube_pos = ee + HELD_OFFSET
                swarm._set_cube_pose_idx(self.held_cube_idx, cube_pos)
            return
        # Not holding: check candidate (the next pile cube this rover wants).
        candidate = swarm.next_pile_cube_for(self)
        if candidate is None:
            return
        cube_pos = self.data.xpos[swarm.cube_body_ids[candidate]]
        if self.gripper < 0.25 and np.linalg.norm(ee - cube_pos) <= PICK_DISTANCE:
            swarm.grasp_reserved_cube(self)


class MarsSwarmBridge:
    """Owns the model + data + cubes + the 3 ``RoverUnit``s."""

    def __init__(self):
        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None
        self.rovers: list[RoverUnit] = []
        # Cube state
        self.cube_body_ids: list[int] = []
        self.cube_geom_ids: list[int] = []
        self.cube_qposadrs: list[int] = []
        self.cube_qveladrs: list[int] = []
        self.cube_default_contype: list[int] = []
        self.cube_default_conaffinity: list[int] = []
        # Dome targets (shared, indexed 0..35)
        self.dome_targets: list[np.ndarray] = []
        self.cube_placed: list[bool] = [False] * NUM_CUBES
        # Pile state
        self.pile_queue: deque[int] = deque()   # cube indices available to grasp, top-down
        self.pile_pending: dict[int, int] = {}  # cube_idx -> rover_idx (claimed but not yet grasped)
        # Reservation: which rover is currently allowed to be at pile staging
        self.pile_lock: int | None = None
        # Tracking: held_cube_idx -> dome target index (for cube_placed bookkeeping)
        self.held_cube_target: dict[int, int] = {}
        # Per-rover "next dome target to place at"
        self.rover_next_dome: list[int | None] = [None, None, None]
        # Cubes that have been released onto the dome; pinned at their target xyz
        # every substep so cube-cube collisions can stay off without the dome
        # collapsing.
        self.placed_cube_targets: dict[int, np.ndarray] = {}
        # Cubes still sitting in their pile; pinned at their initial pile position
        # so the stacks don't collapse (cube-cube collisions are off). Entries are
        # removed when the cube is grasped.
        self.pile_pinned: dict[int, np.ndarray] = {}

    # ---- reset / setup ----
    def reset(self, seed: int = 0) -> None:
        np.random.seed(seed)
        cwd = os.getcwd()
        try:
            os.chdir(SCENE_DIR)
            self.model = mujoco.MjModel.from_xml_path("mars_scene_swarm.xml")
        finally:
            os.chdir(cwd)
        self.data = mujoco.MjData(self.model)

        # Look up cube state
        self.cube_body_ids = []
        self.cube_geom_ids = []
        self.cube_qposadrs = []
        self.cube_qveladrs = []
        self.cube_default_contype = []
        self.cube_default_conaffinity = []
        for i in range(NUM_CUBES):
            j = self.model.joint(f"pick_cube_{i}_free").id
            self.cube_qposadrs.append(int(self.model.jnt_qposadr[j]))
            self.cube_qveladrs.append(int(self.model.jnt_dofadr[j]))
            self.cube_body_ids.append(self.model.body(f"pick_cube_{i}").id)
            g = self.model.geom(f"pick_cube_{i}_geom").id
            self.cube_geom_ids.append(g)
            self.cube_default_contype.append(int(self.model.geom_contype[g]))
            self.cube_default_conaffinity.append(int(self.model.geom_conaffinity[g]))

        # Pin cubes into their per-pile positions and compute dome targets.
        all_pile_positions: list[np.ndarray] = []
        for k in range(NUM_ROVERS):
            cx, cy = _pile_center_for(k)
            all_pile_positions.extend(_swarm_pile_positions(cx, cy))
        self.pile_pinned = {}
        for i, pos in enumerate(all_pile_positions):
            self._set_cube_pose_idx(i, pos)
            self.pile_pinned[i] = np.array(pos, dtype=np.float64)
        self.dome_targets = _dome_positions(DOME_CENTER)

        # Partition dome targets by sector. Each rover gets a deque of dome target
        # indices ordered tier-by-tier (which _dome_positions already does).
        dome_queues: list[deque[int]] = [deque(), deque(), deque()]
        cx, cy = DOME_CENTER[0], DOME_CENTER[1]
        for ti, tgt in enumerate(self.dome_targets):
            angle = math.atan2(tgt[1] - cy, tgt[0] - cx)
            rover = _azimuth_to_rover(angle)
            dome_queues[rover].append(ti)

        # Spawn poses for rovers (these match the XML's <body pos="..."> values).
        spawn_radius = 0.65
        spawns = []
        for angle_deg in (270.0, 150.0, 30.0):
            a = math.radians(angle_deg)
            sx = spawn_radius * math.cos(a)
            sy = spawn_radius * math.sin(a)
            # face dome center
            yaw = math.atan2(-sy, -sx)
            spawns.append((sx, sy, 3.55, yaw))

        # Build RoverUnits with their assigned queues. Each rover owns 12 cubes:
        # rover k -> cube indices [12k, 12(k+1)).
        self.rovers = []
        for k, prefix in enumerate(ROVER_PREFIXES):
            rover_pile_queue = deque(range(k * CUBES_PER_PILE, (k + 1) * CUBES_PER_PILE))
            self.rovers.append(RoverUnit(
                model=self.model,
                data=self.data,
                prefix=prefix,
                spawn_pose=spawns[k],
                pile_center=_pile_center_for(k),
                pile_queue=rover_pile_queue,
                dome_queue=dome_queues[k],
            ))
            self.rovers[k].set_rover_pose(*spawns[k])

        self.pile_queue = deque()  # unused (kept for back-compat)
        self.pile_pending.clear()
        self.held_cube_target.clear()
        self.placed_cube_targets.clear()
        self.cube_placed = [False] * NUM_CUBES
        self.rover_next_dome = [None, None, None]
        self.pile_lock = None

        # Apply rest-pose targets and brief settle.
        for r in self.rovers:
            r.apply_action(np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)
        for i, pos in enumerate(all_pile_positions):
            self._set_cube_pose_idx(i, pos)
        for k in range(3):
            self.rovers[k].set_rover_pose(*spawns[k])
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    # ---- step: each rover writes its action; physics runs once ----
    def step(self, actions: list[np.ndarray]) -> None:
        assert len(actions) == NUM_ROVERS
        for r, a in zip(self.rovers, actions):
            r.apply_action(a)
        for _ in range(PHYSICS_STEPS_PER_ACTION):
            mujoco.mj_step(self.model, self.data)
            for r in self.rovers:
                r.update_grasp(self)
            # Pin every cube that's not currently being held: pile cubes stay at
            # their initial pile slot (so the stack doesn't collapse) and placed
            # cubes stay at their dome target (so the dome doesn't fall apart).
            for idx, pos in self.pile_pinned.items():
                self._set_cube_pose_idx(idx, pos)
            for idx, pos in self.placed_cube_targets.items():
                self._set_cube_pose_idx(idx, pos)

    # ---- cube state helpers ----
    def _set_cube_pose_idx(self, idx: int, pos: np.ndarray) -> None:
        qa = self.cube_qposadrs[idx]
        va = self.cube_qveladrs[idx]
        self.data.qpos[qa:qa + 3] = pos
        self.data.qpos[qa + 3:qa + 7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[va:va + 6] = 0.0

    def _set_cube_collidable(self, idx: int, collidable: bool) -> None:
        g = self.cube_geom_ids[idx]
        if collidable:
            self.model.geom_contype[g] = self.cube_default_contype[idx]
            self.model.geom_conaffinity[g] = self.cube_default_conaffinity[idx]
        else:
            self.model.geom_contype[g] = 0
            self.model.geom_conaffinity[g] = 0

    # ---- pile queue / claim helpers ----
    def next_pile_cube_for(self, rover: RoverUnit) -> int | None:
        """Return the cube index this rover should attempt to grasp next.
        Each rover, when at the pile, claims the next available pile cube.
        Returns None if no cube has been claimed yet for this rover."""
        rover_idx = self.rovers.index(rover)
        for cube_idx, owner in self.pile_pending.items():
            if owner == rover_idx:
                return cube_idx
        return None

    def claim_pile_cube_for(self, rover_idx: int) -> int | None:
        """Pop the next available cube from rover ``rover_idx``'s own pile and
        reserve it for that rover.

        A reservation survives a missed grasp.  Previously a rover that missed
        a cube would reserve a different cube on its next trip, while retaining
        the first cube's target assignment.  That made the cube-to-slot mapping
        drift and allowed the visual build to collapse into one location.
        """
        for cube_idx, owner in self.pile_pending.items():
            if owner == rover_idx:
                return cube_idx

        rover = self.rovers[rover_idx]
        if not rover.pile_queue or not rover.dome_queue:
            return None
        idx = rover.pile_queue.popleft()
        self.pile_pending[idx] = rover_idx
        return idx

    def claim_pile_cube(self, cube_idx: int, rover: RoverUnit) -> None:
        """Called by update_grasp the moment grasp triggers. Pops the dome slot
        from the rover's queue so subsequent grasps target the next slot.

        Assigning at grasp time, rather than reservation time, makes a target
        exclusive only once its cube is actually in a rover's gripper.
        """
        rover_idx = self.rovers.index(rover)
        if not rover.dome_queue:
            raise RuntimeError(f"rover {rover_idx} grasped a cube without a dome target")
        tgt = rover.dome_queue.popleft()
        self.held_cube_target[cube_idx] = tgt
        self.pile_pending.pop(cube_idx, None)

    def grasp_reserved_cube(self, rover: RoverUnit) -> bool:
        """Move a rover's reserved cube into its gripper.

        The arm controller teleports joint positions for the scripted demo, so
        checking one physics substep for a small Euclidean grasp radius is not
        reliable. This operation is the deterministic completion of a planned
        pickup and preserves the same ownership and collision transitions as a
        naturally detected grasp.
        """
        if rover.holding:
            return True
        cube_idx = self.next_pile_cube_for(rover)
        if cube_idx is None:
            return False
        rover.holding = True
        rover.held_cube_idx = cube_idx
        self.claim_pile_cube(cube_idx, rover)
        self.pile_pinned.pop(cube_idx, None)
        self._set_cube_collidable(cube_idx, False)
        self._set_cube_pose_idx(cube_idx, rover.ee_world() + HELD_OFFSET)
        return True

    # ---- pile lock helpers ----
    def acquire_pile(self, rover_idx: int) -> bool:
        if self.pile_lock is None:
            self.pile_lock = rover_idx
            return True
        return self.pile_lock == rover_idx

    def release_pile(self, rover_idx: int) -> None:
        if self.pile_lock == rover_idx:
            self.pile_lock = None

    # ---- progress ----
    def placed_count(self) -> int:
        return sum(self.cube_placed)

    def all_done(self) -> bool:
        return all(self.cube_placed)

    def close(self) -> None:
        self.data = None
        self.model = None
