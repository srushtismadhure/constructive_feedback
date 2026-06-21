"""Adapter: drive the single MuJoCo rover (robot_env) behind the EnvInterface.

The Brain plans for a *fleet* of robots over an 8x8 grid; the MuJoCo scene has
*one* physical rover. This adapter treats that rover as the shared embodiment:
each dispatched Action is run as a drive-to-target maneuver, in the order the
Brain dispatches them. Grid cells map linearly to world metres.

What the orchestration graph actually consumes from `step()` (see graph.py):
  - `info["rejection_reason"]` — set => the task is marked failed => replan.
  - `reward` — logged only.
The returned `Observation`/`done` are not used for control, so they are built
for completeness (and for the demo/RL paths), not relied on by the Brain.

`MarsMujocoBridge.reset` is async (HUD RobotBridge contract) but does only sync
work; we drive it with `asyncio.run`, which is safe because `graph.invoke` is
synchronous (no running event loop).
"""

from __future__ import annotations

import asyncio
import math
import sys
from pathlib import Path

from orchestration.contracts import Action, Observation

# robot_env is a sibling directory, not an installed package — put it on the path.
_ROBOT_ENV = Path(__file__).resolve().parents[1] / "robot_env"
if str(_ROBOT_ENV) not in sys.path:
    sys.path.insert(0, str(_ROBOT_ENV))

from hud_mujoco_bridge import MarsMujocoBridge  # noqa: E402


def _wrap_angle(a: float) -> float:
    """Wrap to (-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


class RoverEnvAdapter:
    """EnvInterface over `MarsMujocoBridge`: grid Actions -> rover maneuvers."""

    def __init__(
        self,
        grid_size: int = 8,
        cell_meters: float = 1.0,
        origin: tuple[float, float] = (-3.5, -3.5),
        render: bool = False,
        max_drive_steps: int = 220,
    ) -> None:
        self.grid_size = grid_size
        self.cell_meters = cell_meters
        self.origin = origin
        self.max_drive_steps = max_drive_steps
        self._bridge = MarsMujocoBridge(render=render)
        self._placed = [[0] * grid_size for _ in range(grid_size)]
        self._step = 0

    # ── EnvInterface ────────────────────────────────────────────────────────

    def reset(self, blueprint_id: str, seed: int = 42) -> Observation:
        asyncio.run(self._bridge.reset(task_id=blueprint_id, seed=seed))
        # The adapter runs many sequential goals on one loaded scene; lift the
        # bridge's single-episode step cap so it doesn't self-terminate mid-build.
        self._bridge.max_steps = 10**9
        self._placed = [[0] * self.grid_size for _ in range(self.grid_size)]
        self._step = 0
        return self._observation()

    def step(self, action: Action) -> tuple[Observation, float, bool, dict]:
        info: dict = {}
        if action.command == "noop" or action.target is None:
            return self._observation(), 0.0, False, info

        gx, gy = int(action.target[0]), int(action.target[1])
        wx, wy = self._grid_to_world(gx, gy)
        self._bridge.set_goal(wx, wy)

        reward_before = self._bridge.total_reward
        reached = self._drive_to(wx, wy)
        reward = self._bridge.total_reward - reward_before
        self._step += 1

        if not reached:
            dx, dy = self._planar_err(wx, wy)
            info["rejection_reason"] = (
                f"rover stalled {math.hypot(dx, dy):.2f} m from target ({gx}, {gy})"
            )
        elif action.command in ("place", "weld", "excavate"):
            # No manipulation in sim yet: arriving at the cell marks it built.
            if 0 <= gy < self.grid_size and 0 <= gx < self.grid_size:
                self._placed[gy][gx] = 1

        return self._observation(), reward, False, info

    def close(self) -> None:
        self._bridge.close()

    # ── internals ───────────────────────────────────────────────────────────

    def _grid_to_world(self, gx: int, gy: int) -> tuple[float, float]:
        return (
            self.origin[0] + gx * self.cell_meters,
            self.origin[1] + gy * self.cell_meters,
        )

    def _planar_err(self, wx: float, wy: float) -> tuple[float, float]:
        x, y, _, _ = self._bridge._robot_pose()
        return wx - x, wy - y

    def _drive_to(self, wx: float, wy: float) -> bool:
        """Pure-pursuit controller: turn toward the goal, crawl when facing it."""
        tol = self._bridge.goal_tol
        for _ in range(self.max_drive_steps):
            x, y, _, yaw = self._bridge._robot_pose()
            dx, dy = wx - x, wy - y
            dist = math.hypot(dx, dy)
            if dist <= tol:
                return True
            err = _wrap_angle(math.atan2(dy, dx) - yaw)
            # Full steering authority by ~60deg of heading error.
            turn = max(-1.0, min(1.0, err / (math.pi / 3)))
            # Pivot (near) in place until roughly facing the goal, then drive —
            # no forward floor, so the rover doesn't orbit the target.
            facing = max(0.0, 1.0 - abs(err) / (math.pi / 3))
            forward = min(1.0, 0.4 + dist) * facing
            self._bridge.step([forward, turn])
        return self._bridge._dist_to_goal() <= tol

    def _observation(self) -> Observation:
        # Flat terrain (the rover drives on a plane) + the cells built so far.
        return Observation(
            terrain=[[0.0] * self.grid_size for _ in range(self.grid_size)],
            placed=[row[:] for row in self._placed],
            robots=[],  # the Brain tracks the fleet in its registry, not here
            resources={"regolith": 500, "metal": 500},
            hazards=[],
            step=self._step,
        )
