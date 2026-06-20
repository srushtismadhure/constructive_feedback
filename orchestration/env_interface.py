"""Protocol that the teammate's MuJoCo adapter (or any future sim) must implement."""

from typing import Protocol, runtime_checkable

from orchestration.contracts import Action, Observation


@runtime_checkable
class EnvInterface(Protocol):
    def reset(self, blueprint_id: str, seed: int = 42) -> Observation:
        """Initialize world for the given blueprint. Returns initial observation."""
        ...

    def step(self, action: Action) -> tuple[Observation, float, bool, dict]:
        """Apply action. Returns (observation, reward, done, info).

        info["rejection_reason"] is set when an illegal action is rejected.
        """
        ...


class MockEnv:
    """Minimal stub for local testing without MuJoCo. Always succeeds."""

    def __init__(self, grid_size: int = 8):
        self._grid_size = grid_size
        self._step = 0
        self._terrain = [[0.0] * grid_size for _ in range(grid_size)]
        self._placed = [[0] * grid_size for _ in range(grid_size)]

    def reset(self, blueprint_id: str, seed: int = 42) -> Observation:
        self._step = 0
        return Observation(
            terrain=self._terrain,
            placed=self._placed,
            robots=[],
            resources={"regolith": 100, "metal": 50},
            hazards=[],
            step=0,
        )

    def step(self, action: Action) -> tuple[Observation, float, bool, dict]:
        self._step += 1
        obs = Observation(
            terrain=self._terrain,
            placed=self._placed,
            robots=[],
            resources={"regolith": 100, "metal": 50},
            hazards=[],
            step=self._step,
        )
        return obs, 1.0, False, {}
