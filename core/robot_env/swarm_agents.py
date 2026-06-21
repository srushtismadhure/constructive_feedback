"""VLA agents for the mars-swarm-manip env's `robot` capability.

Mirrors robot_training/agents/vla_agent.py, but the action is the swarm rover's 5-DoF
arm `[d_yaw, d_shoulder, d_elbow, d_wrist, gripper]` (gripper > 0 opens), not the
7-DoF franka ee-delta. The harness (`RobotAgent`) owns connect/loop/chunking; we only
supply the `Model.infer` seam (or, for `RemoteAgent`, a websocket client to a GPU box).
"""
from __future__ import annotations

from typing import Any

import numpy as np
from hud.agents.robot.agent import RobotAgent
from hud.agents.robot.model import Model

# Hold the arm, gripper open — the 5-DoF analogue of the franka no-op.
_NOOP_ACTION = np.array([[0.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32)


class NoopModel(Model):
    """Holds position, gripper open — proves the wire works with no GPU/model."""

    def infer(self, batch: Any) -> np.ndarray:
        return _NOOP_ACTION


class NoopAgent(RobotAgent):
    """Plumbing check: connect, observe, send a 5-DoF no-op every step. No torch."""

    max_steps = 200
    adapter = None  # raw pass-through: the no-op ignores the observation

    def __init__(self) -> None:
        self.model = NoopModel()


class RemoteAgent(RobotAgent):
    """Run the policy on a remote GPU box (serve/policy_server_mars.py); keep the sim
    + loop here. `OpenPIAdapter` ships the env's raw contract observation as-is (single
    camera + 9-dim state + prompt); the server maps it onto the policy."""

    max_steps = 200

    def __init__(self, host: str = "localhost", port: int = 8000) -> None:
        from hud.agents.robot.adapter import OpenPIAdapter
        from hud.agents.robot.model import RemoteModel

        self.model = RemoteModel(host, port)  # response_key="actions" (serve default)
        self.adapter = OpenPIAdapter()


__all__ = ["NoopAgent", "NoopModel", "RemoteAgent"]
