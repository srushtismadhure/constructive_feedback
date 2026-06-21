"""HUD environment: the Mars swarm single-rover pick-and-place manip task.

Serves ``SwarmManipBridge`` over the ``robot`` (openpi/0) capability so a VLA policy
drives the 5-DoF arm and is graded on placement. In-process bridge (mujoco runs here),
so this uses the ``RobotEndpoint(bridge)`` + ``endpoint.start()`` pattern from
``hud_env.py``; grading mirrors ``robot_training/environment/vla_env.py``.

Serve it like any HUD env: ``run_swarm_vla.py`` (LocalRuntime), a container CMD, or
``python -m hud.environment.server robot_env/swarm_hud_env.py``.
"""
from __future__ import annotations

from hud.environment import Environment
from hud.environment.robot import RobotEndpoint
from hud.graders import EvaluationResult, SubScore
from swarm_manip_bridge import CONTRACT, SwarmManipBridge

env = Environment(name="mars-swarm-manip")
endpoint = RobotEndpoint(SwarmManipBridge(render=True))


@env.initialize
async def _up() -> None:
    await endpoint.start()
    env.add_capability(await endpoint.capability(contract=CONTRACT))


@env.shutdown
async def _down() -> None:
    await endpoint.stop()


@env.template(id="swarm-pick", description="One rover picks a block and places it on its wedge.")
async def swarm_pick(task_id: str = "south", seed: int = 0):
    """Reset the manip task, let the agent's policy drive the arm, grade on placement.

    Reward is the bridge's shaped score (approach < grasped 0.5 < placed 1.0), emitted
    as a single self-consistent subscore.
    """
    prompt = await endpoint.reset(task_id=task_id, seed=seed)
    yield {"prompt": prompt}

    res = await endpoint.result()
    score, success = res["score"], res["success"]
    yield EvaluationResult(
        reward=round(score, 4),
        done=True,
        content=f"swarm pick: score={score:.3f} ({'SUCCESS' if success else 'INCOMPLETE'})",
        subscores=[SubScore(name="placement", weight=1.0, value=round(score, 4))],
    )
