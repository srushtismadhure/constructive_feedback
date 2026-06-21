from hud import Environment
from hud.environment.robot import RobotEndpoint

from hud_mujoco_bridge import CONTRACT, MarsMujocoBridge


env = Environment(name="mars-mujoco-ingenuity")
endpoint = RobotEndpoint(MarsMujocoBridge())


@env.initialize
async def _up():
    await endpoint.start()
    env.add_capability(await endpoint.capability(contract=CONTRACT))


@env.shutdown
async def _down():
    await endpoint.stop()


@env.template()
async def mars_ingenuity(task_id: str = "move-ingenuity", seed: int = 0):
    prompt = await endpoint.reset(task_id=task_id, seed=seed)
    yield {"prompt": prompt}
    yield await endpoint.result()
