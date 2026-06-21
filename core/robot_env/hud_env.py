from hud import Environment
from hud.environment.robot import RobotEndpoint

from hud_arm_bridge import CONTRACT, MarsArmPickPlaceBridge


env = Environment(name="mars-arm-pick-place")
endpoint = RobotEndpoint(MarsArmPickPlaceBridge())


@env.initialize
async def _up():
    await endpoint.start()
    env.add_capability(await endpoint.capability(contract=CONTRACT))


@env.shutdown
async def _down():
    await endpoint.stop()


@env.template()
async def mars_pick_place(task_id: str = "pick-place-cube", seed: int = 0):
    prompt = await endpoint.reset(task_id=task_id, seed=seed)
    yield {"prompt": prompt}
    yield await endpoint.result()
