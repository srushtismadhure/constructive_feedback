import argparse
import asyncio
import time

import mujoco
import mujoco.viewer

from hud_arm_bridge import MarsArmPickPlaceBridge, _scripted_actions


WIREFRAME_FLAG = int(mujoco.mjtRndFlag.mjRND_WIREFRAME)
NO_FRAME = int(mujoco.mjtFrame.mjFRAME_NONE)


def keep_clean_view(viewer):
    viewer.opt.frame = NO_FRAME
    if viewer.user_scn is not None:
        viewer.user_scn.flags[WIREFRAME_FLAG] = 0


async def run_demo(rate_hz: float) -> None:
    bridge = MarsArmPickPlaceBridge(render=False)
    prompt = await bridge.reset(task_id="arm-viewer-demo", seed=0)
    assert bridge.model is not None
    assert bridge.data is not None
    print(prompt)
    print("Opening viewer. Scripted HUD arm actions are being applied.")

    actions = _scripted_actions()
    action_index = 0
    dt = 1.0 / rate_hz
    last_step = 0.0
    last_print = time.time()
    start = time.time()

    with mujoco.viewer.launch_passive(bridge.model, bridge.data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = [1.55, -0.35, 3.65]
        viewer.cam.distance = 2.2
        viewer.cam.elevation = -25
        viewer.cam.azimuth = 135
        keep_clean_view(viewer)

        while viewer.is_running():
            now = time.time()
            elapsed = now - start
            with viewer.lock():
                if action_index < len(actions) and elapsed - last_step >= dt:
                    bridge.step(actions[action_index])
                    action_index += 1
                    last_step = elapsed

                if now - last_print > 1.0:
                    obs, terminated = bridge.get_observation()
                    state = obs["observation/state"]
                    print(
                        f"\rcube=({state[8]:+.2f},{state[9]:+.2f},{state[10]:+.2f}) "
                        f"holding={state[14]:.0f} success={state[15]:.0f} "
                        f"action={action_index}/{len(actions)} terminated={terminated}      ",
                        end="",
                        flush=True,
                    )
                    last_print = now

            keep_clean_view(viewer)
            viewer.sync()
            keep_clean_view(viewer)
            time.sleep(0.005)

    bridge.close()
    print("\nDemo closed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visual demo for the HUD pick/place arm bridge.")
    parser.add_argument("--rate-hz", type=float, default=10.0)
    args = parser.parse_args()
    asyncio.run(run_demo(args.rate_hz))


if __name__ == "__main__":
    main()
