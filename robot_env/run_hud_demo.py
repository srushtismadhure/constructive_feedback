import argparse
import asyncio
import math
import time

import mujoco
import mujoco.viewer
import numpy as np

from hud_mujoco_bridge import MarsMujocoBridge, ROBOT_BODY


WIREFRAME_FLAG = int(mujoco.mjtRndFlag.mjRND_WIREFRAME)
NO_FRAME = int(mujoco.mjtFrame.mjFRAME_NONE)


def keep_clean_view(viewer):
    viewer.opt.frame = NO_FRAME
    if viewer.user_scn is not None:
        viewer.user_scn.flags[WIREFRAME_FLAG] = 0


def demo_action(t: float) -> np.ndarray:
    # Drive forward while slowly oscillating turn rate.
    return np.array(
        [
            0.65,
            0.45 * math.sin(0.7 * t),
        ],
        dtype=np.float32,
    )


async def run_demo(rate_hz: float) -> None:
    bridge = MarsMujocoBridge(render=False)
    prompt = await bridge.reset(task_id="viewer-demo", seed=0)
    assert bridge.model is not None
    assert bridge.data is not None

    print(prompt)
    print("Opening viewer. HUD bridge actions are being applied continuously.")

    dt = 1.0 / rate_hz
    start = time.time()
    last_step = 0.0
    last_print = start

    with mujoco.viewer.launch_passive(bridge.model, bridge.data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = bridge.model.body(ROBOT_BODY).id
        viewer.cam.distance = 4.0
        viewer.cam.elevation = -25
        viewer.cam.azimuth = 135
        keep_clean_view(viewer)

        while viewer.is_running():
            loop_start = time.time()
            t = loop_start - start

            with viewer.lock():
                if t - last_step >= dt:
                    bridge.step(demo_action(t))
                    last_step = t

                if loop_start - last_print > 1.0:
                    obs, terminated = bridge.get_observation()
                    state = obs["observation/state"]
                    print(
                        f"\rrover x={state[0]:+.2f} y={state[1]:+.2f} "
                        f"z={state[2]:+.2f} yaw={math.degrees(state[3]):+.1f} "
                        f"vx={state[4]:+.2f} yaw_rate={state[7]:+.2f} "
                        f"terminated={terminated}      ",
                        end="",
                        flush=True,
                    )
                    last_print = loop_start

            keep_clean_view(viewer)
            viewer.sync()
            keep_clean_view(viewer)

            elapsed = time.time() - loop_start
            remaining = min(0.01, max(0.0, dt - elapsed))
            if remaining:
                time.sleep(remaining)

    bridge.close()
    print("\nDemo closed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visual demo for the HUD MuJoCo bridge.")
    parser.add_argument("--rate-hz", type=float, default=10.0)
    args = parser.parse_args()
    asyncio.run(run_demo(args.rate_hz))


if __name__ == "__main__":
    main()
