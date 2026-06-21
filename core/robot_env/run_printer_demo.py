"""Visual demo: Mars 3-D printer arm running the scripted dome-print policy.

Run with mjpython on macOS:
    mjpython run_printer_demo.py              # full 20-waypoint dome
    mjpython run_printer_demo.py --waypoints 5  # lite: first 5 waypoints only
    mjpython run_printer_demo.py --rate-hz 60   # faster playback
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

import mujoco
import mujoco.viewer
import numpy as np

from hud_printer_bridge import (
    EXTRUDE_THRESHOLD,
    MarsPrinterBridge,
    PRINT_STRUCTURES,
    _scripted_printer_actions,
)

WIREFRAME_FLAG = int(mujoco.mjtRndFlag.mjRND_WIREFRAME)
NO_FRAME = int(mujoco.mjtFrame.mjFRAME_NONE)

# Deposited-filament bead appearance.
BEAD_RADIUS = 0.022
BEAD_RGBA = np.array([0.82, 0.79, 0.72, 1.0], dtype=np.float64)
BEAD_SPACING = 0.012  # min distance between consecutive beads
_IDENTITY_MAT = np.eye(3, dtype=np.float64).flatten()


def _keep_clean(viewer) -> None:
    viewer.opt.frame = NO_FRAME
    if viewer.user_scn is not None:
        viewer.user_scn.flags[WIREFRAME_FLAG] = 0


def _draw_beads(viewer, beads: list[np.ndarray]) -> None:
    """Render deposited filament as persistent spheres in the viewer overlay."""
    scn = viewer.user_scn
    if scn is None:
        return
    size = np.array([BEAD_RADIUS, 0.0, 0.0], dtype=np.float64)
    n = min(len(beads), scn.maxgeom)
    scn.ngeom = n
    for i in range(n):
        mujoco.mjv_initGeom(
            scn.geoms[i],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            size,
            beads[i],
            _IDENTITY_MAT,
            BEAD_RGBA,
        )


async def run_demo(rate_hz: float, max_waypoints: int) -> None:
    bridge = MarsPrinterBridge(render=False)
    prompt = await bridge.reset(task_id="dome", seed=0)
    assert bridge.model is not None
    assert bridge.data is not None

    # Build a lite action list: only actions up to the first max_waypoints waypoints.
    all_actions = _scripted_printer_actions("dome")
    waypoints = PRINT_STRUCTURES["dome"]()
    if max_waypoints < len(waypoints):
        # Estimate how many actions cover the first N waypoints by running headless.
        probe = MarsPrinterBridge(render=False)
        await probe.reset(task_id="dome", seed=0)
        cutoff = len(all_actions)
        for i, act in enumerate(all_actions):
            probe.step(act)
            _, done = probe.get_observation()
            if probe.current_waypoint_idx >= max_waypoints or done:
                cutoff = i + 1
                break
        probe.close()
        actions = all_actions[:cutoff]
        print(f"[lite] playing {len(actions)} actions covering first {max_waypoints} waypoints")
    else:
        actions = all_actions
        print(f"[full] playing {len(actions)} actions for all {len(waypoints)} waypoints")

    print(f"prompt: {prompt}")
    print("Opening viewer — watch the extruder trace the dome path.")

    dt = 1.0 / rate_hz
    action_index = 0
    start = time.time()
    last_step = 0.0
    last_print = start
    beads: list[np.ndarray] = []  # deposited filament positions (world frame)
    done = False  # print finished — stop dispatching but keep the viewer open

    with mujoco.viewer.launch_passive(bridge.model, bridge.data) as viewer:
        # Frame the whole tracked printer: base at (1.25,-0.5), build plate at (1.64,-0.11).
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = [1.45, -0.30, 3.55]
        viewer.cam.distance = 2.7
        viewer.cam.elevation = -22
        viewer.cam.azimuth = 150
        _keep_clean(viewer)

        while viewer.is_running():
            now = time.time()
            elapsed = now - start

            with viewer.lock():
                if not done and action_index < len(actions) and elapsed - last_step >= dt:
                    bridge.step(actions[action_index])
                    action_index += 1
                    last_step = elapsed

                    # Deposit a filament bead when the extruder is active.
                    if bridge.extruder > EXTRUDE_THRESHOLD:
                        ee = bridge.data.site_xpos[bridge.ee_site_id].copy()
                        if not beads or float(np.linalg.norm(ee - beads[-1])) > BEAD_SPACING:
                            beads.append(ee)

                if not done and now - last_print > 0.25:
                    obs, terminated = bridge.get_observation()
                    state = obs["observation/state"]
                    printed = int(state[11])
                    total = int(state[12])
                    pct = state[13] * 100
                    extruder = state[4]
                    at_tgt = int(state[14])
                    print(
                        f"\r  printed={printed:2d}/{total}  "
                        f"pct={pct:5.1f}%  extruder={extruder:.2f}  "
                        f"at_target={at_tgt}  "
                        f"action={action_index}/{len(actions)}   ",
                        end="",
                        flush=True,
                    )
                    last_print = now

                    if terminated or action_index >= len(actions):
                        result = bridge.result()
                        print(f"\n\n  Done! score={result['score']:.3f}  "
                              f"printed={result['printed_count']}/{result['total_waypoints']}  "
                              f"success={result['success']}")
                        print("  Print complete — viewer stays open. Close the window to exit.")
                        done = True

            _draw_beads(viewer, beads)
            _keep_clean(viewer)
            viewer.sync()
            _keep_clean(viewer)
            time.sleep(0.004)

    bridge.close()
    print("\nViewer closed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visual demo for the Mars 3-D printer arm.")
    parser.add_argument("--rate-hz", type=float, default=30.0,
                        help="Action dispatch rate (higher = faster playback).")
    parser.add_argument("--waypoints", type=int, default=5,
                        help="Number of waypoints to print (default 5 for a lite run).")
    args = parser.parse_args()
    asyncio.run(run_demo(args.rate_hz, args.waypoints))


if __name__ == "__main__":
    main()
