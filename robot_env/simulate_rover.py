import os
import time

import mujoco
import mujoco.viewer

# ─── Paths ────────────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
scene_xml_path = os.path.join(script_dir, "mars_scene.xml")

print(f"Loading scene: {scene_xml_path}")

try:
    model = mujoco.MjModel.from_xml_path(scene_xml_path)
    data = mujoco.MjData(model)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

mujoco.mj_forward(model, data)

ingenuity_pos = data.body("ingenuity_display").xpos
print(
    "Ingenuity display position: "
    f"x={ingenuity_pos[0]:.2f}, y={ingenuity_pos[1]:.2f}, z={ingenuity_pos[2]:.2f}"
)
print("Viewer opening …\n")

WIREFRAME_FLAG = int(mujoco.mjtRndFlag.mjRND_WIREFRAME)
NO_FRAME = int(mujoco.mjtFrame.mjFRAME_NONE)


def keep_clean_view(viewer):
    """Keep the passive scene viewer free of debug overlays."""
    viewer.opt.frame = NO_FRAME
    if viewer.user_scn is not None:
        viewer.user_scn.flags[WIREFRAME_FLAG] = 0


print("═" * 55)
print("  MARS SURFACE + INGENUITY DISPLAY")
print("═" * 55)
print("  Mouse L-drag — Pan camera")
print("  Mouse R-drag — Orbit camera")
print("  Scroll       — Zoom")
print("═" * 55 + "\n")

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = model.body("ingenuity_display").id
    viewer.cam.distance = 4.0
    viewer.cam.elevation = -25
    viewer.cam.azimuth = 135
    keep_clean_view(viewer)

    last_print = time.time()

    while viewer.is_running():
        step_start = time.time()

        mujoco.mj_step(model, data)

        now = time.time()
        if now - last_print > 1.0:
            pos = data.body("ingenuity_display").xpos
            print(
                f"\r  Ingenuity display: x={pos[0]:.2f} y={pos[1]:.2f} z={pos[2]:.2f}  ",
                end="",
                flush=True,
            )
            last_print = now

        keep_clean_view(viewer)
        viewer.sync()
        keep_clean_view(viewer)

        elapsed = time.time() - step_start
        remaining = model.opt.timestep - elapsed
        if remaining > 0:
            time.sleep(remaining)

print("\nSimulation closed.")
