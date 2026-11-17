from pathlib import Path

import cv2
import mujoco
import numpy as np


# -------------------------
# 1. Paths
# -------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "two_link_arm.xml"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "step_11_1"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "mujoco_rgb.png"
)


# -------------------------
# 2. Load model
# -------------------------

model = mujoco.MjModel.from_xml_path(
    str(MODEL_PATH)
)

data = mujoco.MjData(model)


# -------------------------
# 3. Set robot pose
# -------------------------

data.qpos[:] = np.array(
    [0.5, -0.8]
)

data.qvel[:] = np.array(
    [0.0, 0.0]
)

mujoco.mj_forward(
    model,
    data,
)


# -------------------------
# 4. Create renderer
# -------------------------

renderer = mujoco.Renderer(
    model,
    height=480,
    width=640,
)


# -------------------------
# 5. Camera
# -------------------------

camera = mujoco.MjvCamera()

mujoco.mjv_defaultFreeCamera(
    model,
    camera,
)

camera.lookat[:] = np.array(
    [0.5, 0.0, 0.0]
)

camera.distance = 4.0
camera.azimuth = 90.0
camera.elevation = -90.0


# -------------------------
# 6. Render
# -------------------------

renderer.update_scene(
    data,
    camera=camera,
)

rgb = renderer.render()


print(
    "RGB shape:",
    rgb.shape,
)

print(
    "dtype:",
    rgb.dtype,
)

print(
    "min:",
    rgb.min(),
)

print(
    "max:",
    rgb.max(),
)


# -------------------------
# 7. Save image
# -------------------------

# MuJoCo -> RGB
# OpenCV -> BGR
bgr = cv2.cvtColor(
    rgb,
    cv2.COLOR_RGB2BGR,
)

cv2.imwrite(
    str(OUTPUT_PATH),
    bgr,
)

print()
print(
    "Saved:",
    OUTPUT_PATH,
)