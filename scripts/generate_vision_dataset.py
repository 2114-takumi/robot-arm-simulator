from pathlib import Path
import csv

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
    / "step_11_2"
)

IMAGE_DIR = (
    OUTPUT_DIR
    / "images"
)

IMAGE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

LABEL_PATH = (
    OUTPUT_DIR
    / "labels.csv"
)


# -------------------------
# 2. Load MuJoCo
# -------------------------

model = mujoco.MjModel.from_xml_path(
    str(MODEL_PATH)
)

data = mujoco.MjData(model)


print(
    "Number of mocap bodies:",
    model.nmocap,
)


# -------------------------
# 3. Robot pose
# -------------------------

data.qpos[:] = np.array(
    [0.5, -0.8]
)

data.qvel[:] = 0.0


# -------------------------
# 4. Renderer
# -------------------------

renderer = mujoco.Renderer(
    model,
    height=480,
    width=640,
)

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
# 5. Random generator
# -------------------------

rng = np.random.default_rng(
    seed=0
)


# -------------------------
# 6. Dataset settings
# -------------------------

num_images = 500

rows = []


# -------------------------
# 7. Generate images
# -------------------------

for index in range(num_images):

    # Random target position
    ball_x = rng.uniform(
        -1.0,
        1.5,
    )

    ball_y = rng.uniform(
        -1.2,
        1.2,
    )

    ball_z = 0.08

    # Move mocap body
    data.mocap_pos[0] = np.array(
        [
            ball_x,
            ball_y,
            ball_z,
        ]
    )

    # Update MuJoCo geometry
    mujoco.mj_forward(
        model,
        data,
    )

    # Render
    renderer.update_scene(
        data,
        camera=camera,
    )

    rgb = renderer.render()

    # Save image
    image_name = (
        f"image_{index:04d}.png"
    )

    image_path = (
        IMAGE_DIR
        / image_name
    )

    bgr = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR,
    )

    cv2.imwrite(
        str(image_path),
        bgr,
    )

    # Save label
    rows.append(
        [
            image_name,
            ball_x,
            ball_y,
        ]
    )


# -------------------------
# 8. Save labels
# -------------------------

with LABEL_PATH.open(
    "w",
    newline="",
    encoding="utf-8",
) as file:

    writer = csv.writer(file)

    writer.writerow(
        [
            "image",
            "ball_x",
            "ball_y",
        ]
    )

    writer.writerows(rows)


print()
print(
    "Generated images:",
    num_images,
)

print(
    "Image directory:",
    IMAGE_DIR,
)

print(
    "Labels:",
    LABEL_PATH,
)