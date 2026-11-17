from pathlib import Path
import math

import cv2
import mujoco
import numpy as np
import torch
import torch.nn as nn


# -------------------------
# 1. Paths
# -------------------------

PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

PYTHON_ROOT = PROJECT_ROOT.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "two_link_arm.xml"
)

CNN_PATH = (
    PYTHON_ROOT
    / "pytorch-learning"
    / "checkpoints"
    / "ball_position_cnn.pt"
)


# -------------------------
# 2. CNN architecture
# -------------------------

class BallPositionCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            padding=1,
        )

        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1,
        )

        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1,
        )

        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(
            64 * 7 * 10,
            128,
        )

        self.relu4 = nn.ReLU()

        self.fc2 = nn.Linear(
            128,
            2,
        )

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = torch.flatten(
            x,
            start_dim=1,
        )

        x = self.fc1(x)
        x = self.relu4(x)

        position = self.fc2(x)

        return position


# -------------------------
# 3. Load trained CNN
# -------------------------

checkpoint = torch.load(
    CNN_PATH,
    map_location="cpu",
)

cnn = BallPositionCNN()

cnn.load_state_dict(
    checkpoint["model_state_dict"]
)

cnn.eval()

print("CNN loaded:")
print(CNN_PATH)


# -------------------------
# 4. Load MuJoCo
# -------------------------

mj_model = mujoco.MjModel.from_xml_path(
    str(MODEL_PATH)
)

data = mujoco.MjData(
    mj_model
)


# -------------------------
# 5. Robot pose
# -------------------------

# Match the robot pose used when
# the CNN training images were generated.

data.qpos[:] = np.array(
    [0.5, -0.8]
)

data.qvel[:] = np.array(
    [0.0, 0.0]
)


# -------------------------
# 6. Ball position
# -------------------------

true_ball_position = np.array(
    [
        1.0,
        0.5,
        0.08,
    ]
)

data.mocap_pos[0] = (
    true_ball_position
)

mujoco.mj_forward(
    mj_model,
    data,
)


# -------------------------
# 7. Renderer / camera
# -------------------------

renderer = mujoco.Renderer(
    mj_model,
    height=480,
    width=640,
)

camera = mujoco.MjvCamera()

mujoco.mjv_defaultFreeCamera(
    mj_model,
    camera,
)

camera.lookat[:] = np.array(
    [0.5, 0.0, 0.0]
)

camera.distance = 4.0
camera.azimuth = 90.0
camera.elevation = -90.0


# -------------------------
# 8. Render image
# -------------------------

renderer.update_scene(
    data,
    camera=camera,
)

rgb = renderer.render()

print()
print(
    "Rendered image shape:",
    rgb.shape,
)


# -------------------------
# 9. CNN preprocessing
# -------------------------

image = cv2.resize(
    rgb,
    (80, 60),
)

# uint8 [0, 255]
# ->
# float32 [0, 1]

image = (
    image.astype(np.float32)
    / 255.0
)

# [height, width, channel]
# ->
# [channel, height, width]

image = np.transpose(
    image,
    (2, 0, 1),
)

image_tensor = torch.tensor(
    image,
    dtype=torch.float32,
)

# One image is still a batch.
#
# [3, 60, 80]
# ->
# [1, 3, 60, 80]

image_tensor = (
    image_tensor.unsqueeze(0)
)

print(
    "CNN input shape:",
    image_tensor.shape,
)


# -------------------------
# 10. Vision inference
# -------------------------

with torch.no_grad():

    prediction = cnn(
        image_tensor
    )[0]

predicted_x = (
    prediction[0].item()
)

predicted_y = (
    prediction[1].item()
)

predicted_position = np.array(
    [
        predicted_x,
        predicted_y,
    ]
)

print()
print(
    "True ball position:",
    true_ball_position[:2],
)

print(
    "CNN predicted position:",
    predicted_position,
)


# -------------------------
# 11. Vision error
# -------------------------

vision_error = np.linalg.norm(
    predicted_position
    - true_ball_position[:2]
)

print(
    "Vision error:",
    vision_error,
)


# -------------------------
# 12. Robot parameters
# -------------------------

L1 = 1.0
L2 = 0.8


# -------------------------
# 13. Inverse kinematics
# -------------------------

def inverse_kinematics(
    x,
    y,
):

    r_squared = (
        x**2
        + y**2
    )

    cos_q2 = (
        r_squared
        - L1**2
        - L2**2
    ) / (
        2.0 * L1 * L2
    )

    if (
        cos_q2 < -1.0
        or cos_q2 > 1.0
    ):
        raise ValueError(
            "Predicted position is "
            "outside the robot workspace."
        )

    q2 = math.acos(
        cos_q2
    )

    q1 = (
        math.atan2(
            y,
            x,
        )
        -
        math.atan2(
            L2 * math.sin(q2),
            L1
            + L2 * math.cos(q2),
        )
    )

    return np.array(
        [
            q1,
            q2,
        ]
    )


q_des = inverse_kinematics(
    predicted_x,
    predicted_y,
)


# -------------------------
# 14. FK verification
# -------------------------

q1 = q_des[0]
q2 = q_des[1]

fk_x = (
    L1 * math.cos(q1)
    + L2
    * math.cos(q1 + q2)
)

fk_y = (
    L1 * math.sin(q1)
    + L2
    * math.sin(q1 + q2)
)

fk_position = np.array(
    [
        fk_x,
        fk_y,
    ]
)


# -------------------------
# 15. Results
# -------------------------

print()
print(
    "Desired joint angles [rad]:",
    q_des,
)

print(
    "Desired joint angles [deg]:",
    np.degrees(q_des),
)

print()
print(
    "FK of desired joint angles:",
    fk_position,
)

print(
    "IK/FK consistency error:",
    np.linalg.norm(
        fk_position
        - predicted_position
    ),
)
