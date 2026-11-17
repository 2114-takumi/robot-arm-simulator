from pathlib import Path
import math

import cv2
import mujoco
import numpy as np
import torch
import torch.nn as nn


# =========================================================
# 1. Paths
# =========================================================

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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "step_12_2"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# 2. CNN
# =========================================================

class BallPositionCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            3,
            16,
            kernel_size=3,
            padding=1,
        )

        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(
            16,
            32,
            kernel_size=3,
            padding=1,
        )

        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(
            32,
            64,
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

        x = self.pool1(
            self.relu1(
                self.conv1(x)
            )
        )

        x = self.pool2(
            self.relu2(
                self.conv2(x)
            )
        )

        x = self.pool3(
            self.relu3(
                self.conv3(x)
            )
        )

        x = torch.flatten(
            x,
            start_dim=1,
        )

        x = self.relu4(
            self.fc1(x)
        )

        position = self.fc2(x)

        return position


# =========================================================
# 3. Load trained CNN
# =========================================================

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


# =========================================================
# 4. MuJoCo
# =========================================================

model = mujoco.MjModel.from_xml_path(
    str(MODEL_PATH)
)

data = mujoco.MjData(model)

dt = model.opt.timestep


# =========================================================
# 5. Initial robot pose
# =========================================================

# Same pose used during CNN training.

data.qpos[:] = np.array(
    [0.5, -0.8]
)

data.qvel[:] = np.array(
    [0.0, 0.0]
)


# =========================================================
# 6. Ball
# =========================================================

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
    model,
    data,
)


# =========================================================
# 7. Camera / Renderer
# =========================================================

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


# =========================================================
# 8. Take one camera image
# =========================================================

renderer.update_scene(
    data,
    camera=camera,
)

rgb = renderer.render()

before_bgr = cv2.cvtColor(
    rgb,
    cv2.COLOR_RGB2BGR,
)

cv2.imwrite(
    str(OUTPUT_DIR / "before.png"),
    before_bgr,
)


# =========================================================
# 9. CNN preprocessing
# =========================================================

image = cv2.resize(
    rgb,
    (80, 60),
)

image = (
    image.astype(np.float32)
    / 255.0
)

image = np.transpose(
    image,
    (2, 0, 1),
)

image_tensor = torch.tensor(
    image,
    dtype=torch.float32,
)

# [3, 60, 80]
# ->
# [1, 3, 60, 80]

image_tensor = (
    image_tensor.unsqueeze(0)
)


# =========================================================
# 10. Vision inference
# =========================================================

with torch.no_grad():

    prediction = cnn(
        image_tensor
    )[0]

predicted_position = np.array(
    [
        prediction[0].item(),
        prediction[1].item(),
    ]
)

print()
print(
    "True ball XY:",
    true_ball_position[:2],
)

print(
    "CNN predicted XY:",
    predicted_position,
)

vision_error = np.linalg.norm(
    predicted_position
    - true_ball_position[:2]
)

print(
    "Vision error:",
    vision_error,
)


# =========================================================
# 11. Inverse kinematics
# =========================================================

L1 = 1.0
L2 = 0.8


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
        2.0
        * L1
        * L2
    )

    if not -1.0 <= cos_q2 <= 1.0:
        raise ValueError(
            "Target is outside workspace."
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


q_goal = inverse_kinematics(
    predicted_position[0],
    predicted_position[1],
)

print()
print(
    "IK goal [rad]:",
    q_goal,
)

print(
    "IK goal [deg]:",
    np.degrees(q_goal),
)


# =========================================================
# 12. Initial state for trajectory
# =========================================================

q_start = data.qpos.copy()

trajectory_duration = 4.0

simulation_duration = 6.0


# =========================================================
# 13. Quintic trajectory
# =========================================================

def quintic_trajectory(
    time,
    q_start,
    q_goal,
    duration,
):

    # After trajectory finishes,
    # simply hold the final goal.

    if time >= duration:

        return (
            q_goal.copy(),
            np.zeros(2),
            np.zeros(2),
        )

    s = time / duration

    # Position interpolation
    h = (
        10.0 * s**3
        - 15.0 * s**4
        + 6.0 * s**5
    )

    # First derivative
    dh = (
        30.0 * s**2
        - 60.0 * s**3
        + 30.0 * s**4
    ) / duration

    # Second derivative
    ddh = (
        60.0 * s
        - 180.0 * s**2
        + 120.0 * s**3
    ) / duration**2

    delta_q = (
        q_goal
        - q_start
    )

    q_des = (
        q_start
        + h * delta_q
    )

    dq_des = (
        dh * delta_q
    )

    ddq_des = (
        ddh * delta_q
    )

    return (
        q_des,
        dq_des,
        ddq_des,
    )


# =========================================================
# 14. Robot dynamics parameters
# =========================================================

m1 = 1.0
m2 = 0.8

lc1 = 0.5
lc2 = 0.4

I1 = 0.0833333333
I2 = 0.0426666667

gravity = 9.81


# =========================================================
# 15. Dynamics model
# =========================================================

def dynamics_terms(
    q,
    dq,
):

    q1 = q[0]
    q2 = q[1]

    dq1 = dq[0]
    dq2 = dq[1]

    cos_q2 = math.cos(q2)
    sin_q2 = math.sin(q2)

    # -------------------------
    # Mass matrix M(q)
    # -------------------------

    M11 = (
        I1
        + I2
        + m1 * lc1**2
        + m2
        * (
            L1**2
            + lc2**2
            + 2.0
            * L1
            * lc2
            * cos_q2
        )
    )

    M12 = (
        I2
        + m2
        * (
            lc2**2
            + L1
            * lc2
            * cos_q2
        )
    )

    M22 = (
        I2
        + m2 * lc2**2
    )

    M = np.array(
        [
            [M11, M12],
            [M12, M22],
        ]
    )

    # -------------------------
    # Coriolis / centrifugal
    # -------------------------

    h = (
        m2
        * L1
        * lc2
        * sin_q2
    )

    c1 = (
        -h
        * (
            2.0
            * dq1
            * dq2
            + dq2**2
        )
    )

    c2 = (
        h
        * dq1**2
    )

    c = np.array(
        [
            c1,
            c2,
        ]
    )

    # -------------------------
    # Gravity
    # -------------------------

    g1 = (
        m1
        * gravity
        * lc1
        * math.cos(q1)
        +
        m2
        * gravity
        * (
            L1
            * math.cos(q1)
            +
            lc2
            * math.cos(
                q1 + q2
            )
        )
    )

    g2 = (
        m2
        * gravity
        * lc2
        * math.cos(
            q1 + q2
        )
    )

    g = np.array(
        [
            g1,
            g2,
        ]
    )

    return M, c, g


# =========================================================
# 16. Computed Torque Control
# =========================================================

Kp = np.array(
    [25.0, 25.0]
)

Kd = np.array(
    [10.0, 10.0]
)


def computed_torque(
    q,
    dq,
    q_des,
    dq_des,
    ddq_des,
):

    error = (
        q_des
        - q
    )

    derror = (
        dq_des
        - dq
    )

    desired_acceleration = (
        ddq_des
        + Kd * derror
        + Kp * error
    )

    M, c, g = dynamics_terms(
        q,
        dq,
    )

    tau = (
        M @ desired_acceleration
        + c
        + g
    )

    return tau


# =========================================================
# 17. Control simulation
# =========================================================

num_steps = int(
    simulation_duration
    / dt
)

for step in range(num_steps):

    time = (
        step * dt
    )

    q = (
        data.qpos.copy()
    )

    dq = (
        data.qvel.copy()
    )

    (
        q_des,
        dq_des,
        ddq_des,
    ) = quintic_trajectory(
        time,
        q_start,
        q_goal,
        trajectory_duration,
    )

    tau = computed_torque(
        q,
        dq,
        q_des,
        dq_des,
        ddq_des,
    )

    data.qfrc_applied[:] = (
        tau
    )

    mujoco.mj_step(
        model,
        data,
    )


# =========================================================
# 18. Final state
# =========================================================

final_q = (
    data.qpos.copy()
)

final_dq = (
    data.qvel.copy()
)

print()
print(
    "Final q [rad]:",
    final_q,
)

print(
    "Goal q [rad]:",
    q_goal,
)

print(
    "Final joint error:",
    final_q - q_goal,
)


# =========================================================
# 19. End-effector position
# =========================================================

tip_site_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_SITE,
    "tip",
)

tip_position = (
    data.site_xpos[
        tip_site_id
    ].copy()
)

print()
print(
    "Final tip XY:",
    tip_position[:2],
)

print(
    "True ball XY:",
    true_ball_position[:2],
)

print(
    "Tip-to-ball XY distance:",
    np.linalg.norm(
        tip_position[:2]
        - true_ball_position[:2]
    ),
)


# =========================================================
# 20. Save final camera image
# =========================================================

renderer.update_scene(
    data,
    camera=camera,
)

final_rgb = (
    renderer.render()
)

final_bgr = cv2.cvtColor(
    final_rgb,
    cv2.COLOR_RGB2BGR,
)

cv2.imwrite(
    str(
        OUTPUT_DIR
        / "after.png"
    ),
    final_bgr,
)

print()
print(
    "Saved before/after images to:",
    OUTPUT_DIR,
)