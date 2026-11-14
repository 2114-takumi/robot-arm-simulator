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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = PROJECT_ROOT.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "two_link_arm.xml"
)

POLICY_PATH = (
    PYTHON_ROOT
    / "pytorch-learning"
    / "checkpoints"
    / "vision_bc_policy.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "step_13_b_3"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# 2. Vision BC Policy
# =========================================================

class VisionBCPolicy(nn.Module):

    def __init__(self):
        super().__init__()

        # Image branch
        self.conv1 = nn.Conv2d(
            3,
            16,
            kernel_size=3,
            padding=1,
        )

        self.conv2 = nn.Conv2d(
            16,
            32,
            kernel_size=3,
            padding=1,
        )

        self.conv3 = nn.Conv2d(
            32,
            64,
            kernel_size=3,
            padding=1,
        )

        self.relu = nn.ReLU()

        self.pool = nn.MaxPool2d(
            kernel_size=2
        )

        self.image_fc = nn.Linear(
            64 * 7 * 10,
            128,
        )

        # Proprioception branch
        self.proprio_fc1 = nn.Linear(
            4,
            32,
        )

        self.proprio_fc2 = nn.Linear(
            32,
            32,
        )

        # Fusion
        self.fusion_fc1 = nn.Linear(
            160,
            128,
        )

        self.output_layer = nn.Linear(
            128,
            2,
        )

    def forward(
        self,
        image,
        proprio,
    ):

        # -------------------------
        # Image
        # -------------------------

        x = self.conv1(image)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)

        x = torch.flatten(
            x,
            start_dim=1,
        )

        image_feature = (
            self.image_fc(x)
        )

        image_feature = self.relu(
            image_feature
        )

        # -------------------------
        # Proprioception
        # -------------------------

        p = self.proprio_fc1(
            proprio
        )

        p = self.relu(p)

        p = self.proprio_fc2(p)

        proprio_feature = self.relu(
            p
        )

        # -------------------------
        # Fusion
        # -------------------------

        combined = torch.cat(
            [
                image_feature,
                proprio_feature,
            ],
            dim=1,
        )

        x = self.fusion_fc1(
            combined
        )

        x = self.relu(x)

        tau = self.output_layer(
            x
        )

        return tau


# =========================================================
# 3. Load policy
# =========================================================

checkpoint = torch.load(
    POLICY_PATH,
    map_location="cpu",
    weights_only=True,
)

policy = VisionBCPolicy()

policy.load_state_dict(
    checkpoint["model_state_dict"]
)

policy.eval()


proprio_mean = (
    checkpoint["proprio_mean"]
    .numpy()
)

proprio_std = (
    checkpoint["proprio_std"]
    .numpy()
)

tau_mean = (
    checkpoint["tau_mean"]
    .numpy()
)

tau_std = (
    checkpoint["tau_std"]
    .numpy()
)


print("Vision BC policy loaded:")
print(POLICY_PATH)


# =========================================================
# 4. Robot geometry
# =========================================================

L1 = 1.0
L2 = 0.8


def forward_kinematics(q):

    q1 = q[0]
    q2 = q[1]

    x = (
        L1 * math.cos(q1)
        + L2 * math.cos(
            q1 + q2
        )
    )

    y = (
        L1 * math.sin(q1)
        + L2 * math.sin(
            q1 + q2
        )
    )

    return np.array(
        [
            x,
            y,
        ]
    )


# =========================================================
# 5. MuJoCo
# =========================================================

model = mujoco.MjModel.from_xml_path(
    str(MODEL_PATH)
)

data = mujoco.MjData(model)

dt = model.opt.timestep


# =========================================================
# 6. Renderer / camera
# =========================================================

renderer = mujoco.Renderer(
    model,
    height=480,
    width=640,
)

camera = mujoco.MjvCamera()

camera.lookat[:] = [
    0.5,
    0.0,
    0.0,
]

camera.distance = 4.0
camera.azimuth = 90
camera.elevation = -90


# =========================================================
# 7. Hidden test goal
# =========================================================
#
# IMPORTANT:
#
# q_goal is ONLY used to place the ball
# and evaluate the result.
#
# It is NEVER passed to the policy.
# =========================================================

q_goal = np.array(
    [
        0.30,
        0.85,
    ]
)

ball_xy = forward_kinematics(
    q_goal
)

ball_position = np.array(
    [
        ball_xy[0],
        ball_xy[1],
        0.08,
    ]
)

data.mocap_pos[0] = (
    ball_position
)


# =========================================================
# 8. Initial robot state
# =========================================================

q_start = np.array(
    [
        -0.35,
        0.25,
    ]
)

dq_start = np.array(
    [
        0.0,
        0.0,
    ]
)

data.qpos[:] = q_start
data.qvel[:] = dq_start

mujoco.mj_forward(
    model,
    data,
)


print()
print(
    "Hidden goal q:",
    q_goal,
)

print(
    "Ball XY:",
    ball_xy,
)

print(
    "Initial q:",
    q_start,
)


# =========================================================
# 9. Vision policy inference
# =========================================================

def vision_policy_action(
    rgb,
    q,
    dq,
):

    # ---------------------------------
    # Image preprocessing
    # ---------------------------------

    image = cv2.resize(
        rgb,
        (
            80,
            60,
        ),
        interpolation=cv2.INTER_AREA,
    )

    image = (
        image.astype(
            np.float32
        )
        / 255.0
    )

    # HWC -> CHW
    image = np.transpose(
        image,
        (
            2,
            0,
            1,
        )
    )

    # [3, 60, 80]
    # ->
    # [1, 3, 60, 80]

    image_tensor = torch.tensor(
        image,
        dtype=torch.float32,
    ).unsqueeze(0)


    # ---------------------------------
    # Proprioception
    # ---------------------------------

    proprio = np.concatenate(
        [
            q,
            dq,
        ]
    )

    proprio_normalized = (
        proprio
        - proprio_mean
    ) / proprio_std

    proprio_tensor = torch.tensor(
        proprio_normalized,
        dtype=torch.float32,
    ).unsqueeze(0)


    # ---------------------------------
    # Policy
    # ---------------------------------

    with torch.no_grad():

        normalized_tau = policy(
            image_tensor,
            proprio_tensor,
        )

    normalized_tau = (
        normalized_tau
        .numpy()[0]
    )


    # ---------------------------------
    # Denormalize torque
    # ---------------------------------

    tau = (
        normalized_tau
        * tau_std
        + tau_mean
    )

    return tau


# =========================================================
# 10. Save initial camera image
# =========================================================

renderer.update_scene(
    data,
    camera=camera,
)

initial_rgb = renderer.render()

initial_bgr = cv2.cvtColor(
    initial_rgb,
    cv2.COLOR_RGB2BGR,
)

cv2.imwrite(
    str(
        OUTPUT_DIR
        / "before.png"
    ),
    initial_bgr,
)


# =========================================================
# 11. Closed-loop rollout
# =========================================================

simulation_duration = 4.0

num_steps = int(
    simulation_duration
    / dt
)


q_history = []
dq_history = []
tau_history = []
joint_error_history = []
tip_error_history = []


for step in range(
    num_steps
):

    # =================================
    # Current robot state
    # =================================

    q = data.qpos.copy()
    dq = data.qvel.copy()


    # =================================
    # Current camera image
    # =================================

    renderer.update_scene(
        data,
        camera=camera,
    )

    rgb = renderer.render()


    # =================================
    # Vision + proprioception -> torque
    #
    # NO IK
    # NO CTC
    # NO q_goal input
    # =================================

    tau = vision_policy_action(
        rgb,
        q,
        dq,
    )


    # =================================
    # Apply NN torque
    # =================================

    data.qfrc_applied[:] = tau

    mujoco.mj_step(
        model,
        data,
    )


    # =================================
    # Evaluation only
    # =================================

    tip_xy = forward_kinematics(
        q
    )

    q_history.append(
        q.copy()
    )

    dq_history.append(
        dq.copy()
    )

    tau_history.append(
        tau.copy()
    )

    joint_error_history.append(
        q - q_goal
    )

    tip_error_history.append(
        np.linalg.norm(
            tip_xy
            - ball_xy
        )
    )


# =========================================================
# 12. Convert logs
# =========================================================

q_history = np.array(
    q_history
)

dq_history = np.array(
    dq_history
)

tau_history = np.array(
    tau_history
)

joint_error_history = np.array(
    joint_error_history
)

tip_error_history = np.array(
    tip_error_history
)


# =========================================================
# 13. Final state
# =========================================================

final_q = data.qpos.copy()
final_dq = data.qvel.copy()

final_tip_xy = forward_kinematics(
    final_q
)

final_joint_error = (
    final_q
    - q_goal
)

final_tip_error = np.linalg.norm(
    final_tip_xy
    - ball_xy
)


# =========================================================
# 14. Last 0.5 seconds
# =========================================================

last_duration = 0.5

last_steps = int(
    last_duration
    / dt
)

last_joint_errors = (
    joint_error_history[
        -last_steps:
    ]
)

joint_rmse = np.sqrt(
    np.mean(
        last_joint_errors**2,
        axis=0,
    )
)

mean_tip_error = np.mean(
    tip_error_history[
        -last_steps:
    ]
)


# =========================================================
# 15. Torque statistics
# =========================================================

max_abs_tau = np.max(
    np.abs(
        tau_history
    ),
    axis=0,
)


# =========================================================
# 16. Save final image
# =========================================================

renderer.update_scene(
    data,
    camera=camera,
)

final_rgb = renderer.render()

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


# =========================================================
# 17. Results
# =========================================================

print()
print(
    "=========================="
)

print(
    "Vision BC Closed-loop"
)

print(
    "=========================="
)

print()
print(
    "Goal q:",
    q_goal,
)

print(
    "Final q:",
    final_q,
)

print()
print(
    "Final joint error [rad]:",
    final_joint_error,
)

print(
    "Final joint error [deg]:",
    np.degrees(
        final_joint_error
    ),
)

print()
print(
    "Final dq:",
    final_dq,
)

print()
print(
    "Final tip XY:",
    final_tip_xy,
)

print(
    "Ball XY:",
    ball_xy,
)

print(
    "Final tip-to-ball error [m]:",
    final_tip_error,
)

print()
print(
    "Last 0.5 s joint RMSE:",
    joint_rmse,
)

print(
    "Last 0.5 s mean tip error [m]:",
    mean_tip_error,
)

print()
print(
    "Max absolute torque:",
    max_abs_tau,
)

print()
print(
    "Before image:",
    OUTPUT_DIR / "before.png",
)

print(
    "After image:",
    OUTPUT_DIR / "after.png",
)


renderer.close()