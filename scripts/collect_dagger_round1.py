from pathlib import Path
import csv
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
    / "step_13_b_5_dagger_round1"
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


# =========================================================
# 2. Random generator
# =========================================================

rng = np.random.default_rng(
    seed=456
)


# =========================================================
# 3. Student Vision Policy
# =========================================================

class VisionBCPolicy(nn.Module):

    def __init__(self):
        super().__init__()

        # -------------------------
        # Image branch
        # -------------------------

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            padding=1,
        )

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1,
        )

        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1,
        )

        self.relu = nn.ReLU()

        self.pool = nn.MaxPool2d(
            kernel_size=2
        )

        # 60x80
        # -> 30x40
        # -> 15x20
        # -> 7x10
        #
        # 64 * 7 * 10 = 4480

        self.image_fc = nn.Linear(
            64 * 7 * 10,
            128,
        )


        # -------------------------
        # Robot-state branch
        # -------------------------

        # q1, q2, dq1, dq2

        self.proprio_fc1 = nn.Linear(
            4,
            32,
        )

        self.proprio_fc2 = nn.Linear(
            32,
            32,
        )


        # -------------------------
        # Fusion
        # -------------------------

        # image feature:
        # 128
        #
        # robot-state feature:
        # 32
        #
        # total:
        # 160

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

        # =================================
        # Image branch
        # =================================

        x = self.conv1(image)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)

        # [batch, 64, 7, 10]
        # ->
        # [batch, 4480]

        x = torch.flatten(
            x,
            start_dim=1,
        )

        image_feature = self.image_fc(x)

        image_feature = self.relu(
            image_feature
        )


        # =================================
        # Robot-state branch
        # =================================

        p = self.proprio_fc1(
            proprio
        )

        p = self.relu(p)

        p = self.proprio_fc2(p)

        proprio_feature = self.relu(
            p
        )


        # =================================
        # Combine image + q,dq
        # =================================

        combined = torch.cat(
            [
                image_feature,
                proprio_feature,
            ],
            dim=1,
        )


        # =================================
        # Predict torque
        # =================================

        x = self.fusion_fc1(
            combined
        )

        x = self.relu(x)

        tau = self.output_layer(
            x
        )

        return tau


# =========================================================
# 4. Load trained Student
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


# Normalization parameters
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


print(
    "Vision BC policy loaded:"
)

print(
    POLICY_PATH
)


# =========================================================
# 5. Robot parameters
# =========================================================

L1 = 1.0
L2 = 0.8

m1 = 1.0
m2 = 0.8

lc1 = 0.5
lc2 = 0.4

I1 = 0.0833333333
I2 = 0.0426666667

gravity = 9.81


# CTC gains

Kp = np.array(
    [
        25.0,
        25.0,
    ]
)

Kd = np.array(
    [
        10.0,
        10.0,
    ]
)


# =========================================================
# 6. Forward kinematics
# =========================================================

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
# 7. Dynamics
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
        + m2
        * lc2**2
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
# 8. Expert CTC
# =========================================================

def expert_ctc(
    q,
    dq,
    q_goal,
):

    # Position error
    position_error = (
        q_goal - q
    )

    # Target velocity is 0
    velocity_error = (
        -dq
    )

    # Desired acceleration
    desired_acceleration = (
        Kp * position_error
        + Kd * velocity_error
    )

    # Robot dynamics
    M, c, g = dynamics_terms(
        q,
        dq,
    )

    # Computed Torque Control
    tau = (
        M @ desired_acceleration
        + c
        + g
    )

    return tau


# =========================================================
# 9. MuJoCo
# =========================================================

model = mujoco.MjModel.from_xml_path(
    str(MODEL_PATH)
)

data = mujoco.MjData(
    model
)

dt = model.opt.timestep

print(
    "Physics timestep:",
    dt,
)


# =========================================================
# 10. Camera
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
# 11. Student inference
# =========================================================

def student_action(
    rgb,
    q,
    dq,
):

    # =================================
    # Image preprocessing
    # =================================

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

    # HWC
    # ->
    # CHW

    image = np.transpose(
        image,
        (
            2,
            0,
            1,
        )
    )

    # [3,60,80]
    # ->
    # [1,3,60,80]

    image_tensor = torch.tensor(
        image,
        dtype=torch.float32,
    ).unsqueeze(0)


    # =================================
    # q,dq preprocessing
    # =================================

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


    # =================================
    # Student predicts action
    # =================================

    with torch.no_grad():

        normalized_tau = policy(
            image_tensor,
            proprio_tensor,
        ).numpy()[0]


    # =================================
    # Back to Nm
    # =================================

    tau = (
        normalized_tau
        * tau_std
        + tau_mean
    )

    return tau


# =========================================================
# 12. DAgger settings
# =========================================================

num_episodes = 50

episode_duration = 4.0

steps_per_episode = int(
    episode_duration
    / dt
)


# Physics:
# 500 Hz
#
# Save every 20 steps:
# 25 Hz
#
# 4 sec:
# 100 samples / episode
#
# 50 episodes:
# 5000 samples

save_every = 20


rows = []

sample_index = 0

action_errors = []

success_count = 0


# =========================================================
# 13. DAgger rollout
# =========================================================

for episode in range(
    num_episodes
):

    # =================================
    # Reset simulation
    # =================================

    mujoco.mj_resetData(
        model,
        data,
    )


    # =================================
    # Random initial joint state
    # =================================

    q_start = rng.uniform(
        low=[
            -0.7,
            -0.8,
        ],
        high=[
            0.7,
            0.8,
        ],
    )


    dq_start = rng.uniform(
        low=-0.3,
        high=0.3,
        size=2,
    )


    # =================================
    # Random hidden goal
    # =================================

    q_goal = np.array(
        [
            rng.uniform(
                -0.8,
                0.8,
            ),

            rng.uniform(
                0.2,
                1.2,
            ),
        ]
    )


    # Convert q_goal
    # ->
    # red ball position

    ball_xy = forward_kinematics(
        q_goal
    )


    # Put ball in simulator

    data.mocap_pos[0] = np.array(
        [
            ball_xy[0],
            ball_xy[1],
            0.08,
        ]
    )


    # Robot initial state

    data.qpos[:] = q_start
    data.qvel[:] = dq_start


    mujoco.mj_forward(
        model,
        data,
    )


    # =================================
    # Student controls for 4 seconds
    # =================================

    for step in range(
        steps_per_episode
    ):

        # -----------------------------
        # Current robot state
        # -----------------------------

        q = data.qpos.copy()
        dq = data.qvel.copy()


        # -----------------------------
        # Current camera image
        # -----------------------------

        renderer.update_scene(
            data,
            camera=camera,
        )

        rgb = renderer.render()


        # =================================
        # STUDENT ACTION
        #
        # image + q,dq
        # ->
        # predicted torque
        # =================================

        tau_student = student_action(
            rgb,
            q,
            dq,
        )


        # =================================
        # Save DAgger sample every 20 steps
        # =================================

        if step % save_every == 0:

            # -----------------------------
            # Ask teacher:
            #
            # "At THIS state,
            # what torque should be used?"
            # -----------------------------

            tau_expert = expert_ctc(
                q,
                dq,
                q_goal,
            )


            # =================================
            # Save Student's current image
            # =================================

            image_small = cv2.resize(
                rgb,
                (
                    80,
                    60,
                ),
                interpolation=cv2.INTER_AREA,
            )

            image_bgr = cv2.cvtColor(
                image_small,
                cv2.COLOR_RGB2BGR,
            )


            filename = (
                f"dagger_r1_"
                f"{sample_index:06d}"
                f".png"
            )

            image_path = (
                IMAGE_DIR
                / filename
            )

            cv2.imwrite(
                str(image_path),
                image_bgr,
            )


            # =================================
            # Save training sample
            #
            # INPUT:
            # image + q,dq
            #
            # LABEL:
            # tau_expert
            #
            # tau_student is debug only
            # =================================

            rows.append(
                [
                    episode,
                    step * dt,
                    filename,

                    q[0],
                    q[1],

                    dq[0],
                    dq[1],

                    # Expert label
                    tau_expert[0],
                    tau_expert[1],

                    # Debug metadata
                    ball_xy[0],
                    ball_xy[1],

                    q_goal[0],
                    q_goal[1],

                    # Student prediction
                    tau_student[0],
                    tau_student[1],
                ]
            )


            # Student vs Teacher difference

            action_errors.append(
                tau_student
                - tau_expert
            )


            sample_index += 1


        # =================================
        # IMPORTANT:
        #
        # Only STUDENT controls MuJoCo.
        #
        # Teacher is NOT used here.
        # =================================

        data.qfrc_applied[:] = (
            tau_student
        )

        mujoco.mj_step(
            model,
            data,
        )


    # =====================================================
    # Episode evaluation
    # =====================================================

    final_q = data.qpos.copy()
    final_dq = data.qvel.copy()


    final_tip = forward_kinematics(
        final_q
    )


    tip_error = np.linalg.norm(
        final_tip
        - ball_xy
    )


    # Success:
    #
    # tip within 3 cm
    #
    # AND
    #
    # robot nearly stopped

    success = (
        tip_error <= 0.03
        and np.all(
            np.abs(final_dq)
            <= 0.05
        )
    )


    if success:

        success_count += 1


    print(
        f"episode={episode:2d}, "
        f"success={success}, "
        f"tip_error="
        f"{tip_error:.4f} m, "
        f"samples="
        f"{sample_index}"
    )


# =========================================================
# 14. Save labels.csv
# =========================================================

header = [
    "episode",
    "time",
    "filename",

    "q1",
    "q2",

    "dq1",
    "dq2",

    # Expert labels
    "tau1",
    "tau2",

    # Debug
    "ball_x",
    "ball_y",

    "goal_q1",
    "goal_q2",

    # Student prediction
    "student_tau1",
    "student_tau2",
]


with LABEL_PATH.open(
    "w",
    newline="",
    encoding="utf-8",
) as file:

    writer = csv.writer(
        file
    )

    writer.writerow(
        header
    )

    writer.writerows(
        rows
    )


# =========================================================
# 15. Statistics
# =========================================================

action_errors = np.array(
    action_errors
)


action_rmse = np.sqrt(
    np.mean(
        action_errors**2,
        axis=0,
    )
)


print()
print(
    "=========================="
)

print(
    "DAgger Round 1"
)

print(
    "=========================="
)

print(
    "Episodes:",
    num_episodes,
)

print(
    "Student success rate:",
    success_count
    / num_episodes,
)

print(
    "Collected samples:",
    sample_index,
)

print()

print(
    "Student vs Expert "
    "action RMSE:"
)

print(
    action_rmse
)

print()

print(
    "Images:",
    IMAGE_DIR,
)

print(
    "Labels:",
    LABEL_PATH,
)


renderer.close()