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
    / "vision_dagger_r1_policy.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "step_13_b_4"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CSV_PATH = (
    OUTPUT_DIR
    / "vision_bc_generalization.csv"
)


# =========================================================
# 2. Reproducibility
# =========================================================

rng = np.random.default_rng(
    seed=123
)


# =========================================================
# 3. Vision BC Policy
# =========================================================

class VisionBCPolicy(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            3, 16, 3, padding=1
        )

        self.conv2 = nn.Conv2d(
            16, 32, 3, padding=1
        )

        self.conv3 = nn.Conv2d(
            32, 64, 3, padding=1
        )

        self.relu = nn.ReLU()

        self.pool = nn.MaxPool2d(
            2
        )

        self.image_fc = nn.Linear(
            64 * 7 * 10,
            128,
        )

        self.proprio_fc1 = nn.Linear(
            4,
            32,
        )

        self.proprio_fc2 = nn.Linear(
            32,
            32,
        )

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

        x = self.pool(
            self.relu(
                self.conv1(image)
            )
        )

        x = self.pool(
            self.relu(
                self.conv2(x)
            )
        )

        x = self.pool(
            self.relu(
                self.conv3(x)
            )
        )

        x = torch.flatten(
            x,
            start_dim=1,
        )

        image_feature = self.relu(
            self.image_fc(x)
        )


        p = self.relu(
            self.proprio_fc1(
                proprio
            )
        )

        proprio_feature = self.relu(
            self.proprio_fc2(p)
        )


        combined = torch.cat(
            [
                image_feature,
                proprio_feature,
            ],
            dim=1,
        )

        x = self.relu(
            self.fusion_fc1(
                combined
            )
        )

        return self.output_layer(x)


# =========================================================
# 4. Load policy
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

Kp = np.array(
    [25.0, 25.0]
)

Kd = np.array(
    [10.0, 10.0]
)


# =========================================================
# 6. Forward kinematics
# =========================================================

def forward_kinematics(q):

    q1, q2 = q

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
        [x, y]
    )


# =========================================================
# 7. Robot dynamics
# =========================================================

def dynamics_terms(
    q,
    dq,
):

    q1, q2 = q
    dq1, dq2 = dq

    cos_q2 = math.cos(q2)
    sin_q2 = math.sin(q2)

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


    h = (
        m2
        * L1
        * lc2
        * sin_q2
    )

    c = np.array(
        [
            -h
            * (
                2.0
                * dq1
                * dq2
                + dq2**2
            ),

            h * dq1**2,
        ]
    )


    g = np.array(
        [
            (
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
            ),

            (
                m2
                * gravity
                * lc2
                * math.cos(
                    q1 + q2
                )
            ),
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

    desired_acceleration = (
        Kp
        * (
            q_goal - q
        )
        -
        Kd * dq
    )

    M, c, g = dynamics_terms(
        q,
        dq,
    )

    return (
        M
        @ desired_acceleration
        + c
        + g
    )


# =========================================================
# 9. MuJoCo / Camera
# =========================================================

model = mujoco.MjModel.from_xml_path(
    str(MODEL_PATH)
)

data = mujoco.MjData(
    model
)

dt = model.opt.timestep


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
# 10. Vision Policy inference
# =========================================================

def vision_policy_action(
    rgb,
    q,
    dq,
):

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

    image = np.transpose(
        image,
        (
            2,
            0,
            1,
        )
    )

    image_tensor = torch.tensor(
        image,
        dtype=torch.float32,
    ).unsqueeze(0)


    proprio = np.concatenate(
        [
            q,
            dq,
        ]
    )

    proprio = (
        proprio
        - proprio_mean
    ) / proprio_std


    proprio_tensor = torch.tensor(
        proprio,
        dtype=torch.float32,
    ).unsqueeze(0)


    with torch.no_grad():

        tau_normalized = policy(
            image_tensor,
            proprio_tensor,
        ).numpy()[0]


    tau = (
        tau_normalized
        * tau_std
        + tau_mean
    )

    return tau


# =========================================================
# 11. Evaluation settings
# =========================================================

num_episodes = 50

simulation_duration = 4.0

num_steps = int(
    simulation_duration
    / dt
)


# Physical goal:
# arm tip within 3 cm of ball

tip_threshold = 0.03


# Robot should also stop.

velocity_threshold = 0.05


results = []


# =========================================================
# 12. Generalization evaluation
# =========================================================

for episode in range(
    num_episodes
):

    mujoco.mj_resetData(
        model,
        data,
    )


    # ---------------------------------
    # Random initial robot state
    # Same range as training dataset
    # ---------------------------------

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


    # ---------------------------------
    # Random hidden goal
    # ---------------------------------

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


    ball_xy = forward_kinematics(
        q_goal
    )


    data.mocap_pos[0] = np.array(
        [
            ball_xy[0],
            ball_xy[1],
            0.08,
        ]
    )


    data.qpos[:] = q_start
    data.qvel[:] = dq_start

    mujoco.mj_forward(
        model,
        data,
    )


    # ---------------------------------
    # Logs
    # ---------------------------------

    policy_taus = []
    expert_taus = []


    # =================================
    # Closed-loop rollout
    # =================================

    for step in range(
        num_steps
    ):

        q = data.qpos.copy()
        dq = data.qvel.copy()


        # -----------------------------
        # Current camera observation
        # -----------------------------

        renderer.update_scene(
            data,
            camera=camera,
        )

        rgb = renderer.render()


        # -----------------------------
        # Student action
        # -----------------------------

        tau_policy = (
            vision_policy_action(
                rgb,
                q,
                dq,
            )
        )


        # -----------------------------
        # Expert action
        #
        # Evaluation only!
        #
        # This is NOT given to Policy.
        # -----------------------------

        tau_expert = expert_ctc(
            q,
            dq,
            q_goal,
        )


        policy_taus.append(
            tau_policy.copy()
        )

        expert_taus.append(
            tau_expert.copy()
        )


        # -----------------------------
        # ONLY student controls robot
        # -----------------------------

        data.qfrc_applied[:] = (
            tau_policy
        )

        mujoco.mj_step(
            model,
            data,
        )


    # =====================================================
    # Episode metrics
    # =====================================================

    policy_taus = np.array(
        policy_taus
    )

    expert_taus = np.array(
        expert_taus
    )


    final_q = data.qpos.copy()
    final_dq = data.qvel.copy()

    final_tip = forward_kinematics(
        final_q
    )

    tip_error = np.linalg.norm(
        final_tip
        - ball_xy
    )

    joint_error = (
        final_q
        - q_goal
    )


    # ---------------------------------
    # How different was student's action
    # from CTC on EXACTLY SAME states?
    # ---------------------------------

    action_error = (
        policy_taus
        - expert_taus
    )

    action_rmse = np.sqrt(
        np.mean(
            action_error**2,
            axis=0,
        )
    )

    max_action_error = np.max(
        np.abs(
            action_error
        ),
        axis=0,
    )


    max_policy_tau = np.max(
        np.abs(
            policy_taus
        ),
        axis=0,
    )

    max_expert_tau = np.max(
        np.abs(
            expert_taus
        ),
        axis=0,
    )


    # ---------------------------------
    # Success
    # ---------------------------------

    tip_ok = (
        tip_error
        <= tip_threshold
    )

    velocity_ok = np.all(
        np.abs(
            final_dq
        )
        <= velocity_threshold
    )

    success = (
        tip_ok
        and velocity_ok
    )


    results.append(
        {
            "episode":
                episode,

            "tip_error":
                tip_error,

            "joint_error_1":
                joint_error[0],

            "joint_error_2":
                joint_error[1],

            "final_dq_1":
                final_dq[0],

            "final_dq_2":
                final_dq[1],

            "action_rmse_1":
                action_rmse[0],

            "action_rmse_2":
                action_rmse[1],

            "max_action_error_1":
                max_action_error[0],

            "max_action_error_2":
                max_action_error[1],

            "max_policy_tau_1":
                max_policy_tau[0],

            "max_policy_tau_2":
                max_policy_tau[1],

            "max_expert_tau_1":
                max_expert_tau[0],

            "max_expert_tau_2":
                max_expert_tau[1],

            "success":
                int(success),
        }
    )


    print(
        f"episode={episode:2d}, "
        f"success={success}, "
        f"tip_error="
        f"{tip_error:.4f} m, "
        f"action_rmse="
        f"{action_rmse}"
    )


# =========================================================
# 13. Aggregate
# =========================================================

success_rate = np.mean(
    [
        result["success"]
        for result in results
    ]
)


tip_errors = np.array(
    [
        result["tip_error"]
        for result in results
    ]
)


action_rmses = np.array(
    [
        [
            result["action_rmse_1"],
            result["action_rmse_2"],
        ]
        for result in results
    ]
)


policy_max_taus = np.array(
    [
        [
            result["max_policy_tau_1"],
            result["max_policy_tau_2"],
        ]
        for result in results
    ]
)


expert_max_taus = np.array(
    [
        [
            result["max_expert_tau_1"],
            result["max_expert_tau_2"],
        ]
        for result in results
    ]
)


# =========================================================
# 14. Summary
# =========================================================

print()
print(
    "=============================="
)

print(
    "Vision BC Generalization"
)

print(
    "=============================="
)

print(
    "Episodes:",
    num_episodes,
)

print(
    "Success rate:",
    success_rate,
)

print()
print(
    "Mean final tip error [m]:",
    tip_errors.mean(),
)

print(
    "Median final tip error [m]:",
    np.median(
        tip_errors
    ),
)

print(
    "95th percentile tip error [m]:",
    np.percentile(
        tip_errors,
        95,
    ),
)

print()
print(
    "Mean action RMSE "
    "[Policy vs CTC]:"
)

print(
    action_rmses.mean(
        axis=0
    )
)

print()
print(
    "Largest Policy torque:"
)

print(
    policy_max_taus.max(
        axis=0
    )
)

print()
print(
    "Largest Expert torque "
    "on same states:"
)

print(
    expert_max_taus.max(
        axis=0
    )
)


# =========================================================
# 15. Save CSV
# =========================================================

with CSV_PATH.open(
    "w",
    newline="",
    encoding="utf-8",
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=results[0].keys(),
    )

    writer.writeheader()

    writer.writerows(
        results
    )


print()
print(
    "Saved:",
    CSV_PATH,
)


renderer.close()