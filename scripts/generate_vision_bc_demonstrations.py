from pathlib import Path
import csv
import math

import cv2
import mujoco
import numpy as np


# =========================================================
# 1. Paths
# =========================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "two_link_arm.xml"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "step_13_b_1"
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
    seed=0
)


# =========================================================
# 3. MuJoCo
# =========================================================

model = mujoco.MjModel.from_xml_path(
    str(MODEL_PATH)
)

data = mujoco.MjData(model)

dt = model.opt.timestep

print(
    "Physics timestep:",
    dt,
)


# =========================================================
# 4. Renderer
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


# =========================================================
# 6. CTC gains
# =========================================================

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
# 7. Forward kinematics
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
# 8. Dynamics
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
    # Mass matrix
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
    # Coriolis
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
# 9. Expert CTC
# =========================================================

def expert_ctc(
    q,
    dq,
    q_goal,
):

    position_error = (
        q_goal - q
    )

    velocity_error = (
        -dq
    )

    desired_acceleration = (
        Kp * position_error
        + Kd * velocity_error
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
# 10. Dataset settings
# =========================================================

num_episodes = 100

episode_duration = 2.0

steps_per_episode = int(
    episode_duration
    / dt
)

# Physics:
# 500 Hz
#
# Image saving:
# every 20 steps
# -> 25 Hz
#
# 2 seconds:
# -> 50 images / episode
#
# 100 episodes:
# -> 5000 images

save_every = 20


# =========================================================
# 11. Dataset rows
# =========================================================

rows = []

sample_index = 0


# =========================================================
# 12. Generate demonstrations
# =========================================================

for episode in range(
    num_episodes
):

    # ---------------------------------
    # Reset
    # ---------------------------------

    mujoco.mj_resetData(
        model,
        data,
    )

    # ---------------------------------
    # Random initial robot state
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
    # Random target joint pose
    # ---------------------------------
    #
    # q2 is kept positive.
    #
    # This prevents the same ball
    # position from corresponding to
    # both elbow-up and elbow-down
    # solutions.
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

    # ---------------------------------
    # Convert goal pose -> ball XY
    # ---------------------------------

    ball_xy = forward_kinematics(
        q_goal
    )

    ball_x = ball_xy[0]
    ball_y = ball_xy[1]

    # ---------------------------------
    # Put red ball at goal
    # ---------------------------------

    data.mocap_pos[0] = np.array(
        [
            ball_x,
            ball_y,
            0.08,
        ]
    )

    # ---------------------------------
    # Robot initial state
    # ---------------------------------

    data.qpos[:] = q_start
    data.qvel[:] = dq_start

    mujoco.mj_forward(
        model,
        data,
    )

    # ---------------------------------
    # Expert rollout
    # ---------------------------------

    for step in range(
        steps_per_episode
    ):

        q = data.qpos.copy()
        dq = data.qvel.copy()

        # Expert action
        tau = expert_ctc(
            q,
            dq,
            q_goal,
        )

        # =================================
        # Save demonstration
        # =================================

        if step % save_every == 0:

            # -----------------------------
            # Render camera image
            # -----------------------------

            renderer.update_scene(
                data,
                camera=camera,
            )

            rgb = renderer.render()

            # -----------------------------
            # Downsample exactly as we used
            # in the previous CNN work
            # -----------------------------

            image_small = cv2.resize(
                rgb,
                (
                    80,
                    60,
                ),
                interpolation=cv2.INTER_AREA,
            )

            # RGB -> BGR for OpenCV save
            image_bgr = cv2.cvtColor(
                image_small,
                cv2.COLOR_RGB2BGR,
            )

            filename = (
                f"sample_"
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

            # -----------------------------
            # Save label
            # -----------------------------

            rows.append(
                [
                    episode,
                    step * dt,
                    filename,

                    q[0],
                    q[1],

                    dq[0],
                    dq[1],

                    tau[0],
                    tau[1],

                    # Debug metadata only.
                    # Student policy will NOT
                    # receive these values.
                    ball_x,
                    ball_y,

                    q_goal[0],
                    q_goal[1],
                ]
            )

            sample_index += 1

        # ---------------------------------
        # Expert controls robot
        # ---------------------------------

        data.qfrc_applied[:] = tau

        mujoco.mj_step(
            model,
            data,
        )

    print(
        f"episode="
        f"{episode:3d}, "
        f"samples="
        f"{sample_index}"
    )


# =========================================================
# 13. Save labels.csv
# =========================================================

header = [
    "episode",
    "time",
    "filename",

    "q1",
    "q2",

    "dq1",
    "dq2",

    "tau1",
    "tau2",

    # Debug only
    "ball_x",
    "ball_y",
    "goal_q1",
    "goal_q2",
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
# 14. Summary
# =========================================================

print()
print(
    "=========================="
)

print(
    "Vision BC Dataset"
)

print(
    "=========================="
)

print(
    "Episodes:",
    num_episodes,
)

print(
    "Samples:",
    sample_index,
)

print(
    "Images:",
    IMAGE_DIR,
)

print(
    "Labels:",
    LABEL_PATH,
)


renderer.close()