from pathlib import Path
import csv
import math

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
    / "step_13_1"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "ctc_demonstrations.csv"
)


# =========================================================
# 2. Reproducibility
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
# 4. Robot parameters
# =========================================================

L1 = 1.0

m1 = 1.0
m2 = 0.8

lc1 = 0.5
lc2 = 0.4

I1 = 0.0833333333
I2 = 0.0426666667

gravity = 9.81


# =========================================================
# 5. CTC gains
# =========================================================

Kp = np.array(
    [25.0, 25.0]
)

Kd = np.array(
    [10.0, 10.0]
)


# =========================================================
# 6. Dynamics model
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
# 7. Expert CTC
# =========================================================

def expert_ctc(
    q,
    dq,
    q_goal,
):

    # Goal is a stationary pose.
    #
    # dq_goal = 0
    # ddq_goal = 0

    position_error = (
        q_goal
        - q
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
# 8. Dataset settings
# =========================================================

num_episodes = 100

episode_duration = 2.0

steps_per_episode = int(
    episode_duration
    / dt
)

# MuJoCo runs at 500 Hz.
#
# Saving every 10 steps:
# 500 Hz / 10 = 50 samples/s

save_every = 10

rows = []


# =========================================================
# 9. Generate demonstrations
# =========================================================

for episode in range(
    num_episodes
):

    # Reset simulation.
    mujoco.mj_resetData(
        model,
        data,
    )

    # -------------------------
    # Random initial state
    # -------------------------

    q_start = rng.uniform(
        low=[
            -0.7,
            -1.0,
        ],
        high=[
            0.7,
            1.0,
        ],
    )

    dq_start = rng.uniform(
        low=-0.3,
        high=0.3,
        size=2,
    )

    # -------------------------
    # Random goal
    # -------------------------

    q_goal = (
        q_start
        + rng.uniform(
            low=-0.6,
            high=0.6,
            size=2,
        )
    )

    # Keep goals inside a
    # moderate operating region.

    q_goal = np.clip(
        q_goal,
        [
            -1.2,
            -1.5,
        ],
        [
            1.2,
            1.5,
        ],
    )

    data.qpos[:] = q_start
    data.qvel[:] = dq_start

    mujoco.mj_forward(
        model,
        data,
    )

    # -------------------------
    # Rollout expert
    # -------------------------

    for step in range(
        steps_per_episode
    ):

        time = step * dt

        q = data.qpos.copy()
        dq = data.qvel.copy()

        tau = expert_ctc(
            q,
            dq,
            q_goal,
        )

        # Save state-action pair.
        if step % save_every == 0:

            rows.append(
                [
                    episode,
                    time,

                    q[0],
                    q[1],

                    dq[0],
                    dq[1],

                    q_goal[0],
                    q_goal[1],

                    tau[0],
                    tau[1],
                ]
            )

        # Expert acts on robot.
        data.qfrc_applied[:] = tau

        mujoco.mj_step(
            model,
            data,
        )

    if (
        episode % 10 == 0
    ):
        print(
            f"episode="
            f"{episode:3d}, "
            f"samples="
            f"{len(rows)}"
        )


# =========================================================
# 10. Save CSV
# =========================================================

header = [
    "episode",
    "time",

    "q1",
    "q2",

    "dq1",
    "dq2",

    "goal_q1",
    "goal_q2",

    "tau1",
    "tau2",
]


with OUTPUT_PATH.open(
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
# 11. Summary
# =========================================================

print()
print(
    "Saved:",
    OUTPUT_PATH,
)

print(
    "Episodes:",
    num_episodes,
)

print(
    "Total samples:",
    len(rows),
)