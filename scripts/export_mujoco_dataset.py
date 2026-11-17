from pathlib import Path
import csv

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
    / "step_10_6"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "mujoco_dataset.csv"
)


# -------------------------
# 2. Load MuJoCo model
# -------------------------

model = mujoco.MjModel.from_xml_path(
    str(MODEL_PATH)
)

data = mujoco.MjData(model)

dt = model.opt.timestep

print("Simulation timestep:", dt)


# -------------------------
# 3. Initial state
# -------------------------

data.qpos[:] = np.array(
    [0.2, -0.3]
)

data.qvel[:] = np.array(
    [0.0, 0.0]
)

mujoco.mj_forward(
    model,
    data,
)


# -------------------------
# 4. Controller parameters
# -------------------------

kp = np.array(
    [20.0, 15.0]
)

kd = np.array(
    [5.0, 4.0]
)


# -------------------------
# 5. Simulation settings
# -------------------------

simulation_time = 10.0

num_steps = int(
    simulation_time / dt
)

rows = []


# -------------------------
# 6. Simulation loop
# -------------------------

for step in range(num_steps):

    time = step * dt

    # Moving target
    q_des = np.array(
        [
            0.5 * np.sin(0.8 * time),
            0.4 * np.sin(1.1 * time),
        ]
    )

    dq_des = np.array(
        [
            0.5 * 0.8
            * np.cos(0.8 * time),

            0.4 * 1.1
            * np.cos(1.1 * time),
        ]
    )

    # Current state
    q = data.qpos.copy()
    dq = data.qvel.copy()

    # Simple PD torque
    tau = (
        kp * (q_des - q)
        + kd * (dq_des - dq)
    )

    # Apply torque
    data.qfrc_applied[:] = tau

    # Advance simulation by one physics step
    mujoco.mj_step(
        model,
        data,
    )

    # Next state
    q_next = data.qpos.copy()
    dq_next = data.qvel.copy()

    rows.append(
        [
            time,

            q[0],
            q[1],

            dq[0],
            dq[1],

            tau[0],
            tau[1],

            q_next[0],
            q_next[1],

            dq_next[0],
            dq_next[1],
        ]
    )


# -------------------------
# 7. Save CSV
# -------------------------

header = [
    "time",

    "q1",
    "q2",

    "dq1",
    "dq2",

    "tau1",
    "tau2",

    "q1_next",
    "q2_next",

    "dq1_next",
    "dq2_next",
]


with OUTPUT_PATH.open(
    "w",
    newline="",
    encoding="utf-8",
) as file:

    writer = csv.writer(file)

    writer.writerow(header)

    writer.writerows(rows)


print()
print(
    "Saved dataset:",
    OUTPUT_PATH,
)

print(
    "Number of samples:",
    len(rows),
)