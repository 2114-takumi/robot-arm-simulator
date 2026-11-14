from pathlib import Path
import csv

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

POLICY_PATH = (
    PYTHON_ROOT
    / "pytorch-learning"
    / "checkpoints"
    / "bc_ctc_policy.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "step_13_4"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CSV_PATH = (
    OUTPUT_DIR
    / "bc_generalization_results.csv"
)


# =========================================================
# 2. Reproducibility
# =========================================================

rng = np.random.default_rng(
    seed=123
)


# =========================================================
# 3. Policy Network
# =========================================================

class BCPolicy(nn.Module):

    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(
            6,
            128,
        )

        self.relu1 = nn.ReLU()

        self.layer2 = nn.Linear(
            128,
            128,
        )

        self.relu2 = nn.ReLU()

        self.layer3 = nn.Linear(
            128,
            2,
        )

    def forward(self, x):

        x = self.layer1(x)
        x = self.relu1(x)

        x = self.layer2(x)
        x = self.relu2(x)

        tau = self.layer3(x)

        return tau


# =========================================================
# 4. Load trained policy
# =========================================================

checkpoint = torch.load(
    POLICY_PATH,
    map_location="cpu",
    weights_only=False,
)

policy = BCPolicy()

policy.load_state_dict(
    checkpoint["model_state_dict"]
)

policy.eval()

x_mean = checkpoint["x_mean"]
x_std = checkpoint["x_std"]

y_mean = checkpoint["y_mean"]
y_std = checkpoint["y_std"]

print("BC policy loaded:")
print(POLICY_PATH)


# =========================================================
# 5. Policy inference
# =========================================================

def bc_policy_action(
    q,
    dq,
    q_goal,
):

    x = np.concatenate(
        [
            q,
            dq,
            q_goal,
        ]
    )

    x = x.reshape(
        1,
        6,
    )

    x_normalized = (
        x - x_mean
    ) / x_std

    x_tensor = torch.tensor(
        x_normalized,
        dtype=torch.float32,
    )

    with torch.no_grad():

        normalized_tau = (
            policy(x_tensor)
            .numpy()
        )

    tau = (
        normalized_tau
        * y_std
        + y_mean
    )

    return tau[0]


# =========================================================
# 6. MuJoCo
# =========================================================

model = mujoco.MjModel.from_xml_path(
    str(MODEL_PATH)
)

data = mujoco.MjData(model)

dt = model.opt.timestep


# =========================================================
# 7. Evaluation settings
# =========================================================

num_episodes = 50

simulation_duration = 4.0

num_steps = int(
    simulation_duration
    / dt
)

last_duration = 0.5

last_steps = int(
    last_duration
    / dt
)


# Success criteria
#
# final joint error:
# |error| <= 0.05 rad
#
# final velocity:
# |dq| <= 0.05 rad/s

error_threshold = 0.05

velocity_threshold = 0.05


# =========================================================
# 8. Storage
# =========================================================

results = []


# =========================================================
# 9. Evaluate many unseen conditions
# =========================================================

for episode in range(
    num_episodes
):

    # ---------------------------------
    # Reset MuJoCo
    # ---------------------------------

    mujoco.mj_resetData(
        model,
        data,
    )

    # ---------------------------------
    # Random unseen initial state
    # ---------------------------------

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

    # ---------------------------------
    # Random unseen goal
    # ---------------------------------

    q_goal = (
        q_start
        + rng.uniform(
            low=-0.6,
            high=0.6,
            size=2,
        )
    )

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

    # ---------------------------------
    # Initial state
    # ---------------------------------

    data.qpos[:] = q_start
    data.qvel[:] = dq_start

    mujoco.mj_forward(
        model,
        data,
    )

    # ---------------------------------
    # Logs
    # ---------------------------------

    error_history = []
    tau_history = []

    # ---------------------------------
    # NN-only rollout
    # ---------------------------------

    for step in range(
        num_steps
    ):

        q = data.qpos.copy()
        dq = data.qvel.copy()

        tau = bc_policy_action(
            q,
            dq,
            q_goal,
        )

        data.qfrc_applied[:] = tau

        mujoco.mj_step(
            model,
            data,
        )

        error_history.append(
            q_goal - q
        )

        tau_history.append(
            tau.copy()
        )

    # ---------------------------------
    # Episode metrics
    # ---------------------------------

    error_history = np.array(
        error_history
    )

    tau_history = np.array(
        tau_history
    )

    final_q = data.qpos.copy()
    final_dq = data.qvel.copy()

    final_error = (
        final_q
        - q_goal
    )

    last_errors = (
        error_history[
            -last_steps:
        ]
    )

    steady_rmse = np.sqrt(
        np.mean(
            last_errors**2,
            axis=0,
        )
    )

    max_abs_tau = np.max(
        np.abs(
            tau_history
        ),
        axis=0,
    )

    # ---------------------------------
    # Success judgment
    # ---------------------------------

    error_ok = np.all(
        np.abs(final_error)
        <= error_threshold
    )

    velocity_ok = np.all(
        np.abs(final_dq)
        <= velocity_threshold
    )

    success = (
        error_ok
        and velocity_ok
    )

    # ---------------------------------
    # Save result
    # ---------------------------------

    results.append(
        {
            "episode":
                episode,

            "q_start_1":
                q_start[0],

            "q_start_2":
                q_start[1],

            "dq_start_1":
                dq_start[0],

            "dq_start_2":
                dq_start[1],

            "goal_1":
                q_goal[0],

            "goal_2":
                q_goal[1],

            "final_error_1":
                final_error[0],

            "final_error_2":
                final_error[1],

            "final_dq_1":
                final_dq[0],

            "final_dq_2":
                final_dq[1],

            "steady_rmse_1":
                steady_rmse[0],

            "steady_rmse_2":
                steady_rmse[1],

            "max_tau_1":
                max_abs_tau[0],

            "max_tau_2":
                max_abs_tau[1],

            "success":
                int(success),
        }
    )

    print(
        f"episode={episode:2d}, "
        f"success={success}, "
        f"final_error="
        f"{final_error}"
    )


# =========================================================
# 10. Aggregate metrics
# =========================================================

successes = np.array(
    [
        result["success"]
        for result in results
    ]
)

final_errors = np.array(
    [
        [
            result["final_error_1"],
            result["final_error_2"],
        ]
        for result in results
    ]
)

steady_rmses = np.array(
    [
        [
            result["steady_rmse_1"],
            result["steady_rmse_2"],
        ]
        for result in results
    ]
)

max_torques = np.array(
    [
        [
            result["max_tau_1"],
            result["max_tau_2"],
        ]
        for result in results
    ]
)


success_rate = (
    successes.mean()
)

mean_abs_final_error = np.mean(
    np.abs(
        final_errors
    ),
    axis=0,
)

median_abs_final_error = np.median(
    np.abs(
        final_errors
    ),
    axis=0,
)

p95_abs_final_error = np.percentile(
    np.abs(
        final_errors
    ),
    95,
    axis=0,
)

mean_steady_rmse = np.mean(
    steady_rmses,
    axis=0,
)

max_observed_torque = np.max(
    max_torques,
    axis=0,
)


# =========================================================
# 11. Summary
# =========================================================

print()
print(
    "=========================="
)

print(
    "Generalization Summary"
)

print(
    "=========================="
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
    "Mean abs final error [rad]:"
)

print(
    mean_abs_final_error
)

print()
print(
    "Median abs final error [rad]:"
)

print(
    median_abs_final_error
)

print()
print(
    "95th percentile abs error [rad]:"
)

print(
    p95_abs_final_error
)

print()
print(
    "Mean last-0.5s RMSE:"
)

print(
    mean_steady_rmse
)

print()
print(
    "Max observed torque:"
)

print(
    max_observed_torque
)


# =========================================================
# 12. Save CSV
# =========================================================

fieldnames = list(
    results[0].keys()
)

with CSV_PATH.open(
    "w",
    newline="",
    encoding="utf-8",
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames,
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