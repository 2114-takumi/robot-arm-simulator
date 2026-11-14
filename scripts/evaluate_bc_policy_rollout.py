from pathlib import Path

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


# =========================================================
# 2. Policy Network
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
# 3. Load trained policy
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


# Normalization parameters
x_mean = checkpoint["x_mean"]
x_std = checkpoint["x_std"]

y_mean = checkpoint["y_mean"]
y_std = checkpoint["y_std"]

print("BC policy loaded:")
print(POLICY_PATH)


# =========================================================
# 4. Policy inference function
# =========================================================

def bc_policy_action(
    q,
    dq,
    q_goal,
):

    # ---------------------------------
    # Construct input
    # ---------------------------------

    x = np.concatenate(
        [
            q,
            dq,
            q_goal,
        ]
    )

    # [6]
    # ->
    # [1, 6]

    x = x.reshape(
        1,
        6,
    )

    # ---------------------------------
    # Normalize input
    # ---------------------------------

    x_normalized = (
        x - x_mean
    ) / x_std

    # ---------------------------------
    # NumPy -> Tensor
    # ---------------------------------

    x_tensor = torch.tensor(
        x_normalized,
        dtype=torch.float32,
    )

    # ---------------------------------
    # Neural Network
    # ---------------------------------

    with torch.no_grad():

        normalized_tau = (
            policy(x_tensor)
            .numpy()
        )

    # ---------------------------------
    # Denormalize output
    # ---------------------------------

    tau = (
        normalized_tau
        * y_std
        + y_mean
    )

    return tau[0]


# =========================================================
# 5. MuJoCo
# =========================================================

model = mujoco.MjModel.from_xml_path(
    str(MODEL_PATH)
)

data = mujoco.MjData(
    model
)

dt = model.opt.timestep


# =========================================================
# 6. Unseen test condition
# =========================================================

q_start = np.array(
    [
        -0.4,
        0.2,
    ]
)

dq_start = np.array(
    [
        0.1,
        -0.1,
    ]
)

q_goal = np.array(
    [
        0.35,
        -0.55,
    ]
)


data.qpos[:] = q_start
data.qvel[:] = dq_start

mujoco.mj_forward(
    model,
    data,
)


# =========================================================
# 7. Rollout settings
# =========================================================

simulation_duration = 4.0

num_steps = int(
    simulation_duration
    / dt
)

q_history = []
dq_history = []
tau_history = []
error_history = []


# =========================================================
# 8. BC policy controls robot
# =========================================================

for step in range(
    num_steps
):

    q = data.qpos.copy()
    dq = data.qvel.copy()

    # ---------------------------------
    # No CTC here.
    #
    # Neural network decides torque.
    # ---------------------------------

    tau = bc_policy_action(
        q,
        dq,
        q_goal,
    )

    # ---------------------------------
    # Apply NN action to MuJoCo
    # ---------------------------------

    data.qfrc_applied[:] = tau

    mujoco.mj_step(
        model,
        data,
    )

    # ---------------------------------
    # Log
    # ---------------------------------

    q_history.append(
        q.copy()
    )

    dq_history.append(
        dq.copy()
    )

    tau_history.append(
        tau.copy()
    )

    error_history.append(
        q_goal - q
    )


# =========================================================
# 9. Convert logs
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

error_history = np.array(
    error_history
)


# =========================================================
# 10. Final results
# =========================================================

final_q = data.qpos.copy()
final_dq = data.qvel.copy()

final_error = (
    final_q
    - q_goal
)


print()
print("Initial q:")
print(q_start)

print()
print("Goal q:")
print(q_goal)

print()
print("Final q:")
print(final_q)

print()
print("Final joint error [rad]:")
print(final_error)

print(
    "Final joint error [deg]:"
)

print(
    np.degrees(
        final_error
    )
)


# =========================================================
# 11. Final velocity
# =========================================================

print()
print(
    "Final dq:",
    final_dq,
)


# =========================================================
# 12. Last 0.5 s RMSE
# =========================================================

last_duration = 0.5

last_steps = int(
    last_duration
    / dt
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

print()
print(
    "Last 0.5 s joint RMSE:",
    steady_rmse,
)


# =========================================================
# 13. Torque
# =========================================================

max_abs_tau = np.max(
    np.abs(
        tau_history
    ),
    axis=0,
)

print()
print(
    "Max absolute torque:",
    max_abs_tau,
)
