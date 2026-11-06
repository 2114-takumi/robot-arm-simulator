import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "two_link_arm.xml"
OUTPUT_DIR = ROOT / "outputs" / "step_8_22"

sys.path.insert(0, str(ROOT))

from robot_arm.dynamics import (
    mass_matrix,
    coriolis_centrifugal_vector,
    gravity_vector,
)
from scripts.run_coupled_dynamics_demo import physical_parameters


CTC_KP = np.array([25.0, 25.0])
CTC_KD = np.array([10.0, 10.0])

MOVE_DURATION = 4.0
SIM_DURATION = 5.0

Q_START = np.array([0.0, 0.0])
Q_GOAL = np.array([0.3, -0.4])


def desired_trajectory(t):
    """5次多項式による点-to-点軌道."""

    T = MOVE_DURATION

    if t <= 0.0:
        return (
            Q_START.copy(),
            np.zeros(2),
            np.zeros(2),
        )

    if t >= T:
        return (
            Q_GOAL.copy(),
            np.zeros(2),
            np.zeros(2),
        )

    s = t / T

    h = 10 * s**3 - 15 * s**4 + 6 * s**5

    dh = (
        30 * s**2
        - 60 * s**3
        + 30 * s**4
    ) / T

    ddh = (
        60 * s
        - 180 * s**2
        + 120 * s**3
    ) / T**2

    delta_q = Q_GOAL - Q_START

    q_des = Q_START + delta_q * h
    dq_des = delta_q * dh
    ddq_des = delta_q * ddh

    return q_des, dq_des, ddq_des


def trajectory_computed_torque(
    q,
    dq,
    q_des,
    dq_des,
    ddq_des,
    kp,
    kd,
    params,
):
    """軌道追従用Computed Torque Control."""

    position_error = q_des - q
    velocity_error = dq_des - dq

    v = (
        ddq_des
        + kd * velocity_error
        + kp * position_error
    )

    M = mass_matrix(
        q[0],
        q[1],
        L1=params["L1"],
        m1=params["m1"],
        m2=params["m2"],
        lc1=params["lc1"],
        lc2=params["lc2"],
        I1=params["I1"],
        I2=params["I2"],
    )

    c = coriolis_centrifugal_vector(
        q[0],
        q[1],
        dq[0],
        dq[1],
        L1=params["L1"],
        m2=params["m2"],
        lc2=params["lc2"],
    )

    g = gravity_vector(
        q[0],
        q[1],
        L1=params["L1"],
        m1=params["m1"],
        m2=params["m2"],
        lc1=params["lc1"],
        lc2=params["lc2"],
        g=9.81,
    )

    return M @ v + c + g


def simulate():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    data.qpos[:] = Q_START
    data.qvel[:] = 0.0

    mujoco.mj_forward(model, data)

    params = physical_parameters()

    dt = model.opt.timestep
    num_steps = int(SIM_DURATION / dt) + 1

    time_log = np.zeros(num_steps)

    q_log = np.zeros((num_steps, 2))
    dq_log = np.zeros((num_steps, 2))

    q_des_log = np.zeros((num_steps, 2))
    dq_des_log = np.zeros((num_steps, 2))

    tau_log = np.zeros((num_steps, 2))

    for i in range(num_steps):
        t = data.time

        q = data.qpos.copy()
        dq = data.qvel.copy()

        q_des, dq_des, ddq_des = desired_trajectory(t)

        tau = trajectory_computed_torque(
            q,
            dq,
            q_des,
            dq_des,
            ddq_des,
            CTC_KP,
            CTC_KD,
            params,
        )

        time_log[i] = t

        q_log[i] = q
        dq_log[i] = dq

        q_des_log[i] = q_des
        dq_des_log[i] = dq_des

        tau_log[i] = tau

        if i < num_steps - 1:
            data.qfrc_applied[:] = tau
            mujoco.mj_step(model, data)

    return (
        time_log,
        q_log,
        dq_log,
        q_des_log,
        dq_des_log,
        tau_log,
    )


def calculate_metrics(
    q_log,
    dq_log,
    q_des_log,
    dq_des_log,
    tau_log,
):
    position_error = q_des_log - q_log
    velocity_error = dq_des_log - dq_log

    rmse_position = np.sqrt(
        np.mean(position_error**2, axis=0)
    )

    max_position_error = np.max(
        np.abs(position_error),
        axis=0,
    )

    rmse_velocity = np.sqrt(
        np.mean(velocity_error**2, axis=0)
    )

    max_torque = np.max(
        np.abs(tau_log),
        axis=0,
    )

    final_position_error = position_error[-1]

    return {
        "position_error": position_error,
        "velocity_error": velocity_error,
        "rmse_position": rmse_position,
        "max_position_error": max_position_error,
        "rmse_velocity": rmse_velocity,
        "max_torque": max_torque,
        "final_position_error": final_position_error,
    }


def save_plots(
    time_log,
    q_log,
    q_des_log,
    tau_log,
    position_error,
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 関節角度
    plt.figure()

    plt.plot(
        time_log,
        q_log[:, 0],
        label="Joint 1",
    )

    plt.plot(
        time_log,
        q_des_log[:, 0],
        "--",
        label="Joint 1 desired",
    )

    plt.plot(
        time_log,
        q_log[:, 1],
        label="Joint 2",
    )

    plt.plot(
        time_log,
        q_des_log[:, 1],
        "--",
        label="Joint 2 desired",
    )

    plt.xlabel("Time [s]")
    plt.ylabel("Joint angle [rad]")
    plt.legend()
    plt.grid()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "joint_tracking.png",
        dpi=150,
    )

    plt.close()

    # 追従誤差
    plt.figure()

    plt.plot(
        time_log,
        position_error[:, 0],
        label="Joint 1 error",
    )

    plt.plot(
        time_log,
        position_error[:, 1],
        label="Joint 2 error",
    )

    plt.xlabel("Time [s]")
    plt.ylabel("Position error [rad]")
    plt.legend()
    plt.grid()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "tracking_error.png",
        dpi=150,
    )

    plt.close()

    # トルク
    plt.figure()

    plt.plot(
        time_log,
        tau_log[:, 0],
        label="Joint 1 torque",
    )

    plt.plot(
        time_log,
        tau_log[:, 1],
        label="Joint 2 torque",
    )

    plt.xlabel("Time [s]")
    plt.ylabel("Torque [N m]")
    plt.legend()
    plt.grid()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "joint_torque.png",
        dpi=150,
    )

    plt.close()


def main():
    (
        time_log,
        q_log,
        dq_log,
        q_des_log,
        dq_des_log,
        tau_log,
    ) = simulate()

    metrics = calculate_metrics(
        q_log,
        dq_log,
        q_des_log,
        dq_des_log,
        tau_log,
    )

    save_plots(
        time_log,
        q_log,
        q_des_log,
        tau_log,
        metrics["position_error"],
    )

    print("=== MuJoCo Trajectory Tracking Evaluation ===")

    print(
        "Position RMSE [rad]:",
        metrics["rmse_position"],
    )

    print(
        "Maximum position error [rad]:",
        metrics["max_position_error"],
    )

    print(
        "Velocity RMSE [rad/s]:",
        metrics["rmse_velocity"],
    )

    print(
        "Maximum torque [N m]:",
        metrics["max_torque"],
    )

    print(
        "Final position error [rad]:",
        metrics["final_position_error"],
    )

    print("Output directory:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
    