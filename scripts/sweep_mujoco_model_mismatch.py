import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "two_link_arm.xml"
OUTPUT_DIR = ROOT / "outputs" / "step_8_24"

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

# 0.5 ～ 1.5 を5%刻み
M2_SCALES = np.linspace(0.5, 1.5, 21)


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
    """指定モデルparamsを使う軌道追従CTC."""

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


def simulate(m2_scale):
    """1つのm2倍率でMuJoCoシミュレーションを実行."""

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    data.qpos[:] = Q_START
    data.qvel[:] = 0.0

    mujoco.mj_forward(model, data)

    controller_params = physical_parameters().copy()

    # CTC内部だけm2を変更
    controller_params["m2"] *= m2_scale

    dt = model.opt.timestep
    num_steps = int(SIM_DURATION / dt) + 1

    position_errors = np.zeros((num_steps, 2))
    velocity_errors = np.zeros((num_steps, 2))
    torques = np.zeros((num_steps, 2))

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
            controller_params,
        )

        position_errors[i] = q_des - q
        velocity_errors[i] = dq_des - dq
        torques[i] = tau

        if i < num_steps - 1:
            data.qfrc_applied[:] = tau
            mujoco.mj_step(model, data)

    return {
        "position_rmse": np.sqrt(
            np.mean(position_errors**2, axis=0)
        ),
        "max_position_error": np.max(
            np.abs(position_errors),
            axis=0,
        ),
        "velocity_rmse": np.sqrt(
            np.mean(velocity_errors**2, axis=0)
        ),
        "max_torque": np.max(
            np.abs(torques),
            axis=0,
        ),
        "final_position_error": position_errors[-1],
    }


def save_plots(
    error_percent,
    position_rmse,
    max_position_error,
    max_torque,
    final_position_error,
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # RMSE
    plt.figure()

    plt.plot(
        error_percent,
        position_rmse[:, 0],
        marker="o",
        label="Joint 1",
    )

    plt.plot(
        error_percent,
        position_rmse[:, 1],
        marker="o",
        label="Joint 2",
    )

    plt.axvline(
        0.0,
        linestyle="--",
    )

    plt.xlabel("m2 model error [%]")
    plt.ylabel("Position RMSE [rad]")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "rmse_vs_model_error.png",
        dpi=150,
    )

    plt.close()

    # 最大位置誤差
    plt.figure()

    plt.plot(
        error_percent,
        max_position_error[:, 0],
        marker="o",
        label="Joint 1",
    )

    plt.plot(
        error_percent,
        max_position_error[:, 1],
        marker="o",
        label="Joint 2",
    )

    plt.axvline(
        0.0,
        linestyle="--",
    )

    plt.xlabel("m2 model error [%]")
    plt.ylabel("Maximum position error [rad]")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "max_error_vs_model_error.png",
        dpi=150,
    )

    plt.close()

    # 最大トルク
    plt.figure()

    plt.plot(
        error_percent,
        max_torque[:, 0],
        marker="o",
        label="Joint 1",
    )

    plt.plot(
        error_percent,
        max_torque[:, 1],
        marker="o",
        label="Joint 2",
    )

    plt.axvline(
        0.0,
        linestyle="--",
    )

    plt.xlabel("m2 model error [%]")
    plt.ylabel("Maximum torque [N m]")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "max_torque_vs_model_error.png",
        dpi=150,
    )

    plt.close()

    # 最終位置誤差
    plt.figure()

    plt.plot(
        error_percent,
        final_position_error[:, 0],
        marker="o",
        label="Joint 1",
    )

    plt.plot(
        error_percent,
        final_position_error[:, 1],
        marker="o",
        label="Joint 2",
    )

    plt.axhline(
        0.0,
        linestyle="--",
    )

    plt.axvline(
        0.0,
        linestyle="--",
    )

    plt.xlabel("m2 model error [%]")
    plt.ylabel("Final position error [rad]")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "final_error_vs_model_error.png",
        dpi=150,
    )

    plt.close()


def main():
    num_cases = len(M2_SCALES)

    position_rmse = np.zeros((num_cases, 2))
    max_position_error = np.zeros((num_cases, 2))
    velocity_rmse = np.zeros((num_cases, 2))
    max_torque = np.zeros((num_cases, 2))
    final_position_error = np.zeros((num_cases, 2))

    for i, m2_scale in enumerate(M2_SCALES):
        metrics = simulate(m2_scale)

        position_rmse[i] = metrics["position_rmse"]
        max_position_error[i] = metrics["max_position_error"]
        velocity_rmse[i] = metrics["velocity_rmse"]
        max_torque[i] = metrics["max_torque"]
        final_position_error[i] = metrics[
            "final_position_error"
        ]

        error_percent = (m2_scale - 1.0) * 100.0

        print(
            f"m2 error={error_percent:+5.1f}% | "
            f"RMSE=[{position_rmse[i, 0]:.5f}, "
            f"{position_rmse[i, 1]:.5f}] | "
            f"Max error=[{max_position_error[i, 0]:.5f}, "
            f"{max_position_error[i, 1]:.5f}]"
        )

    error_percent = (M2_SCALES - 1.0) * 100.0

    save_plots(
        error_percent,
        position_rmse,
        max_position_error,
        max_torque,
        final_position_error,
    )

    print()
    print("=== Sweep completed ===")

    print(
        "Best Joint 1 RMSE:",
        np.min(position_rmse[:, 0]),
        "at",
        error_percent[
            np.argmin(position_rmse[:, 0])
        ],
        "%",
    )

    print(
        "Best Joint 2 RMSE:",
        np.min(position_rmse[:, 1]),
        "at",
        error_percent[
            np.argmin(position_rmse[:, 1])
        ],
        "%",
    )

    print("Output directory:", OUTPUT_DIR)


if __name__ == "__main__":
    main()