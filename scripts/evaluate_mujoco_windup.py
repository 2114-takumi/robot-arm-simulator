import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "two_link_arm.xml"
OUTPUT_DIR = ROOT / "outputs" / "step_8_26"

sys.path.insert(0, str(ROOT))

from robot_arm.dynamics import (
    mass_matrix,
    coriolis_centrifugal_vector,
    gravity_vector,
)
from scripts.run_coupled_dynamics_demo import physical_parameters


CTC_KP = np.array([25.0, 25.0])
CTC_KD = np.array([10.0, 10.0])
CTC_KI = np.array([10.0, 10.0])

INTEGRAL_LIMIT = 0.5

# 実トルク制限
TORQUE_LIMIT = np.array([10.0, 2.0])

MOVE_DURATION = 4.0
SIM_DURATION = 12.0

Q_START = np.array([0.0, 0.0])
Q_GOAL = np.array([0.3, -0.4])

# 前回と同じモデル誤差
M2_SCALE = 1.2


def desired_trajectory(t):
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


def trajectory_ctc_integral(
    q,
    dq,
    q_des,
    dq_des,
    ddq_des,
    integral_error,
    params,
):
    position_error = q_des - q
    velocity_error = dq_des - dq

    v = (
        ddq_des
        + CTC_KD * velocity_error
        + CTC_KP * position_error
        + CTC_KI * integral_error
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
    model = mujoco.MjModel.from_xml_path(
        str(MODEL_PATH)
    )

    data = mujoco.MjData(model)

    data.qpos[:] = Q_START
    data.qvel[:] = 0.0

    mujoco.mj_forward(model, data)

    controller_params = physical_parameters().copy()

    # CTC内部だけm2を20%過大評価
    controller_params["m2"] *= M2_SCALE

    dt = model.opt.timestep
    num_steps = int(SIM_DURATION / dt) + 1

    time_log = np.zeros(num_steps)

    error_log = np.zeros((num_steps, 2))
    integral_log = np.zeros((num_steps, 2))

    torque_command_log = np.zeros((num_steps, 2))
    torque_actual_log = np.zeros((num_steps, 2))

    saturation_log = np.zeros((num_steps, 2), dtype=bool)

    integral_error = np.zeros(2)

    for i in range(num_steps):
        t = data.time

        q = data.qpos.copy()
        dq = data.qvel.copy()

        q_des, dq_des, ddq_des = desired_trajectory(t)

        position_error = q_des - q

        # 制御器が要求するトルク
        torque_command = trajectory_ctc_integral(
            q,
            dq,
            q_des,
            dq_des,
            ddq_des,
            integral_error,
            controller_params,
        )

        # アクチュエータが実際に出せるトルク
        torque_actual = np.clip(
            torque_command,
            -TORQUE_LIMIT,
            TORQUE_LIMIT,
        )

        saturated = (
            np.abs(torque_command - torque_actual)
            > 1e-12
        )

        time_log[i] = t
        error_log[i] = position_error
        integral_log[i] = integral_error

        torque_command_log[i] = torque_command
        torque_actual_log[i] = torque_actual

        saturation_log[i] = saturated

        if i < num_steps - 1:
            # MuJoCoには飽和後の実トルクを入力
            data.qfrc_applied[:] = torque_actual

            mujoco.mj_step(
                model,
                data,
            )

            # 飽和していても積分を続ける
            # → windupを意図的に発生させる
            integral_error += (
                position_error * dt
            )

            integral_error = np.clip(
                integral_error,
                -INTEGRAL_LIMIT,
                INTEGRAL_LIMIT,
            )

    return {
        "time": time_log,
        "error": error_log,
        "integral": integral_log,
        "torque_command": torque_command_log,
        "torque_actual": torque_actual_log,
        "saturation": saturation_log,
    }


def save_plots(result):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    time = result["time"]

    # 1. 位置誤差
    plt.figure()

    plt.plot(
        time,
        result["error"][:, 0],
        label="Joint 1 error",
    )

    plt.plot(
        time,
        result["error"][:, 1],
        label="Joint 2 error",
    )

    plt.axhline(
        0.0,
        linestyle="--",
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

    # 2. 積分状態
    plt.figure()

    plt.plot(
        time,
        result["integral"][:, 0],
        label="Joint 1 integral",
    )

    plt.plot(
        time,
        result["integral"][:, 1],
        label="Joint 2 integral",
    )

    plt.axhline(
        INTEGRAL_LIMIT,
        linestyle="--",
    )

    plt.axhline(
        -INTEGRAL_LIMIT,
        linestyle="--",
    )

    plt.xlabel("Time [s]")
    plt.ylabel("Integral error [rad s]")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "integral_state.png",
        dpi=150,
    )

    plt.close()

    # 3. 要求トルクと実トルク
    plt.figure()

    plt.plot(
        time,
        result["torque_command"][:, 0],
        label="Joint 1 command",
    )

    plt.plot(
        time,
        result["torque_actual"][:, 0],
        "--",
        label="Joint 1 actual",
    )

    plt.axhline(
        TORQUE_LIMIT[0],
        linestyle="--",
    )

    plt.axhline(
        -TORQUE_LIMIT[0],
        linestyle="--",
    )

    plt.xlabel("Time [s]")
    plt.ylabel("Joint 1 torque [N m]")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "joint1_torque.png",
        dpi=150,
    )

    plt.close()

    plt.figure()

    plt.plot(
        time,
        result["torque_command"][:, 1],
        label="Joint 2 command",
    )

    plt.plot(
        time,
        result["torque_actual"][:, 1],
        "--",
        label="Joint 2 actual",
    )

    plt.axhline(
        TORQUE_LIMIT[1],
        linestyle="--",
    )

    plt.axhline(
        -TORQUE_LIMIT[1],
        linestyle="--",
    )

    plt.xlabel("Time [s]")
    plt.ylabel("Joint 2 torque [N m]")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "joint2_torque.png",
        dpi=150,
    )

    plt.close()


def print_metrics(result):
    error = result["error"]
    integral = result["integral"]

    torque_command = result[
        "torque_command"
    ]

    torque_actual = result[
        "torque_actual"
    ]

    saturation = result["saturation"]

    print(
        "=== MuJoCo CTC+I Windup Evaluation ==="
    )

    print(
        "Torque limits [N m]:",
        TORQUE_LIMIT,
    )

    print(
        "Position RMSE [rad]:",
        np.sqrt(
            np.mean(
                error**2,
                axis=0,
            )
        ),
    )

    print(
        "Maximum position error [rad]:",
        np.max(
            np.abs(error),
            axis=0,
        ),
    )

    print(
        "Final position error [rad]:",
        error[-1],
    )

    print(
        "Maximum integral [rad s]:",
        np.max(
            np.abs(integral),
            axis=0,
        ),
    )

    print(
        "Final integral [rad s]:",
        integral[-1],
    )

    print(
        "Maximum commanded torque [N m]:",
        np.max(
            np.abs(torque_command),
            axis=0,
        ),
    )

    print(
        "Maximum actual torque [N m]:",
        np.max(
            np.abs(torque_actual),
            axis=0,
        ),
    )

    print(
        "Saturation fraction:",
        np.mean(
            saturation,
            axis=0,
        ),
    )


def main():
    result = simulate()

    print_metrics(result)

    save_plots(result)

    print()
    print(
        "Output directory:",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()