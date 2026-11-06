import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "two_link_arm.xml"
OUTPUT_DIR = ROOT / "outputs" / "step_8_25"

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

MOVE_DURATION = 4.0
SIM_DURATION = 10.0

Q_START = np.array([0.0, 0.0])
Q_GOAL = np.array([0.3, -0.4])

M2_SCALE = 1.2

CONTROLLERS = (
    "CTC",
    "CTC_I",
)


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

    h = (
        10 * s**3
        - 15 * s**4
        + 6 * s**5
    )

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


def trajectory_ctc(
    q,
    dq,
    q_des,
    dq_des,
    ddq_des,
    integral_error,
    use_integral,
    params,
):
    """軌道追従CTC / CTC+I."""

    position_error = q_des - q
    velocity_error = dq_des - dq

    v = (
        ddq_des
        + CTC_KD * velocity_error
        + CTC_KP * position_error
    )

    if use_integral:
        v += CTC_KI * integral_error

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


def simulate(controller):
    model = mujoco.MjModel.from_xml_path(
        str(MODEL_PATH)
    )

    data = mujoco.MjData(model)

    data.qpos[:] = Q_START
    data.qvel[:] = 0.0

    mujoco.mj_forward(model, data)

    controller_params = physical_parameters().copy()

    # 制御器だけm2を20%過大評価
    controller_params["m2"] *= M2_SCALE

    dt = model.opt.timestep

    num_steps = int(
        SIM_DURATION / dt
    ) + 1

    time_log = np.zeros(num_steps)

    q_log = np.zeros((num_steps, 2))
    q_des_log = np.zeros((num_steps, 2))

    error_log = np.zeros((num_steps, 2))
    integral_log = np.zeros((num_steps, 2))

    tau_log = np.zeros((num_steps, 2))

    integral_error = np.zeros(2)

    use_integral = (
        controller == "CTC_I"
    )

    for i in range(num_steps):
        t = data.time

        q = data.qpos.copy()
        dq = data.qvel.copy()

        q_des, dq_des, ddq_des = (
            desired_trajectory(t)
        )

        position_error = q_des - q

        tau = trajectory_ctc(
            q,
            dq,
            q_des,
            dq_des,
            ddq_des,
            integral_error,
            use_integral,
            controller_params,
        )

        time_log[i] = t

        q_log[i] = q
        q_des_log[i] = q_des

        error_log[i] = position_error
        integral_log[i] = integral_error

        tau_log[i] = tau

        if i < num_steps - 1:
            data.qfrc_applied[:] = tau

            mujoco.mj_step(
                model,
                data,
            )

            if use_integral:
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
        "q": q_log,
        "q_des": q_des_log,
        "error": error_log,
        "integral": integral_log,
        "tau": tau_log,
    }


def calculate_metrics(result):
    error = result["error"]

    return {
        "position_rmse": np.sqrt(
            np.mean(
                error**2,
                axis=0,
            )
        ),
        "max_position_error": np.max(
            np.abs(error),
            axis=0,
        ),
        "final_position_error": (
            error[-1]
        ),
        "max_torque": np.max(
            np.abs(result["tau"]),
            axis=0,
        ),
        "final_integral": (
            result["integral"][-1]
        ),
    }


def save_plots(results):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 位置誤差
    plt.figure()

    for name, result in results.items():
        plt.plot(
            result["time"],
            result["error"][:, 0],
            label=f"{name} Joint 1",
        )

        plt.plot(
            result["time"],
            result["error"][:, 1],
            "--",
            label=f"{name} Joint 2",
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
        OUTPUT_DIR
        / "tracking_error_comparison.png",
        dpi=150,
    )

    plt.close()

    # 積分状態
    plt.figure()

    result = results["CTC_I"]

    plt.plot(
        result["time"],
        result["integral"][:, 0],
        label="Joint 1 integral",
    )

    plt.plot(
        result["time"],
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
        OUTPUT_DIR
        / "integral_state.png",
        dpi=150,
    )

    plt.close()

    # トルク
    plt.figure()

    for name, result in results.items():
        plt.plot(
            result["time"],
            result["tau"][:, 0],
            label=f"{name} Joint 1",
        )

        plt.plot(
            result["time"],
            result["tau"][:, 1],
            "--",
            label=f"{name} Joint 2",
        )

    plt.xlabel("Time [s]")
    plt.ylabel("Torque [N m]")

    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "torque_comparison.png",
        dpi=150,
    )

    plt.close()


def main():
    results = {}
    metrics = {}

    for controller in CONTROLLERS:
        result = simulate(controller)

        results[controller] = result

        metrics[controller] = (
            calculate_metrics(result)
        )

    print(
        "=== MuJoCo CTC vs CTC+I ==="
    )

    print(
        "Controller model mismatch: "
        "m2 +20%"
    )

    for name in CONTROLLERS:
        print()
        print(f"--- {name} ---")

        print(
            "Position RMSE [rad]:",
            metrics[name][
                "position_rmse"
            ],
        )

        print(
            "Maximum position error [rad]:",
            metrics[name][
                "max_position_error"
            ],
        )

        print(
            "Final position error [rad]:",
            metrics[name][
                "final_position_error"
            ],
        )

        print(
            "Maximum torque [N m]:",
            metrics[name][
                "max_torque"
            ],
        )

        print(
            "Final integral [rad s]:",
            metrics[name][
                "final_integral"
            ],
        )

    save_plots(results)

    print()
    print(
        "Output directory:",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()