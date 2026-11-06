import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "two_link_arm.xml"
OUTPUT_DIR = ROOT / "outputs" / "step_8_29"

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

TORQUE_LIMIT = np.array([10.0, 2.0])

MOVE_DURATION = 4.0
SIM_DURATION = 12.0

Q_START = np.array([0.0, 0.0])
Q_GOAL = np.array([0.3, -0.4])

M2_SCALE = 1.2

KAW_VALUES = (
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
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

    q_des = (
        Q_START
        + delta_q * h
    )

    dq_des = (
        delta_q * dh
    )

    ddq_des = (
        delta_q * ddh
    )

    return (
        q_des,
        dq_des,
        ddq_des,
    )


def controller_mass_matrix(q, params):
    """CTC内部モデルの質量行列."""

    return mass_matrix(
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


def trajectory_ctc_integral(
    q,
    dq,
    q_des,
    dq_des,
    ddq_des,
    integral_error,
    params,
):
    """CTC+Iが要求するトルクを計算."""

    position_error = (
        q_des - q
    )

    velocity_error = (
        dq_des - dq
    )

    v = (
        ddq_des
        + CTC_KD * velocity_error
        + CTC_KP * position_error
        + CTC_KI * integral_error
    )

    M = controller_mass_matrix(
        q,
        params,
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

    return (
        M @ v
        + c
        + g
    )


def update_back_calculation(
    q,
    integral_error,
    position_error,
    torque_command,
    torque_actual,
    kaw,
    params,
    dt,
):
    """Back-calculationによる積分状態更新."""

    # 飽和によって失われたトルク
    delta_tau = (
        torque_actual
        - torque_command
    )

    # トルク差を仮想加速度差へ変換
    M = controller_mass_matrix(
        q,
        params,
    )

    delta_v = np.linalg.solve(
        M,
        delta_tau,
    )

    # 積分器への補正
    correction = (
        kaw
        * delta_v
        / CTC_KI
    )

    integral_next = (
        integral_error
        + dt
        * (
            position_error
            + correction
        )
    )

    integral_next = np.clip(
        integral_next,
        -INTEGRAL_LIMIT,
        INTEGRAL_LIMIT,
    )

    return (
        integral_next,
        delta_tau,
        correction,
    )


def simulate(kaw):
    """指定KawでMuJoCoを実行."""

    model = mujoco.MjModel.from_xml_path(
        str(MODEL_PATH)
    )

    data = mujoco.MjData(model)

    data.qpos[:] = Q_START
    data.qvel[:] = 0.0

    mujoco.mj_forward(
        model,
        data,
    )

    controller_params = (
        physical_parameters().copy()
    )

    controller_params["m2"] *= (
        M2_SCALE
    )

    dt = model.opt.timestep

    num_steps = (
        int(SIM_DURATION / dt)
        + 1
    )

    time_log = np.zeros(
        num_steps
    )

    error_log = np.zeros(
        (num_steps, 2)
    )

    integral_log = np.zeros(
        (num_steps, 2)
    )

    command_log = np.zeros(
        (num_steps, 2)
    )

    actual_log = np.zeros(
        (num_steps, 2)
    )

    delta_tau_log = np.zeros(
        (num_steps, 2)
    )

    correction_log = np.zeros(
        (num_steps, 2)
    )

    saturation_log = np.zeros(
        (num_steps, 2),
        dtype=bool,
    )

    integral_error = np.zeros(2)

    for i in range(num_steps):
        t = data.time

        q = data.qpos.copy()
        dq = data.qvel.copy()

        (
            q_des,
            dq_des,
            ddq_des,
        ) = desired_trajectory(t)

        position_error = (
            q_des - q
        )

        torque_command = (
            trajectory_ctc_integral(
                q,
                dq,
                q_des,
                dq_des,
                ddq_des,
                integral_error,
                controller_params,
            )
        )

        torque_actual = np.clip(
            torque_command,
            -TORQUE_LIMIT,
            TORQUE_LIMIT,
        )

        saturated = (
            np.abs(
                torque_command
                - torque_actual
            )
            > 1e-12
        )

        time_log[i] = t
        error_log[i] = position_error
        integral_log[i] = integral_error
        command_log[i] = torque_command
        actual_log[i] = torque_actual
        saturation_log[i] = saturated

        if i < num_steps - 1:
            data.qfrc_applied[:] = (
                torque_actual
            )

            mujoco.mj_step(
                model,
                data,
            )

            (
                integral_error,
                delta_tau,
                correction,
            ) = update_back_calculation(
                q,
                integral_error,
                position_error,
                torque_command,
                torque_actual,
                kaw,
                controller_params,
                dt,
            )

            delta_tau_log[i] = (
                delta_tau
            )

            correction_log[i] = (
                correction
            )

    return {
        "time": time_log,
        "error": error_log,
        "integral": integral_log,
        "command": command_log,
        "actual": actual_log,
        "delta_tau": delta_tau_log,
        "correction": correction_log,
        "saturation": saturation_log,
    }


def calculate_metrics(result):
    """評価指標を計算."""

    return {
        "position_rmse": np.sqrt(
            np.mean(
                result["error"]**2,
                axis=0,
            )
        ),

        "max_position_error": np.max(
            np.abs(
                result["error"]
            ),
            axis=0,
        ),

        "final_position_error": (
            result["error"][-1]
        ),

        "max_integral": np.max(
            np.abs(
                result["integral"]
            ),
            axis=0,
        ),

        "final_integral": (
            result["integral"][-1]
        ),

        "max_command_torque": np.max(
            np.abs(
                result["command"]
            ),
            axis=0,
        ),

        "max_actual_torque": np.max(
            np.abs(
                result["actual"]
            ),
            axis=0,
        ),

        "saturation_fraction": np.mean(
            result["saturation"],
            axis=0,
        ),

        "max_correction": np.max(
            np.abs(
                result["correction"]
            ),
            axis=0,
        ),
    }


def save_plots(results):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------
    # Position error
    # ------------------------

    plt.figure()

    for kaw, result in results.items():
        plt.plot(
            result["time"],
            result["error"][:, 0],
            label=f"Kaw={kaw:g}",
        )

    plt.axhline(
        0.0,
        linestyle="--",
    )

    plt.xlabel("Time [s]")
    plt.ylabel(
        "Joint 1 position error [rad]"
    )

    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "joint1_error_comparison.png",
        dpi=150,
    )

    plt.close()

    plt.figure()

    for kaw, result in results.items():
        plt.plot(
            result["time"],
            result["error"][:, 1],
            label=f"Kaw={kaw:g}",
        )

    plt.axhline(
        0.0,
        linestyle="--",
    )

    plt.xlabel("Time [s]")
    plt.ylabel(
        "Joint 2 position error [rad]"
    )

    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "joint2_error_comparison.png",
        dpi=150,
    )

    plt.close()

    # ------------------------
    # Integral state
    # ------------------------

    plt.figure()

    for kaw, result in results.items():
        plt.plot(
            result["time"],
            result["integral"][:, 0],
            label=f"Kaw={kaw:g}",
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
    plt.ylabel(
        "Joint 1 integral [rad s]"
    )

    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "joint1_integral_comparison.png",
        dpi=150,
    )

    plt.close()

    plt.figure()

    for kaw, result in results.items():
        plt.plot(
            result["time"],
            result["integral"][:, 1],
            label=f"Kaw={kaw:g}",
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
    plt.ylabel(
        "Joint 2 integral [rad s]"
    )

    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "joint2_integral_comparison.png",
        dpi=150,
    )

    plt.close()

    # ------------------------
    # RMSE vs Kaw
    # ------------------------

    kaw_array = np.array(
        list(results.keys())
    )

    metrics = [
        calculate_metrics(
            results[kaw]
        )
        for kaw in kaw_array
    ]

    rmse = np.array(
        [
            value["position_rmse"]
            for value in metrics
        ]
    )

    plt.figure()

    plt.plot(
        kaw_array,
        rmse[:, 0],
        marker="o",
        label="Joint 1",
    )

    plt.plot(
        kaw_array,
        rmse[:, 1],
        marker="o",
        label="Joint 2",
    )

    plt.xlabel("Kaw")
    plt.ylabel(
        "Position RMSE [rad]"
    )

    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "rmse_vs_kaw.png",
        dpi=150,
    )

    plt.close()

    # ------------------------
    # command torque vs Kaw
    # ------------------------

    maximum_command = np.array(
        [
            value[
                "max_command_torque"
            ]
            for value in metrics
        ]
    )

    plt.figure()

    plt.plot(
        kaw_array,
        maximum_command[:, 0],
        marker="o",
        label="Joint 1",
    )

    plt.plot(
        kaw_array,
        maximum_command[:, 1],
        marker="o",
        label="Joint 2",
    )

    plt.xlabel("Kaw")
    plt.ylabel(
        "Maximum command torque [N m]"
    )

    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "max_command_vs_kaw.png",
        dpi=150,
    )

    plt.close()


def print_metrics(all_metrics):
    print(
        "=== MuJoCo Back-Calculation Evaluation ==="
    )

    print(
        "Torque limits [N m]:",
        TORQUE_LIMIT,
    )

    print(
        "Controller model mismatch: "
        "m2 +20%"
    )

    for kaw in KAW_VALUES:
        m = all_metrics[kaw]

        print()
        print(
            f"--- Kaw={kaw:g} ---"
        )

        print(
            "Position RMSE [rad]:",
            m["position_rmse"],
        )

        print(
            "Maximum position error [rad]:",
            m[
                "max_position_error"
            ],
        )

        print(
            "Final position error [rad]:",
            m[
                "final_position_error"
            ],
        )

        print(
            "Maximum integral [rad s]:",
            m["max_integral"],
        )

        print(
            "Final integral [rad s]:",
            m["final_integral"],
        )

        print(
            "Maximum commanded torque [N m]:",
            m[
                "max_command_torque"
            ],
        )

        print(
            "Maximum actual torque [N m]:",
            m[
                "max_actual_torque"
            ],
        )

        print(
            "Saturation fraction:",
            m[
                "saturation_fraction"
            ],
        )

        print(
            "Maximum BC correction:",
            m[
                "max_correction"
            ],
        )


def main():
    results = {}
    all_metrics = {}

    for kaw in KAW_VALUES:
        print(
            f"Running Kaw={kaw:g}..."
        )

        result = simulate(
            kaw
        )

        results[kaw] = result

        all_metrics[kaw] = (
            calculate_metrics(
                result
            )
        )

    print()
    print_metrics(
        all_metrics
    )

    save_plots(
        results
    )

    # RMSE合計が最小のKaw
    scores = {
        kaw: np.sum(
            all_metrics[kaw][
                "position_rmse"
            ]
        )
        for kaw in KAW_VALUES
    }

    best_kaw = min(
        scores,
        key=scores.get,
    )

    print()
    print(
        "Best Kaw by sum of RMSE:",
        best_kaw,
    )

    print(
        "RMSE:",
        all_metrics[
            best_kaw
        ]["position_rmse"],
    )

    print(
        "Output directory:",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
    