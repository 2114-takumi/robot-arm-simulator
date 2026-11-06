import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "two_link_arm.xml"
OUTPUT_DIR = ROOT / "outputs" / "step_8_28"

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

# CTC内部だけm2を20%過大評価
M2_SCALE = 1.2

METHODS = (
    "standard",
    "global_conditional",
    "per_joint_conditional",
)


def desired_trajectory(t):
    """5次多項式による滑らかな点-to-点軌道."""

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


def trajectory_ctc_integral(
    q,
    dq,
    q_des,
    dq_des,
    ddq_des,
    integral_error,
    params,
):
    """CTC+Iの要求トルクを計算する."""

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


def simulate(method):
    """指定したanti-windup方式でMuJoCoを実行する."""

    if method not in METHODS:
        raise ValueError(
            f"Unknown method: {method}"
        )

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

    # 制御器のモデルだけ変更
    controller_params["m2"] *= M2_SCALE

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

    saturation_log = np.zeros(
        (num_steps, 2),
        dtype=bool,
    )

    integral_stopped_log = np.zeros(
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

        error_log[i] = (
            position_error
        )

        integral_log[i] = (
            integral_error
        )

        command_log[i] = (
            torque_command
        )

        actual_log[i] = (
            torque_actual
        )

        saturation_log[i] = (
            saturated
        )

        if i < num_steps - 1:
            # MuJoCoには飽和後の実トルクを入力
            data.qfrc_applied[:] = (
                torque_actual
            )

            mujoco.mj_step(
                model,
                data,
            )

            # ----------------------------
            # 積分更新
            # ----------------------------

            if method == "standard":
                # 飽和していても常に積分
                integral_error += (
                    position_error
                    * dt
                )

            elif method == "global_conditional":
                # どちらか一方でも飽和したら
                # 両関節の積分を止める
                if np.any(saturated):
                    integral_stopped_log[
                        i, :
                    ] = True

                else:
                    integral_error += (
                        position_error
                        * dt
                    )

            elif method == "per_joint_conditional":
                # 各関節を独立して判定
                for joint in range(2):
                    if saturated[joint]:
                        integral_stopped_log[
                            i,
                            joint,
                        ] = True

                    else:
                        integral_error[
                            joint
                        ] += (
                            position_error[
                                joint
                            ]
                            * dt
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
        "command": command_log,
        "actual": actual_log,
        "saturation": saturation_log,
        "integral_stopped": (
            integral_stopped_log
        ),
    }


def calculate_metrics(result):
    """各方式の評価指標を計算する."""

    return {
        "position_rmse": np.sqrt(
            np.mean(
                result["error"]**2,
                axis=0,
            )
        ),

        "max_position_error": (
            np.max(
                np.abs(
                    result["error"]
                ),
                axis=0,
            )
        ),

        "final_position_error": (
            result["error"][-1]
        ),

        "max_integral": (
            np.max(
                np.abs(
                    result["integral"]
                ),
                axis=0,
            )
        ),

        "final_integral": (
            result["integral"][-1]
        ),

        "max_command_torque": (
            np.max(
                np.abs(
                    result["command"]
                ),
                axis=0,
            )
        ),

        "max_actual_torque": (
            np.max(
                np.abs(
                    result["actual"]
                ),
                axis=0,
            )
        ),

        "saturation_fraction": (
            np.mean(
                result["saturation"],
                axis=0,
            )
        ),

        "integral_stopped_fraction": (
            np.mean(
                result[
                    "integral_stopped"
                ][:-1],
                axis=0,
            )
        ),
    }


def save_plots(results):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------
    # 1. 位置誤差比較
    # ----------------------------

    plt.figure()

    for method, result in results.items():
        plt.plot(
            result["time"],
            result["error"][:, 0],
            label=f"{method} J1",
        )

        plt.plot(
            result["time"],
            result["error"][:, 1],
            "--",
            label=f"{method} J2",
        )

    plt.axhline(
        0.0,
        linestyle="--",
    )

    plt.xlabel(
        "Time [s]"
    )

    plt.ylabel(
        "Position error [rad]"
    )

    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "error_comparison.png",
        dpi=150,
    )

    plt.close()

    # ----------------------------
    # 2. 積分状態比較
    # ----------------------------

    plt.figure()

    for method, result in results.items():
        plt.plot(
            result["time"],
            result["integral"][:, 0],
            label=f"{method} J1",
        )

        plt.plot(
            result["time"],
            result["integral"][:, 1],
            "--",
            label=f"{method} J2",
        )

    plt.axhline(
        INTEGRAL_LIMIT,
        linestyle="--",
    )

    plt.axhline(
        -INTEGRAL_LIMIT,
        linestyle="--",
    )

    plt.xlabel(
        "Time [s]"
    )

    plt.ylabel(
        "Integral error [rad s]"
    )

    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "integral_comparison.png",
        dpi=150,
    )

    plt.close()

    # ----------------------------
    # 3. Joint 1 要求トルク比較
    # ----------------------------

    plt.figure()

    for method, result in results.items():
        plt.plot(
            result["time"],
            result["command"][:, 0],
            label=method,
        )

    plt.axhline(
        TORQUE_LIMIT[0],
        linestyle="--",
    )

    plt.axhline(
        -TORQUE_LIMIT[0],
        linestyle="--",
    )

    plt.xlabel(
        "Time [s]"
    )

    plt.ylabel(
        "Joint 1 command torque [N m]"
    )

    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "joint1_command_comparison.png",
        dpi=150,
    )

    plt.close()

    # ----------------------------
    # 4. Joint 2 要求トルク比較
    # ----------------------------

    plt.figure()

    for method, result in results.items():
        plt.plot(
            result["time"],
            result["command"][:, 1],
            label=method,
        )

    plt.axhline(
        TORQUE_LIMIT[1],
        linestyle="--",
    )

    plt.axhline(
        -TORQUE_LIMIT[1],
        linestyle="--",
    )

    plt.xlabel(
        "Time [s]"
    )

    plt.ylabel(
        "Joint 2 command torque [N m]"
    )

    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "joint2_command_comparison.png",
        dpi=150,
    )

    plt.close()

    # ----------------------------
    # 5. 飽和率比較
    # ----------------------------

    labels = list(results.keys())

    joint1_fraction = [
        np.mean(
            results[method][
                "saturation"
            ][:, 0]
        )
        for method in labels
    ]

    joint2_fraction = [
        np.mean(
            results[method][
                "saturation"
            ][:, 1]
        )
        for method in labels
    ]

    x = np.arange(
        len(labels)
    )

    width = 0.35

    plt.figure()

    plt.bar(
        x - width / 2,
        joint1_fraction,
        width,
        label="Joint 1",
    )

    plt.bar(
        x + width / 2,
        joint2_fraction,
        width,
        label="Joint 2",
    )

    plt.xticks(
        x,
        labels,
        rotation=15,
    )

    plt.ylabel(
        "Saturation fraction"
    )

    plt.legend()
    plt.grid(
        axis="y"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "saturation_fraction.png",
        dpi=150,
    )

    plt.close()


def print_metrics(metrics):
    print(
        "=== MuJoCo Per-Joint "
        "Anti-Windup Evaluation ==="
    )

    print(
        "Torque limits [N m]:",
        TORQUE_LIMIT,
    )

    print(
        "Controller model mismatch: "
        "m2 +20%"
    )

    for method in METHODS:
        m = metrics[method]

        print()
        print(
            f"--- {method} ---"
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
            "Integral stopped fraction:",
            m[
                "integral_stopped_fraction"
            ],
        )


def main():
    results = {}
    metrics = {}

    for method in METHODS:
        result = simulate(
            method
        )

        results[method] = (
            result
        )

        metrics[method] = (
            calculate_metrics(
                result
            )
        )

    print_metrics(
        metrics
    )

    save_plots(
        results
    )

    print()

    print(
        "Output directory:",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()