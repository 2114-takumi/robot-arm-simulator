import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "two_link_arm.xml"
OUTPUT_DIR = ROOT / "outputs" / "step_8_30"

sys.path.insert(0, str(ROOT))

from robot_arm.dynamics import (
    mass_matrix,
    coriolis_centrifugal_vector,
    gravity_vector,
    integrate_dynamics,
)
from scripts.run_coupled_dynamics_demo import physical_parameters


CTC_KP = np.array([25.0, 25.0])
CTC_KD = np.array([10.0, 10.0])

MOVE_DURATION = 4.0
SIM_DURATION = 5.0

Q_START = np.array([0.0, 0.0])
Q_GOAL = np.array([0.3, -0.4])


CASES = {
    "nominal": 1.0,
    "controller_m2_plus_20": 1.2,
}


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


def trajectory_computed_torque(
    q,
    dq,
    q_des,
    dq_des,
    ddq_des,
    controller_params,
):
    """軌道追従用Computed Torque Control."""

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
    )

    M = mass_matrix(
        q[0],
        q[1],
        L1=controller_params["L1"],
        m1=controller_params["m1"],
        m2=controller_params["m2"],
        lc1=controller_params["lc1"],
        lc2=controller_params["lc2"],
        I1=controller_params["I1"],
        I2=controller_params["I2"],
    )

    c = coriolis_centrifugal_vector(
        q[0],
        q[1],
        dq[0],
        dq[1],
        L1=controller_params["L1"],
        m2=controller_params["m2"],
        lc2=controller_params["lc2"],
    )

    g = gravity_vector(
        q[0],
        q[1],
        L1=controller_params["L1"],
        m1=controller_params["m1"],
        m2=controller_params["m2"],
        lc1=controller_params["lc1"],
        lc2=controller_params["lc2"],
        g=9.81,
    )

    return (
        M @ v
        + c
        + g
    )


def make_controller_params(
    m2_scale,
):
    """制御器が使用するモデルを作る."""

    params = (
        physical_parameters().copy()
    )

    params["m2"] *= (
        m2_scale
    )

    return params


def simulate_custom(
    m2_scale,
    dt,
):
    """自作2リンク動力学でシミュレーション."""

    plant_params = (
        physical_parameters().copy()
    )

    controller_params = (
        make_controller_params(
            m2_scale
        )
    )

    num_steps = (
        int(SIM_DURATION / dt)
        + 1
    )

    time_log = (
        np.arange(num_steps)
        * dt
    )

    state_log = np.zeros(
        (num_steps, 4)
    )

    q_des_log = np.zeros(
        (num_steps, 2)
    )

    tau_log = np.zeros(
        (num_steps, 2)
    )

    state_log[0, :2] = (
        Q_START
    )

    for i in range(num_steps):
        t = time_log[i]

        q = (
            state_log[i, :2]
        )

        dq = (
            state_log[i, 2:]
        )

        (
            q_des,
            dq_des,
            ddq_des,
        ) = desired_trajectory(t)

        tau = (
            trajectory_computed_torque(
                q,
                dq,
                q_des,
                dq_des,
                ddq_des,
                controller_params,
            )
        )

        q_des_log[i] = (
            q_des
        )

        tau_log[i] = (
            tau
        )

        if i < num_steps - 1:
            state_log[i + 1] = (
                integrate_dynamics(
                    q[0],
                    q[1],
                    dq[0],
                    dq[1],
                    tau[0],
                    tau[1],
                    dt,
                    **plant_params,
                    b1=0.0,
                    b2=0.0,
                    g=9.81,
                    method="rk2",
                )
            )

    return {
        "time": time_log,
        "q": state_log[:, :2],
        "dq": state_log[:, 2:],
        "q_des": q_des_log,
        "tau": tau_log,
    }


def simulate_mujoco(
    m2_scale,
):
    """MuJoCoで同じ制御実験を行う."""

    model = (
        mujoco.MjModel.from_xml_path(
            str(MODEL_PATH)
        )
    )

    data = mujoco.MjData(
        model
    )

    data.qpos[:] = (
        Q_START
    )

    data.qvel[:] = 0.0

    mujoco.mj_forward(
        model,
        data,
    )

    controller_params = (
        make_controller_params(
            m2_scale
        )
    )

    dt = (
        model.opt.timestep
    )

    num_steps = (
        int(SIM_DURATION / dt)
        + 1
    )

    time_log = np.zeros(
        num_steps
    )

    q_log = np.zeros(
        (num_steps, 2)
    )

    dq_log = np.zeros(
        (num_steps, 2)
    )

    q_des_log = np.zeros(
        (num_steps, 2)
    )

    tau_log = np.zeros(
        (num_steps, 2)
    )

    for i in range(num_steps):
        t = data.time

        q = data.qpos.copy()
        dq = data.qvel.copy()

        (
            q_des,
            dq_des,
            ddq_des,
        ) = desired_trajectory(t)

        tau = (
            trajectory_computed_torque(
                q,
                dq,
                q_des,
                dq_des,
                ddq_des,
                controller_params,
            )
        )

        time_log[i] = t
        q_log[i] = q
        dq_log[i] = dq
        q_des_log[i] = q_des
        tau_log[i] = tau

        if i < num_steps - 1:
            data.qfrc_applied[:] = (
                tau
            )

            mujoco.mj_step(
                model,
                data,
            )

    return {
        "time": time_log,
        "q": q_log,
        "dq": dq_log,
        "q_des": q_des_log,
        "tau": tau_log,
        "dt": dt,
    }


def tracking_metrics(
    result,
):
    """各シミュレータ単体の追従性能."""

    error = (
        result["q_des"]
        - result["q"]
    )

    return {
        "position_rmse": (
            np.sqrt(
                np.mean(
                    error**2,
                    axis=0,
                )
            )
        ),

        "max_position_error": (
            np.max(
                np.abs(error),
                axis=0,
            )
        ),

        "final_position_error": (
            error[-1]
        ),

        "max_torque": (
            np.max(
                np.abs(
                    result["tau"]
                ),
                axis=0,
            )
        ),
    }


def cross_validation_metrics(
    custom,
    mujoco_result,
):
    """自作シミュレータとMuJoCoの差を計算."""

    if len(custom["time"]) != len(
        mujoco_result["time"]
    ):
        raise ValueError(
            "The two simulations have "
            "different sample counts."
        )

    q_difference = (
        custom["q"]
        - mujoco_result["q"]
    )

    dq_difference = (
        custom["dq"]
        - mujoco_result["dq"]
    )

    tau_difference = (
        custom["tau"]
        - mujoco_result["tau"]
    )

    return {
        "q_difference": (
            q_difference
        ),

        "dq_difference": (
            dq_difference
        ),

        "tau_difference": (
            tau_difference
        ),

        "angle_difference_rmse": (
            np.sqrt(
                np.mean(
                    q_difference**2,
                    axis=0,
                )
            )
        ),

        "max_angle_difference": (
            np.max(
                np.abs(
                    q_difference
                ),
                axis=0,
            )
        ),

        "velocity_difference_rmse": (
            np.sqrt(
                np.mean(
                    dq_difference**2,
                    axis=0,
                )
            )
        ),

        "max_velocity_difference": (
            np.max(
                np.abs(
                    dq_difference
                ),
                axis=0,
            )
        ),

        "torque_difference_rmse": (
            np.sqrt(
                np.mean(
                    tau_difference**2,
                    axis=0,
                )
            )
        ),

        "max_torque_difference": (
            np.max(
                np.abs(
                    tau_difference
                ),
                axis=0,
            )
        ),
    }


def save_case_plots(
    case_name,
    custom,
    mujoco_result,
    cross_metrics,
):
    """1ケース分の比較グラフを保存."""

    case_dir = (
        OUTPUT_DIR
        / case_name
    )

    case_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    time = (
        mujoco_result["time"]
    )

    # -------------------------
    # Joint angle
    # -------------------------

    plt.figure()

    plt.plot(
        time,
        custom["q"][:, 0],
        label="Custom J1",
    )

    plt.plot(
        time,
        mujoco_result["q"][:, 0],
        "--",
        label="MuJoCo J1",
    )

    plt.plot(
        time,
        custom["q"][:, 1],
        label="Custom J2",
    )

    plt.plot(
        time,
        mujoco_result["q"][:, 1],
        "--",
        label="MuJoCo J2",
    )

    plt.xlabel(
        "Time [s]"
    )

    plt.ylabel(
        "Joint angle [rad]"
    )

    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(
        case_dir
        / "angle_comparison.png",
        dpi=150,
    )

    plt.close()

    # -------------------------
    # Difference
    # -------------------------

    plt.figure()

    plt.plot(
        time,
        cross_metrics[
            "q_difference"
        ][:, 0],
        label="J1",
    )

    plt.plot(
        time,
        cross_metrics[
            "q_difference"
        ][:, 1],
        label="J2",
    )

    plt.axhline(
        0.0,
        linestyle="--",
    )

    plt.xlabel(
        "Time [s]"
    )

    plt.ylabel(
        "Custom - MuJoCo angle [rad]"
    )

    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(
        case_dir
        / "angle_difference.png",
        dpi=150,
    )

    plt.close()

    # -------------------------
    # Torque comparison
    # -------------------------

    plt.figure()

    plt.plot(
        time,
        custom["tau"][:, 0],
        label="Custom J1",
    )

    plt.plot(
        time,
        mujoco_result["tau"][:, 0],
        "--",
        label="MuJoCo J1",
    )

    plt.plot(
        time,
        custom["tau"][:, 1],
        label="Custom J2",
    )

    plt.plot(
        time,
        mujoco_result["tau"][:, 1],
        "--",
        label="MuJoCo J2",
    )

    plt.xlabel(
        "Time [s]"
    )

    plt.ylabel(
        "Torque [N m]"
    )

    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(
        case_dir
        / "torque_comparison.png",
        dpi=150,
    )

    plt.close()


def print_case_metrics(
    case_name,
    custom_metrics,
    mujoco_metrics,
    cross_metrics,
):
    print()
    print(
        f"=== {case_name} ==="
    )

    print()
    print("--- Custom simulator ---")

    print(
        "Position RMSE [rad]:",
        custom_metrics[
            "position_rmse"
        ],
    )

    print(
        "Maximum position error [rad]:",
        custom_metrics[
            "max_position_error"
        ],
    )

    print(
        "Final position error [rad]:",
        custom_metrics[
            "final_position_error"
        ],
    )

    print()
    print("--- MuJoCo ---")

    print(
        "Position RMSE [rad]:",
        mujoco_metrics[
            "position_rmse"
        ],
    )

    print(
        "Maximum position error [rad]:",
        mujoco_metrics[
            "max_position_error"
        ],
    )

    print(
        "Final position error [rad]:",
        mujoco_metrics[
            "final_position_error"
        ],
    )

    print()
    print("--- Custom vs MuJoCo ---")

    print(
        "Angle difference RMSE [rad]:",
        cross_metrics[
            "angle_difference_rmse"
        ],
    )

    print(
        "Maximum angle difference [rad]:",
        cross_metrics[
            "max_angle_difference"
        ],
    )

    print(
        "Velocity difference RMSE [rad/s]:",
        cross_metrics[
            "velocity_difference_rmse"
        ],
    )

    print(
        "Maximum velocity difference [rad/s]:",
        cross_metrics[
            "max_velocity_difference"
        ],
    )

    print(
        "Torque difference RMSE [N m]:",
        cross_metrics[
            "torque_difference_rmse"
        ],
    )

    print(
        "Maximum torque difference [N m]:",
        cross_metrics[
            "max_torque_difference"
        ],
    )


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"MuJoCo model was not found: "
            f"{MODEL_PATH}"
        )

    # MuJoCoのdtを基準にする
    test_model = (
        mujoco.MjModel.from_xml_path(
            str(MODEL_PATH)
        )
    )

    dt = (
        test_model.opt.timestep
    )

    print(
        "=== Custom Simulator vs MuJoCo ==="
    )

    print(
        "Simulation timestep:",
        dt,
    )

    print(
        "Simulation duration:",
        SIM_DURATION,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for (
        case_name,
        m2_scale,
    ) in CASES.items():

        custom = simulate_custom(
            m2_scale,
            dt,
        )

        mujoco_result = (
            simulate_mujoco(
                m2_scale
            )
        )

        custom_metrics = (
            tracking_metrics(
                custom
            )
        )

        mujoco_metrics = (
            tracking_metrics(
                mujoco_result
            )
        )

        cross_metrics = (
            cross_validation_metrics(
                custom,
                mujoco_result,
            )
        )

        print_case_metrics(
            case_name,
            custom_metrics,
            mujoco_metrics,
            cross_metrics,
        )

        save_case_plots(
            case_name,
            custom,
            mujoco_result,
            cross_metrics,
        )

    print()
    print(
        "Output directory:",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()