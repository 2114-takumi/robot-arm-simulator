import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "two_link_arm.xml"
OUTPUT_DIR = ROOT / "outputs" / "step_8_23"

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

MODEL_CASES = {
    "nominal": 1.0,
    "m2_plus_20": 1.2,
    "m2_minus_20": 0.8,
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
    """モデルパラメータparamsを使ってCTCトルクを計算する."""

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
    """CTC内部のm2だけを変更してMuJoCoを実行する."""

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    data.qpos[:] = Q_START
    data.qvel[:] = 0.0

    mujoco.mj_forward(model, data)

    # CTCが信じているモデル
    controller_params = physical_parameters().copy()

    # MuJoCo側は変更せず、制御器側だけ変更
    controller_params["m2"] *= m2_scale

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
            controller_params,
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

    return {
        "time": time_log,
        "q": q_log,
        "dq": dq_log,
        "q_des": q_des_log,
        "dq_des": dq_des_log,
        "tau": tau_log,
    }


def calculate_metrics(result):
    position_error = result["q_des"] - result["q"]
    velocity_error = result["dq_des"] - result["dq"]

    result["position_error"] = position_error
    result["velocity_error"] = velocity_error

    return {
        "position_rmse": np.sqrt(
            np.mean(position_error**2, axis=0)
        ),
        "max_position_error": np.max(
            np.abs(position_error),
            axis=0,
        ),
        "velocity_rmse": np.sqrt(
            np.mean(velocity_error**2, axis=0)
        ),
        "max_torque": np.max(
            np.abs(result["tau"]),
            axis=0,
        ),
        "final_position_error": position_error[-1],
    }


def save_error_plot(results):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Joint 1
    plt.figure()

    for case_name, result in results.items():
        plt.plot(
            result["time"],
            result["position_error"][:, 0],
            label=case_name,
        )

    plt.xlabel("Time [s]")
    plt.ylabel("Joint 1 position error [rad]")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "joint1_error_comparison.png",
        dpi=150,
    )

    plt.close()

    # Joint 2
    plt.figure()

    for case_name, result in results.items():
        plt.plot(
            result["time"],
            result["position_error"][:, 1],
            label=case_name,
        )

    plt.xlabel("Time [s]")
    plt.ylabel("Joint 2 position error [rad]")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "joint2_error_comparison.png",
        dpi=150,
    )

    plt.close()


def print_metrics(all_metrics):
    print("=== MuJoCo Model Mismatch Evaluation ===")

    for case_name, metrics in all_metrics.items():
        print()
        print(f"--- {case_name} ---")

        print(
            "Position RMSE [rad]:",
            metrics["position_rmse"],
        )

        print(
            "Maximum position error [rad]:",
            metrics["max_position_error"],
        )

        print(
            "Velocity RMSE [rad/s]:",
            metrics["velocity_rmse"],
        )

        print(
            "Maximum torque [N m]:",
            metrics["max_torque"],
        )

        print(
            "Final position error [rad]:",
            metrics["final_position_error"],
        )


def main():
    results = {}
    all_metrics = {}

    for case_name, m2_scale in MODEL_CASES.items():
        result = simulate(m2_scale)

        metrics = calculate_metrics(result)

        results[case_name] = result
        all_metrics[case_name] = metrics

    save_error_plot(results)

    print_metrics(all_metrics)

    print()
    print("Output directory:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
    