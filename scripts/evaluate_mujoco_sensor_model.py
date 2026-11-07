import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "two_link_arm.xml"
OUTPUT_DIR = ROOT / "outputs" / "step_9_1"

sys.path.insert(0, str(ROOT))

from robot_arm.dynamics import (
    mass_matrix,
    coriolis_centrifugal_vector,
    gravity_vector,
)
from robot_arm.sensors import JointEncoderSensor
from scripts.run_coupled_dynamics_demo import (
    physical_parameters,
)


CTC_KP = np.array([25.0, 25.0])
CTC_KD = np.array([10.0, 10.0])

Q_START = np.array([0.0, 0.0])
Q_GOAL = np.array([0.3, -0.4])

MOVE_DURATION = 4.0
SIM_DURATION = 5.0

CASES = (
    "truth",
    "encoder",
)


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

    delta_q = (
        Q_GOAL - Q_START
    )

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


def computed_torque_tracking(
    q,
    dq,
    q_des,
    dq_des,
    ddq_des,
    params,
):
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

    return (
        M @ v
        + c
        + g
    )


def simulate(case):
    if case not in CASES:
        raise ValueError(
            f"Unknown case: {case}"
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

    params = (
        physical_parameters()
    )

    sensor = JointEncoderSensor(
        counts_per_rev=4096,
        angle_noise_std=0.0005,
        velocity_cutoff_hz=20.0,
        seed=0,
    )

    dt = model.opt.timestep

    num_steps = (
        int(SIM_DURATION / dt)
        + 1
    )

    time_log = np.zeros(num_steps)

    q_true_log = np.zeros(
        (num_steps, 2)
    )

    dq_true_log = np.zeros(
        (num_steps, 2)
    )

    q_measured_log = np.zeros(
        (num_steps, 2)
    )

    dq_measured_log = np.zeros(
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

        # MuJoCoの真値
        q_true = data.qpos.copy()
        dq_true = data.qvel.copy()

        if case == "truth":
            q_control = (
                q_true.copy()
            )

            dq_control = (
                dq_true.copy()
            )

        else:
            (
                q_control,
                dq_control,
            ) = sensor.measure(
                q_true,
                dt,
            )

        (
            q_des,
            dq_des,
            ddq_des,
        ) = desired_trajectory(t)

        # 制御器にはq_control/dq_controlだけを渡す
        tau = computed_torque_tracking(
            q_control,
            dq_control,
            q_des,
            dq_des,
            ddq_des,
            params,
        )

        time_log[i] = t

        q_true_log[i] = (
            q_true
        )

        dq_true_log[i] = (
            dq_true
        )

        q_measured_log[i] = (
            q_control
        )

        dq_measured_log[i] = (
            dq_control
        )

        q_des_log[i] = (
            q_des
        )

        tau_log[i] = (
            tau
        )

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
        "q_true": q_true_log,
        "dq_true": dq_true_log,
        "q_measured": q_measured_log,
        "dq_measured": dq_measured_log,
        "q_des": q_des_log,
        "tau": tau_log,
    }


def calculate_metrics(result):
    tracking_error = (
        result["q_des"]
        - result["q_true"]
    )

    measurement_error = (
        result["q_measured"]
        - result["q_true"]
    )

    velocity_error = (
        result["dq_measured"]
        - result["dq_true"]
    )

    return {
        "tracking_rmse": np.sqrt(
            np.mean(
                tracking_error**2,
                axis=0,
            )
        ),

        "max_tracking_error": np.max(
            np.abs(
                tracking_error
            ),
            axis=0,
        ),

        "angle_measurement_rmse": np.sqrt(
            np.mean(
                measurement_error**2,
                axis=0,
            )
        ),

        "velocity_estimation_rmse": np.sqrt(
            np.mean(
                velocity_error**2,
                axis=0,
            )
        ),

        "max_torque": np.max(
            np.abs(
                result["tau"]
            ),
            axis=0,
        ),
    }


def save_plots(results):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Tracking error comparison
    plt.figure()

    for case, result in results.items():
        error = (
            result["q_des"]
            - result["q_true"]
        )

        plt.plot(
            result["time"],
            error[:, 0],
            label=f"{case} J1",
        )

        plt.plot(
            result["time"],
            error[:, 1],
            "--",
            label=f"{case} J2",
        )

    plt.axhline(
        0.0,
        linestyle="--",
    )

    plt.xlabel("Time [s]")
    plt.ylabel(
        "Tracking error [rad]"
    )

    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "tracking_error_comparison.png",
        dpi=150,
    )

    plt.close()

    # Angle measurement error
    result = results["encoder"]

    measurement_error = (
        result["q_measured"]
        - result["q_true"]
    )

    plt.figure()

    plt.plot(
        result["time"],
        measurement_error[:, 0],
        label="Joint 1",
    )

    plt.plot(
        result["time"],
        measurement_error[:, 1],
        label="Joint 2",
    )

    plt.xlabel("Time [s]")
    plt.ylabel(
        "Angle measurement error [rad]"
    )

    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "angle_measurement_error.png",
        dpi=150,
    )

    plt.close()

    # Velocity estimate comparison
    plt.figure()

    plt.plot(
        result["time"],
        result["dq_true"][:, 0],
        label="J1 true",
    )

    plt.plot(
        result["time"],
        result["dq_measured"][:, 0],
        "--",
        label="J1 estimated",
    )

    plt.plot(
        result["time"],
        result["dq_true"][:, 1],
        label="J2 true",
    )

    plt.plot(
        result["time"],
        result["dq_measured"][:, 1],
        "--",
        label="J2 estimated",
    )

    plt.xlabel("Time [s]")
    plt.ylabel(
        "Angular velocity [rad/s]"
    )

    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "velocity_estimation.png",
        dpi=150,
    )

    plt.close()


def main():
    results = {}
    metrics = {}

    for case in CASES:
        result = simulate(case)

        results[case] = result
        metrics[case] = (
            calculate_metrics(
                result
            )
        )

    print(
        "=== Step 9-1 Sensor Model Evaluation ==="
    )

    for case in CASES:
        print()
        print(f"--- {case} ---")

        print(
            "Tracking RMSE [rad]:",
            metrics[case][
                "tracking_rmse"
            ],
        )

        print(
            "Maximum tracking error [rad]:",
            metrics[case][
                "max_tracking_error"
            ],
        )

        print(
            "Angle measurement RMSE [rad]:",
            metrics[case][
                "angle_measurement_rmse"
            ],
        )

        print(
            "Velocity estimation RMSE [rad/s]:",
            metrics[case][
                "velocity_estimation_rmse"
            ],
        )

        print(
            "Maximum torque [N m]:",
            metrics[case][
                "max_torque"
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