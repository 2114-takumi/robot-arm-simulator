import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "two_link_arm.xml"
OUTPUT_DIR = ROOT / "outputs" / "step_9_2"

sys.path.insert(0, str(ROOT))

from robot_arm.dynamics import (
    mass_matrix,
    coriolis_centrifugal_vector,
    gravity_vector,
)
from robot_arm.sensors import JointEncoderSensor
from scripts.run_coupled_dynamics_demo import physical_parameters


CTC_KP = np.array([25.0, 25.0])
CTC_KD = np.array([10.0, 10.0])

Q_START = np.array([0.0, 0.0])
Q_GOAL = np.array([0.3, -0.4])

MOVE_DURATION = 4.0
SIM_DURATION = 5.0


CASES = {
    "500Hz_0ms": {
        "sensor_hz": 500.0,
        "controller_hz": 500.0,
        "delay_s": 0.0,
    },
    "100Hz_0ms": {
        "sensor_hz": 100.0,
        "controller_hz": 100.0,
        "delay_s": 0.0,
    },
    "100Hz_10ms": {
        "sensor_hz": 100.0,
        "controller_hz": 100.0,
        "delay_s": 0.010,
    },
    "100Hz_20ms": {
        "sensor_hz": 100.0,
        "controller_hz": 100.0,
        "delay_s": 0.020,
    },
    "100Hz_50ms": {
        "sensor_hz": 100.0,
        "controller_hz": 100.0,
        "delay_s": 0.050,
    },
}


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

    delta_q = Q_GOAL - Q_START

    q_des = Q_START + delta_q * h
    dq_des = delta_q * dh
    ddq_des = delta_q * ddh

    return q_des, dq_des, ddq_des


def computed_torque_tracking(
    q,
    dq,
    q_des,
    dq_des,
    ddq_des,
    params,
):
    position_error = q_des - q
    velocity_error = dq_des - dq

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

    return M @ v + c + g


def frequency_to_steps(
    frequency_hz,
    physics_dt,
):
    """指定周波数がPhysics何ステップごとかを求める."""

    period = 1.0 / frequency_hz

    steps = round(
        period / physics_dt
    )

    if steps < 1:
        raise ValueError(
            "Update frequency is too high"
        )

    actual_period = (
        steps * physics_dt
    )

    if not np.isclose(
        actual_period,
        period,
        atol=1e-12,
    ):
        raise ValueError(
            "Frequency must align with "
            "the physics timestep"
        )

    return steps, period


def simulate(case_name):
    config = CASES[case_name]

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

    params = physical_parameters()

    sensor = JointEncoderSensor(
        counts_per_rev=4096,
        angle_noise_std=0.0005,
        velocity_cutoff_hz=20.0,
        seed=0,
    )

    physics_dt = model.opt.timestep

    sensor_steps, sensor_period = (
        frequency_to_steps(
            config["sensor_hz"],
            physics_dt,
        )
    )

    controller_steps, _ = (
        frequency_to_steps(
            config["controller_hz"],
            physics_dt,
        )
    )

    delay_samples = round(
        config["delay_s"]
        / sensor_period
    )

    if not np.isclose(
        delay_samples * sensor_period,
        config["delay_s"],
        atol=1e-12,
    ):
        raise ValueError(
            "Delay must be an integer "
            "multiple of sensor period"
        )

    num_steps = (
        int(SIM_DURATION / physics_dt)
        + 1
    )

    time_log = np.zeros(num_steps)

    q_true_log = np.zeros(
        (num_steps, 2)
    )

    q_des_log = np.zeros(
        (num_steps, 2)
    )

    tau_log = np.zeros(
        (num_steps, 2)
    )

    # センサがこれまで測定したデータ
    measurement_history = []

    # Controllerが現在使っている状態
    q_control = Q_START.copy()
    dq_control = np.zeros(2)

    # Controllerが現在出しているトルク
    tau = np.zeros(2)

    for i in range(num_steps):
        t = data.time

        q_true = data.qpos.copy()

        (
            q_des,
            dq_des,
            ddq_des,
        ) = desired_trajectory(t)

        # ---------------------------------
        # Sensor update
        # ---------------------------------

        if i % sensor_steps == 0:
            (
                q_measured,
                dq_measured,
            ) = sensor.measure(
                q_true,
                sensor_period,
            )

            measurement_history.append(
                (
                    q_measured.copy(),
                    dq_measured.copy(),
                )
            )

        # ---------------------------------
        # Controller update
        # ---------------------------------

        if i % controller_steps == 0:
            if measurement_history:
                history_index = (
                    len(measurement_history)
                    - 1
                    - delay_samples
                )

                # 起動直後で十分な過去データが
                # まだない場合は最古の値を使う
                history_index = max(
                    history_index,
                    0,
                )

                (
                    q_control,
                    dq_control,
                ) = measurement_history[
                    history_index
                ]

            tau = computed_torque_tracking(
                q_control,
                dq_control,
                q_des,
                dq_des,
                ddq_des,
                params,
            )

        # Controller更新がないPhysics stepでは
        # 前回のtauをそのまま保持する

        time_log[i] = t
        q_true_log[i] = q_true
        q_des_log[i] = q_des
        tau_log[i] = tau

        if i < num_steps - 1:
            data.qfrc_applied[:] = tau

            mujoco.mj_step(
                model,
                data,
            )

    return {
        "time": time_log,
        "q_true": q_true_log,
        "q_des": q_des_log,
        "tau": tau_log,
    }


def calculate_metrics(result):
    error = (
        result["q_des"]
        - result["q_true"]
    )

    return {
        "tracking_rmse": np.sqrt(
            np.mean(
                error**2,
                axis=0,
            )
        ),

        "max_tracking_error": np.max(
            np.abs(error),
            axis=0,
        ),

        "final_tracking_error":
            error[-1],

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

    # Tracking error
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

    plt.axhline(
        0.0,
        linestyle="--",
    )

    plt.xlabel("Time [s]")
    plt.ylabel(
        "Joint 1 tracking error [rad]"
    )

    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "joint1_delay_comparison.png",
        dpi=150,
    )

    plt.close()

    plt.figure()

    for case, result in results.items():
        error = (
            result["q_des"]
            - result["q_true"]
        )

        plt.plot(
            result["time"],
            error[:, 1],
            label=f"{case} J2",
        )

    plt.axhline(
        0.0,
        linestyle="--",
    )

    plt.xlabel("Time [s]")
    plt.ylabel(
        "Joint 2 tracking error [rad]"
    )

    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "joint2_delay_comparison.png",
        dpi=150,
    )

    plt.close()

    # RMSE comparison
    labels = list(results.keys())

    metrics = [
        calculate_metrics(
            results[name]
        )
        for name in labels
    ]

    rmse = np.array(
        [
            item["tracking_rmse"]
            for item in metrics
        ]
    )

    x = np.arange(
        len(labels)
    )

    width = 0.35

    plt.figure()

    plt.bar(
        x - width / 2,
        rmse[:, 0],
        width,
        label="Joint 1",
    )

    plt.bar(
        x + width / 2,
        rmse[:, 1],
        width,
        label="Joint 2",
    )

    plt.xticks(
        x,
        labels,
        rotation=20,
    )

    plt.ylabel(
        "Tracking RMSE [rad]"
    )

    plt.legend()
    plt.grid(
        axis="y"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "rmse_vs_delay.png",
        dpi=150,
    )

    plt.close()


def main():
    results = {}
    metrics = {}

    print(
        "=== Step 9-2 Rate / Delay Evaluation ==="
    )

    for case in CASES:
        result = simulate(case)

        results[case] = result
        metrics[case] = (
            calculate_metrics(
                result
            )
        )

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
            "Final tracking error [rad]:",
            metrics[case][
                "final_tracking_error"
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
    