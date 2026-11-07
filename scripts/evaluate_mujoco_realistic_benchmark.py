import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "two_link_arm.xml"
OUTPUT_DIR = ROOT / "outputs" / "step_9_4"

sys.path.insert(0, str(ROOT))

from robot_arm.sensors import JointEncoderSensor
from scripts.run_coupled_dynamics_demo import physical_parameters
from scripts.evaluate_mujoco_rate_delay import (
    Q_START,
    desired_trajectory,
    computed_torque_tracking,
    frequency_to_steps,
)


SIM_DURATION = 5.0

# Realistic condition
SENSOR_HZ = 100.0
CONTROLLER_HZ = 100.0
SENSOR_DELAY_S = 0.010

DISTURBANCE_START = 2.0
DISTURBANCE_END = 2.2

DISTURBANCE_TORQUE = np.array([
    0.0,
    1.0,
])

CASES = (
    "ideal",
    "realistic",
)


def external_disturbance(t, enabled):
    """Realistic条件でだけ、一時的な外乱トルクを返す."""

    if (
        enabled
        and DISTURBANCE_START <= t < DISTURBANCE_END
    ):
        return DISTURBANCE_TORQUE.copy()

    return np.zeros(2)


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

    params = physical_parameters()

    physics_dt = model.opt.timestep

    num_steps = (
        int(SIM_DURATION / physics_dt)
        + 1
    )

    # --------------------------------
    # Realistic case settings
    # --------------------------------

    if case == "realistic":
        sensor = JointEncoderSensor(
            counts_per_rev=4096,
            angle_noise_std=0.0005,
            velocity_cutoff_hz=20.0,
            seed=0,
        )

        sensor_steps, sensor_period = (
            frequency_to_steps(
                SENSOR_HZ,
                physics_dt,
            )
        )

        controller_steps, _ = (
            frequency_to_steps(
                CONTROLLER_HZ,
                physics_dt,
            )
        )

        delay_samples = round(
            SENSOR_DELAY_S
            / sensor_period
        )

        measurement_history = []

    else:
        sensor = None

        # IdealではPhysicsごとに更新
        sensor_steps = 1
        controller_steps = 1

        sensor_period = physics_dt
        delay_samples = 0

        measurement_history = []

    # --------------------------------
    # Logs
    # --------------------------------

    time_log = np.zeros(
        num_steps
    )

    q_true_log = np.zeros(
        (num_steps, 2)
    )

    dq_true_log = np.zeros(
        (num_steps, 2)
    )

    q_control_log = np.zeros(
        (num_steps, 2)
    )

    dq_control_log = np.zeros(
        (num_steps, 2)
    )

    q_des_log = np.zeros(
        (num_steps, 2)
    )

    control_torque_log = np.zeros(
        (num_steps, 2)
    )

    disturbance_log = np.zeros(
        (num_steps, 2)
    )

    total_torque_log = np.zeros(
        (num_steps, 2)
    )

    # Controllerが現在使っている状態
    q_control = Q_START.copy()
    dq_control = np.zeros(2)

    # 現在保持しているController出力
    control_torque = np.zeros(2)

    # --------------------------------
    # Simulation loop
    # --------------------------------

    for i in range(num_steps):
        t = data.time

        q_true = data.qpos.copy()
        dq_true = data.qvel.copy()

        (
            q_des,
            dq_des,
            ddq_des,
        ) = desired_trajectory(t)

        # ================================
        # Ideal
        # ================================

        if case == "ideal":
            # MuJoCoの真値をそのままControllerへ
            q_control = q_true.copy()
            dq_control = dq_true.copy()

            control_torque = (
                computed_torque_tracking(
                    q_control,
                    dq_control,
                    q_des,
                    dq_des,
                    ddq_des,
                    params,
                )
            )

        # ================================
        # Realistic
        # ================================

        else:
            # ----------------------------
            # Sensor update: 100 Hz
            # ----------------------------

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

            # ----------------------------
            # Controller update: 100 Hz
            # ----------------------------

            if i % controller_steps == 0:
                history_index = (
                    len(measurement_history)
                    - 1
                    - delay_samples
                )

                # 起動直後は十分な過去データがない
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

                control_torque = (
                    computed_torque_tracking(
                        q_control,
                        dq_control,
                        q_des,
                        dq_des,
                        ddq_des,
                        params,
                    )
                )

            # Controller更新がない間は
            # 前回のcontrol_torqueを保持する

        # --------------------------------
        # Disturbance
        # --------------------------------

        disturbance = (
            external_disturbance(
                t,
                case == "realistic",
            )
        )

        total_torque = (
            control_torque
            + disturbance
        )

        # --------------------------------
        # Logs
        # --------------------------------

        time_log[i] = t

        q_true_log[i] = q_true
        dq_true_log[i] = dq_true

        q_control_log[i] = q_control
        dq_control_log[i] = dq_control

        q_des_log[i] = q_des

        control_torque_log[i] = (
            control_torque
        )

        disturbance_log[i] = (
            disturbance
        )

        total_torque_log[i] = (
            total_torque
        )

        # --------------------------------
        # Physics update: 500 Hz
        # --------------------------------

        if i < num_steps - 1:
            data.qfrc_applied[:] = (
                total_torque
            )

            mujoco.mj_step(
                model,
                data,
            )

    return {
        "time": time_log,
        "q_true": q_true_log,
        "dq_true": dq_true_log,
        "q_control": q_control_log,
        "dq_control": dq_control_log,
        "q_des": q_des_log,
        "control_torque": control_torque_log,
        "disturbance": disturbance_log,
        "total_torque": total_torque_log,
    }


def calculate_metrics(result):
    tracking_error = (
        result["q_des"]
        - result["q_true"]
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

        "final_tracking_error":
            tracking_error[-1],

        "max_control_torque": np.max(
            np.abs(
                result["control_torque"]
            ),
            axis=0,
        ),

        "max_total_torque": np.max(
            np.abs(
                result["total_torque"]
            ),
            axis=0,
        ),
    }


def calculate_degradation(
    ideal_metrics,
    realistic_metrics,
):
    """IdealからRealisticへ性能がどれだけ悪化したか."""

    ideal_rmse = (
        ideal_metrics[
            "tracking_rmse"
        ]
    )

    realistic_rmse = (
        realistic_metrics[
            "tracking_rmse"
        ]
    )

    return {
        "rmse_ratio":
            realistic_rmse
            / ideal_rmse,

        "rmse_increase":
            realistic_rmse
            - ideal_rmse,
    }


def save_plots(results):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =================================
    # 1. Joint angle tracking
    # =================================

    plt.figure()

    for case, result in results.items():
        plt.plot(
            result["time"],
            result["q_true"][:, 0],
            label=f"{case} J1",
        )

    plt.plot(
        results["ideal"]["time"],
        results["ideal"]["q_des"][:, 0],
        "--",
        label="J1 desired",
    )

    plt.axvspan(
        DISTURBANCE_START,
        DISTURBANCE_END,
        alpha=0.2,
        label="disturbance",
    )

    plt.xlabel("Time [s]")
    plt.ylabel(
        "Joint 1 angle [rad]"
    )

    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "joint1_tracking.png",
        dpi=150,
    )

    plt.close()

    plt.figure()

    for case, result in results.items():
        plt.plot(
            result["time"],
            result["q_true"][:, 1],
            label=f"{case} J2",
        )

    plt.plot(
        results["ideal"]["time"],
        results["ideal"]["q_des"][:, 1],
        "--",
        label="J2 desired",
    )

    plt.axvspan(
        DISTURBANCE_START,
        DISTURBANCE_END,
        alpha=0.2,
        label="disturbance",
    )

    plt.xlabel("Time [s]")
    plt.ylabel(
        "Joint 2 angle [rad]"
    )

    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "joint2_tracking.png",
        dpi=150,
    )

    plt.close()

    # =================================
    # 2. Tracking error
    # =================================

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

    plt.axvspan(
        DISTURBANCE_START,
        DISTURBANCE_END,
        alpha=0.2,
        label="disturbance",
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
        / "tracking_error.png",
        dpi=150,
    )

    plt.close()

    # =================================
    # 3. Controller torque
    # =================================

    plt.figure()

    for case, result in results.items():
        plt.plot(
            result["time"],
            result[
                "control_torque"
            ][:, 0],
            label=f"{case} J1",
        )

        plt.plot(
            result["time"],
            result[
                "control_torque"
            ][:, 1],
            "--",
            label=f"{case} J2",
        )

    plt.axvspan(
        DISTURBANCE_START,
        DISTURBANCE_END,
        alpha=0.2,
        label="disturbance",
    )

    plt.xlabel("Time [s]")
    plt.ylabel(
        "Control torque [N m]"
    )

    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "control_torque.png",
        dpi=150,
    )

    plt.close()

    # =================================
    # 4. RMSE summary
    # =================================

    labels = list(results.keys())

    metrics = [
        calculate_metrics(
            results[case]
        )
        for case in labels
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
        / "rmse_summary.png",
        dpi=150,
    )

    plt.close()


def save_summary(
    metrics,
    degradation,
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = {
        "ideal": {
            key: value.tolist()
            for key, value
            in metrics["ideal"].items()
        },

        "realistic": {
            key: value.tolist()
            for key, value
            in metrics["realistic"].items()
        },

        "degradation": {
            key: value.tolist()
            for key, value
            in degradation.items()
        },

        "realistic_conditions": {
            "physics_hz": 500.0,
            "sensor_hz": SENSOR_HZ,
            "controller_hz":
                CONTROLLER_HZ,
            "sensor_delay_ms":
                SENSOR_DELAY_S * 1000.0,
            "counts_per_rev": 4096,
            "angle_noise_std_rad":
                0.0005,
            "velocity_cutoff_hz":
                20.0,
            "disturbance_start_s":
                DISTURBANCE_START,
            "disturbance_end_s":
                DISTURBANCE_END,
            "disturbance_torque_Nm":
                DISTURBANCE_TORQUE.tolist(),
        },
    }

    with open(
        OUTPUT_DIR / "summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    with open(
        OUTPUT_DIR / "summary.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)

        writer.writerow([
            "case",
            "joint",
            "tracking_rmse_rad",
            "max_tracking_error_rad",
            "final_tracking_error_rad",
            "max_control_torque_Nm",
            "max_total_torque_Nm",
        ])

        for case in CASES:
            for joint in range(2):
                writer.writerow([
                    case,
                    joint + 1,
                    metrics[
                        case
                    ][
                        "tracking_rmse"
                    ][joint],
                    metrics[
                        case
                    ][
                        "max_tracking_error"
                    ][joint],
                    metrics[
                        case
                    ][
                        "final_tracking_error"
                    ][joint],
                    metrics[
                        case
                    ][
                        "max_control_torque"
                    ][joint],
                    metrics[
                        case
                    ][
                        "max_total_torque"
                    ][joint],
                ])


def main():
    print(
        "=== Step 9-4 "
        "Realistic Control Benchmark ==="
    )

    results = {}
    metrics = {}

    for case in CASES:
        result = simulate(
            case
        )

        results[case] = result

        metrics[case] = (
            calculate_metrics(
                result
            )
        )

        print()
        print(
            f"--- {case} ---"
        )

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
            "Maximum control torque [N m]:",
            metrics[case][
                "max_control_torque"
            ],
        )

        print(
            "Maximum total torque [N m]:",
            metrics[case][
                "max_total_torque"
            ],
        )

    degradation = (
        calculate_degradation(
            metrics["ideal"],
            metrics["realistic"],
        )
    )

    print()
    print(
        "=== Ideal -> Realistic degradation ==="
    )

    print(
        "RMSE ratio:",
        degradation[
            "rmse_ratio"
        ],
    )

    print(
        "RMSE increase [rad]:",
        degradation[
            "rmse_increase"
        ],
    )

    save_plots(
        results
    )

    save_summary(
        metrics,
        degradation,
    )

    print()
    print(
        "Output directory:",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()