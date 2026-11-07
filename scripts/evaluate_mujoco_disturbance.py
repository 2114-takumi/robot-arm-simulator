import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "two_link_arm.xml"
OUTPUT_DIR = ROOT / "outputs" / "step_9_3"

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

SENSOR_HZ = 100.0
CONTROLLER_HZ = 100.0
DELAY_S = 0.010

DISTURBANCE_START = 2.0
DISTURBANCE_END = 2.2

# Joint 2だけに外乱
DISTURBANCE_TORQUE = np.array([
    0.0,
    1.0,
])

CASES = (
    "baseline",
    "disturbance",
)

# 回復したと判定する条件
RECOVERY_THRESHOLD = 0.001  # rad
RECOVERY_HOLD_TIME = 0.2    # s


def disturbance_torque(t, enabled):
    """指定時間だけ外乱トルクを返す."""

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

    disturbance_enabled = (
        case == "disturbance"
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

    sensor = JointEncoderSensor(
        counts_per_rev=4096,
        angle_noise_std=0.0005,
        velocity_cutoff_hz=20.0,
        seed=0,
    )

    physics_dt = model.opt.timestep

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
        DELAY_S
        / sensor_period
    )

    num_steps = (
        int(SIM_DURATION / physics_dt)
        + 1
    )

    time_log = np.zeros(
        num_steps
    )

    q_true_log = np.zeros(
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

    measurement_history = []

    q_control = Q_START.copy()
    dq_control = np.zeros(2)

    control_torque = np.zeros(2)

    for i in range(num_steps):
        t = data.time

        q_true = data.qpos.copy()

        (
            q_des,
            dq_des,
            ddq_des,
        ) = desired_trajectory(t)

        # --------------------------------
        # Sensor update
        # --------------------------------

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

        # --------------------------------
        # Controller update
        # --------------------------------

        if i % controller_steps == 0:
            history_index = (
                len(measurement_history)
                - 1
                - delay_samples
            )

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

        # --------------------------------
        # External disturbance
        # --------------------------------

        external_torque = (
            disturbance_torque(
                t,
                disturbance_enabled,
            )
        )

        # MuJoCoへ実際に入る総トルク
        total_torque = (
            control_torque
            + external_torque
        )

        # --------------------------------
        # Log
        # --------------------------------

        time_log[i] = t
        q_true_log[i] = q_true
        q_des_log[i] = q_des

        control_torque_log[i] = (
            control_torque
        )

        disturbance_log[i] = (
            external_torque
        )

        total_torque_log[i] = (
            total_torque
        )

        # --------------------------------
        # Physics update
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
        "q_des": q_des_log,
        "control_torque": control_torque_log,
        "disturbance": disturbance_log,
        "total_torque": total_torque_log,
    }


def calculate_tracking_metrics(result):
    """通常の軌道追従性能."""

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

        "max_control_torque": np.max(
            np.abs(
                result["control_torque"]
            ),
            axis=0,
        ),
    }


def calculate_recovery_time(
    time,
    deviation,
    joint,
):
    """
    外乱終了後、deviationがしきい値以内に入り、
    0.2秒間連続して収まった最初の時間を返す.
    """

    dt = time[1] - time[0]

    hold_steps = int(
        np.ceil(
            RECOVERY_HOLD_TIME / dt
        )
    )

    start_index = np.searchsorted(
        time,
        DISTURBANCE_END,
    )

    for i in range(
        start_index,
        len(time) - hold_steps + 1,
    ):
        window = deviation[
            i:i + hold_steps,
            joint,
        ]

        if np.all(
            np.abs(window)
            <= RECOVERY_THRESHOLD
        ):
            return (
                time[i]
                - DISTURBANCE_END
            )

    return None


def calculate_disturbance_metrics(
    baseline,
    disturbed,
):
    """
    外乱によってbaselineからどれだけ状態が変化したかを評価.
    """

    angle_deviation = (
        disturbed["q_true"]
        - baseline["q_true"]
    )

    control_torque_difference = (
        disturbed["control_torque"]
        - baseline["control_torque"]
    )

    recovery_times = [
        calculate_recovery_time(
            disturbed["time"],
            angle_deviation,
            joint,
        )
        for joint in range(2)
    ]

    return {
        "angle_deviation":
            angle_deviation,

        "max_disturbance_deviation":
            np.max(
                np.abs(
                    angle_deviation
                ),
                axis=0,
            ),

        "recovery_time":
            recovery_times,

        "max_extra_control_torque":
            np.max(
                np.abs(
                    control_torque_difference
                ),
                axis=0,
            ),
    }


def save_plots(
    results,
    disturbance_metrics,
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------
    # Tracking error
    # --------------------------------

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

    plt.axvspan(
        DISTURBANCE_START,
        DISTURBANCE_END,
        alpha=0.2,
        label="disturbance",
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

    # --------------------------------
    # Disturbance-induced deviation
    # --------------------------------

    deviation = disturbance_metrics[
        "angle_deviation"
    ]

    time = results[
        "disturbance"
    ]["time"]

    plt.figure()

    plt.plot(
        time,
        deviation[:, 0],
        label="Joint 1",
    )

    plt.plot(
        time,
        deviation[:, 1],
        label="Joint 2",
    )

    plt.axvspan(
        DISTURBANCE_START,
        DISTURBANCE_END,
        alpha=0.2,
        label="disturbance",
    )

    plt.axhline(
        RECOVERY_THRESHOLD,
        linestyle="--",
    )

    plt.axhline(
        -RECOVERY_THRESHOLD,
        linestyle="--",
    )

    plt.xlabel("Time [s]")
    plt.ylabel(
        "Disturbed - baseline angle [rad]"
    )

    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "disturbance_deviation.png",
        dpi=150,
    )

    plt.close()

    # --------------------------------
    # Joint 2 control torque
    # --------------------------------

    plt.figure()

    for case, result in results.items():
        plt.plot(
            result["time"],
            result[
                "control_torque"
            ][:, 1],
            label=case,
        )

    plt.axvspan(
        DISTURBANCE_START,
        DISTURBANCE_END,
        alpha=0.2,
        label="disturbance",
    )

    plt.xlabel("Time [s]")
    plt.ylabel(
        "Joint 2 control torque [N m]"
    )

    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "joint2_control_torque.png",
        dpi=150,
    )

    plt.close()


def main():
    results = {}
    tracking_metrics = {}

    print(
        "=== Step 9-3 Disturbance Evaluation ==="
    )

    print(
        "Physics: 500 Hz"
    )

    print(
        "Sensor / Controller:",
        f"{SENSOR_HZ:g} Hz",
    )

    print(
        "Sensor delay:",
        f"{DELAY_S * 1000:.0f} ms",
    )

    print(
        "Disturbance:",
        DISTURBANCE_TORQUE,
        "N m",
    )

    print(
        "Disturbance interval:",
        f"{DISTURBANCE_START:.1f}"
        f"-{DISTURBANCE_END:.1f} s",
    )

    for case in CASES:
        result = simulate(
            case
        )

        results[case] = result

        tracking_metrics[case] = (
            calculate_tracking_metrics(
                result
            )
        )

        print()
        print(
            f"--- {case} ---"
        )

        print(
            "Tracking RMSE [rad]:",
            tracking_metrics[
                case
            ]["tracking_rmse"],
        )

        print(
            "Maximum tracking error [rad]:",
            tracking_metrics[
                case
            ]["max_tracking_error"],
        )

        print(
            "Final tracking error [rad]:",
            tracking_metrics[
                case
            ]["final_tracking_error"],
        )

        print(
            "Maximum control torque [N m]:",
            tracking_metrics[
                case
            ]["max_control_torque"],
        )

    disturbance_metrics = (
        calculate_disturbance_metrics(
            results["baseline"],
            results["disturbance"],
        )
    )

    print()
    print(
        "=== Disturbance-specific metrics ==="
    )

    print(
        "Maximum disturbance-induced "
        "angle deviation [rad]:",
        disturbance_metrics[
            "max_disturbance_deviation"
        ],
    )

    print(
        "Recovery time [s]:",
        disturbance_metrics[
            "recovery_time"
        ],
    )

    print(
        "Maximum extra controller torque [N m]:",
        disturbance_metrics[
            "max_extra_control_torque"
        ],
    )

    save_plots(
        results,
        disturbance_metrics,
    )

    print()
    print(
        "Output directory:",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()