import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "step_8_31"

sys.path.insert(0, str(ROOT))


# Step 8-22
from scripts.evaluate_mujoco_trajectory_tracking import (
    simulate as simulate_tracking,
    calculate_metrics as tracking_metrics,
)

# Step 8-23
from scripts.evaluate_mujoco_model_mismatch import (
    simulate as simulate_mismatch,
    calculate_metrics as mismatch_metrics,
)

# Step 8-25
from scripts.evaluate_mujoco_ctc_integral import (
    simulate as simulate_integral,
    calculate_metrics as integral_metrics,
)

# Step 8-27
from scripts.evaluate_mujoco_anti_windup import (
    simulate as simulate_anti_windup,
    calculate_metrics as anti_windup_metrics,
)

# Step 8-30
from scripts.compare_custom_vs_mujoco import (
    simulate_custom,
    simulate_mujoco,
    tracking_metrics as cross_tracking_metrics,
    cross_validation_metrics,
)


def to_list(value):
    """NumPy配列・スカラーをJSON保存可能な形式に変換する."""

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, dict):
        return {
            key: to_list(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            to_list(item)
            for item in value
        ]

    return value


def run_nominal_tracking():
    """Step 8-22: nominal CTC trajectory tracking."""

    (
        time_log,
        q_log,
        dq_log,
        q_des_log,
        dq_des_log,
        tau_log,
    ) = simulate_tracking()

    metrics = tracking_metrics(
        q_log,
        dq_log,
        q_des_log,
        dq_des_log,
        tau_log,
    )

    return {
        "position_rmse_rad":
            metrics["rmse_position"],

        "max_position_error_rad":
            metrics["max_position_error"],

        "velocity_rmse_rad_s":
            metrics["rmse_velocity"],

        "max_torque_Nm":
            metrics["max_torque"],

        "final_position_error_rad":
            metrics["final_position_error"],
    }


def run_model_mismatch():
    """Step 8-23: CTC model mismatch +20%."""

    result = simulate_mismatch(
        1.2
    )

    metrics = mismatch_metrics(
        result
    )

    return {
        "position_rmse_rad":
            metrics["position_rmse"],

        "max_position_error_rad":
            metrics["max_position_error"],

        "velocity_rmse_rad_s":
            metrics["velocity_rmse"],

        "max_torque_Nm":
            metrics["max_torque"],

        "final_position_error_rad":
            metrics["final_position_error"],
    }


def run_ctc_integral_comparison():
    """Step 8-25: CTC vs CTC+I."""

    result_ctc = simulate_integral(
        "CTC"
    )

    result_ctc_i = simulate_integral(
        "CTC_I"
    )

    metrics_ctc = integral_metrics(
        result_ctc
    )

    metrics_ctc_i = integral_metrics(
        result_ctc_i
    )

    return {
        "CTC": {
            "position_rmse_rad":
                metrics_ctc["position_rmse"],

            "final_position_error_rad":
                metrics_ctc["final_position_error"],

            "max_torque_Nm":
                metrics_ctc["max_torque"],
        },

        "CTC_I": {
            "position_rmse_rad":
                metrics_ctc_i["position_rmse"],

            "final_position_error_rad":
                metrics_ctc_i["final_position_error"],

            "max_torque_Nm":
                metrics_ctc_i["max_torque"],

            "final_integral_rad_s":
                metrics_ctc_i["final_integral"],
        },
    }


def run_anti_windup_comparison():
    """Step 8-27: standard vs global conditional."""

    standard = simulate_anti_windup(
        "standard"
    )

    conditional = simulate_anti_windup(
        "conditional"
    )

    metrics_standard = (
        anti_windup_metrics(
            standard
        )
    )

    metrics_conditional = (
        anti_windup_metrics(
            conditional
        )
    )

    return {
        "standard": {
            "position_rmse_rad":
                metrics_standard["position_rmse"],

            "max_integral_rad_s":
                metrics_standard["max_integral"],

            "max_command_torque_Nm":
                metrics_standard["max_command_torque"],

            "saturation_fraction":
                metrics_standard["saturation_fraction"],
        },

        "conditional": {
            "position_rmse_rad":
                metrics_conditional["position_rmse"],

            "max_integral_rad_s":
                metrics_conditional["max_integral"],

            "max_command_torque_Nm":
                metrics_conditional["max_command_torque"],

            "saturation_fraction":
                metrics_conditional["saturation_fraction"],

            "integral_stopped_fraction":
                metrics_conditional[
                    "integral_stopped_fraction"
                ],
        },
    }


def run_cross_validation():
    """Step 8-30: Custom simulator vs MuJoCo."""

    # MuJoCoと同じdt
    mujoco_nominal = simulate_mujoco(
        1.0
    )

    dt = mujoco_nominal["dt"]

    custom_nominal = simulate_custom(
        1.0,
        dt,
    )

    nominal_cross = (
        cross_validation_metrics(
            custom_nominal,
            mujoco_nominal,
        )
    )

    # m2 +20%
    custom_mismatch = simulate_custom(
        1.2,
        dt,
    )

    mujoco_mismatch = simulate_mujoco(
        1.2
    )

    mismatch_cross = (
        cross_validation_metrics(
            custom_mismatch,
            mujoco_mismatch,
        )
    )

    custom_metrics = (
        cross_tracking_metrics(
            custom_mismatch
        )
    )

    mujoco_metrics = (
        cross_tracking_metrics(
            mujoco_mismatch
        )
    )

    return {
        "nominal": {
            "angle_difference_rmse_rad":
                nominal_cross[
                    "angle_difference_rmse"
                ],

            "max_angle_difference_rad":
                nominal_cross[
                    "max_angle_difference"
                ],

            "torque_difference_rmse_Nm":
                nominal_cross[
                    "torque_difference_rmse"
                ],
        },

        "m2_plus_20": {
            "angle_difference_rmse_rad":
                mismatch_cross[
                    "angle_difference_rmse"
                ],

            "max_angle_difference_rad":
                mismatch_cross[
                    "max_angle_difference"
                ],

            "custom_position_rmse_rad":
                custom_metrics[
                    "position_rmse"
                ],

            "mujoco_position_rmse_rad":
                mujoco_metrics[
                    "position_rmse"
                ],

            "torque_difference_rmse_Nm":
                mismatch_cross[
                    "torque_difference_rmse"
                ],
        },
    }


def calculate_improvements(
    summary,
):
    """代表的な改善率を計算."""

    ctc = np.array(
        summary[
            "ctc_vs_ctc_i"
        ]["CTC"][
            "position_rmse_rad"
        ],
        dtype=float,
    )

    ctc_i = np.array(
        summary[
            "ctc_vs_ctc_i"
        ]["CTC_I"][
            "position_rmse_rad"
        ],
        dtype=float,
    )

    integral_rmse_reduction = (
        (ctc - ctc_i)
        / ctc
        * 100.0
    )

    standard_command = np.array(
        summary[
            "anti_windup"
        ]["standard"][
            "max_command_torque_Nm"
        ],
        dtype=float,
    )

    conditional_command = np.array(
        summary[
            "anti_windup"
        ]["conditional"][
            "max_command_torque_Nm"
        ],
        dtype=float,
    )

    command_reduction = (
        (standard_command
         - conditional_command)
        / standard_command
        * 100.0
    )

    return {
        "ctc_i_rmse_reduction_percent":
            integral_rmse_reduction,

        "conditional_command_reduction_percent":
            command_reduction,
    }


def save_summary_plot(
    summary,
):
    """代表的な指標だけを図にまとめる."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------
    # 1. Nominal vs model mismatch RMSE
    # ---------------------------------

    nominal = np.array(
        summary[
            "nominal_tracking"
        ][
            "position_rmse_rad"
        ]
    )

    mismatch = np.array(
        summary[
            "model_mismatch_plus_20"
        ][
            "position_rmse_rad"
        ]
    )

    x = np.arange(2)
    width = 0.35

    plt.figure()

    plt.bar(
        x - width / 2,
        nominal,
        width,
        label="Nominal",
    )

    plt.bar(
        x + width / 2,
        mismatch,
        width,
        label="m2 +20%",
    )

    plt.xticks(
        x,
        ["Joint 1", "Joint 2"],
    )

    plt.ylabel(
        "Position RMSE [rad]"
    )

    plt.legend()
    plt.grid(
        axis="y"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "tracking_rmse_summary.png",
        dpi=150,
    )

    plt.close()

    # ---------------------------------
    # 2. CTC vs CTC+I final error
    # ---------------------------------

    ctc_error = np.abs(
        np.array(
            summary[
                "ctc_vs_ctc_i"
            ]["CTC"][
                "final_position_error_rad"
            ]
        )
    )

    ctc_i_error = np.abs(
        np.array(
            summary[
                "ctc_vs_ctc_i"
            ]["CTC_I"][
                "final_position_error_rad"
            ]
        )
    )

    plt.figure()

    plt.bar(
        x - width / 2,
        ctc_error,
        width,
        label="CTC",
    )

    plt.bar(
        x + width / 2,
        ctc_i_error,
        width,
        label="CTC+I",
    )

    plt.xticks(
        x,
        ["Joint 1", "Joint 2"],
    )

    plt.ylabel(
        "Absolute final error [rad]"
    )

    plt.legend()
    plt.grid(
        axis="y"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "integral_effect_summary.png",
        dpi=150,
    )

    plt.close()

    # ---------------------------------
    # 3. Custom vs MuJoCo max difference
    # ---------------------------------

    nominal_difference = np.array(
        summary[
            "cross_validation"
        ]["nominal"][
            "max_angle_difference_rad"
        ]
    )

    mismatch_difference = np.array(
        summary[
            "cross_validation"
        ]["m2_plus_20"][
            "max_angle_difference_rad"
        ]
    )

    plt.figure()

    plt.bar(
        x - width / 2,
        nominal_difference,
        width,
        label="Nominal",
    )

    plt.bar(
        x + width / 2,
        mismatch_difference,
        width,
        label="m2 +20%",
    )

    plt.xticks(
        x,
        ["Joint 1", "Joint 2"],
    )

    plt.ylabel(
        "Max Custom-MuJoCo difference [rad]"
    )

    plt.legend()
    plt.grid(
        axis="y"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "cross_validation_summary.png",
        dpi=150,
    )

    plt.close()


def save_csv(
    summary,
):
    """README等で使いやすい代表指標をCSV保存."""

    nominal = summary[
        "nominal_tracking"
    ]

    mismatch = summary[
        "model_mismatch_plus_20"
    ]

    ctc = summary[
        "ctc_vs_ctc_i"
    ]["CTC"]

    ctc_i = summary[
        "ctc_vs_ctc_i"
    ]["CTC_I"]

    cross = summary[
        "cross_validation"
    ]

    rows = []

    for joint in range(2):
        rows.append(
            [
                joint + 1,
                nominal[
                    "position_rmse_rad"
                ][joint],
                mismatch[
                    "position_rmse_rad"
                ][joint],
                ctc[
                    "final_position_error_rad"
                ][joint],
                ctc_i[
                    "final_position_error_rad"
                ][joint],
                cross[
                    "nominal"
                ][
                    "max_angle_difference_rad"
                ][joint],
                cross[
                    "m2_plus_20"
                ][
                    "max_angle_difference_rad"
                ][joint],
            ]
        )

    rows = np.asarray(
        rows,
        dtype=float,
    )

    np.savetxt(
        OUTPUT_DIR
        / "validation_summary.csv",
        rows,
        delimiter=",",
        comments="",
        header=(
            "joint,"
            "nominal_rmse_rad,"
            "m2_plus20_rmse_rad,"
            "ctc_final_error_rad,"
            "ctc_i_final_error_rad,"
            "custom_mujoco_nominal_max_diff_rad,"
            "custom_mujoco_mismatch_max_diff_rad"
        ),
    )


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=== Step 8-31 "
        "MuJoCo Validation Summary ==="
    )

    print(
        "Running nominal tracking..."
    )

    nominal = (
        run_nominal_tracking()
    )

    print(
        "Running model mismatch..."
    )

    mismatch = (
        run_model_mismatch()
    )

    print(
        "Running CTC vs CTC+I..."
    )

    integral = (
        run_ctc_integral_comparison()
    )

    print(
        "Running anti-windup comparison..."
    )

    anti_windup = (
        run_anti_windup_comparison()
    )

    print(
        "Running custom vs MuJoCo..."
    )

    cross_validation = (
        run_cross_validation()
    )

    summary = {
        "nominal_tracking":
            nominal,

        "model_mismatch_plus_20":
            mismatch,

        "ctc_vs_ctc_i":
            integral,

        "anti_windup":
            anti_windup,

        "cross_validation":
            cross_validation,
    }

    summary["improvements"] = (
        calculate_improvements(
            summary
        )
    )

    summary_json = (
        to_list(summary)
    )

    with open(
        OUTPUT_DIR
        / "summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary_json,
            f,
            indent=2,
        )

    save_summary_plot(
        summary_json
    )

    save_csv(
        summary_json
    )

    print()
    print(
        "=== Key Results ==="
    )

    print(
        "Nominal position RMSE [rad]:",
        nominal[
            "position_rmse_rad"
        ],
    )

    print(
        "m2 +20% position RMSE [rad]:",
        mismatch[
            "position_rmse_rad"
        ],
    )

    print(
        "CTC final error [rad]:",
        integral[
            "CTC"
        ][
            "final_position_error_rad"
        ],
    )

    print(
        "CTC+I final error [rad]:",
        integral[
            "CTC_I"
        ][
            "final_position_error_rad"
        ],
    )

    print(
        "Custom vs MuJoCo nominal "
        "max angle difference [rad]:",
        cross_validation[
            "nominal"
        ][
            "max_angle_difference_rad"
        ],
    )

    print(
        "Custom vs MuJoCo mismatch "
        "max angle difference [rad]:",
        cross_validation[
            "m2_plus_20"
        ][
            "max_angle_difference_rad"
        ],
    )

    print()
    print(
        "Output directory:",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()