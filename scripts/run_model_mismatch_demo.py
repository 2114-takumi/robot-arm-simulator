"""Step 8-8: 制御器の想定モデルを固定し、第2リンクの質量・慣性誤差を比較。"""

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from robot_arm.controllers import pd_gravity_torque
from robot_arm.computed_torque import computed_torque
from robot_arm.dynamics import integrate_dynamics
from scripts.run_coupled_dynamics_demo import physical_parameters
from scripts.run_computed_torque_demo import KP, KD, CTC_KP, CTC_KD, TARGET, DT, DURATION
from scripts.run_pd_gravity_demo import settling_times

CONTROLLERS = ("PD_gravity", "computed_torque")
MODELS = ("baseline", "mismatch")
CASES = tuple((controller, model) for controller in CONTROLLERS for model in MODELS)
CSV_HEADER = "time_s,theta1_rad,theta2_rad,omega1_rad_s,omega2_rad_s,tau1_Nm,tau2_Nm,error1_rad,error2_rad"


def model_parameters(model):
    """互いに独立した辞書を返す。実機のm2とI2だけに20%誤差を与える。"""
    if model not in MODELS:
        raise ValueError("Unknown model case")
    controller_params = physical_parameters()
    plant_params = physical_parameters()
    if model == "mismatch":
        plant_params["m2"] *= 1.2
        plant_params["I2"] *= 1.2
    return controller_params, plant_params


def simulate(controller, model):
    """時刻、状態、トルク、角度誤差を返す。終端のトルクは未適用。"""
    if controller not in CONTROLLERS:
        raise ValueError("Unknown controller")
    controller_params, plant_params = model_parameters(model)
    gravity_params = {key: controller_params[key] for key in ("L1", "m1", "m2", "lc1", "lc2")}
    time = np.arange(round(DURATION / DT) + 1) * DT
    states = np.zeros((len(time), 4))
    torques = np.zeros((len(time), 2))
    for i in range(len(time)):
        q, omega = states[i, :2], states[i, 2:]
        if controller == "PD_gravity":
            torques[i] = pd_gravity_torque(q, omega, TARGET, Kp=KP, Kd=KD, **gravity_params, g=9.81)
        else:
            torques[i] = computed_torque(q, omega, TARGET, CTC_KP, CTC_KD, controller_params)
        if i < len(time) - 1:
            # 同一ステップのRK2中点でも、このトルクを保持する。
            states[i + 1] = integrate_dynamics(
                *states[i], *torques[i], DT, **plant_params,
                b1=0., b2=0., g=9.81, method="rk2",
            )
    errors = TARGET - states[:, :2]
    if not np.all(np.isfinite(np.column_stack((time, states, torques, errors)))):
        raise ValueError("Nonfinite simulation data")
    return time, states, torques, errors


def save_results(results, output):
    """新規保存先に4 CSV、3 PNG、JSONを保存する。既存保存先は拒否する。"""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    summary = dict(
        dt_s=DT, duration_s=DURATION, initial_state=[0., 0., 0., 0.],
        target_rad=TARGET.tolist(), target_velocity_rad_s=[0., 0.],
        method="rk2 midpoint", b1=0., b2=0., g=9.81,
        gains={
            "PD_gravity": dict(Kp=KP.tolist(), Kd=KD.tolist(), Kp_unit="N m/rad", Kd_unit="N m s/rad"),
            "computed_torque": dict(Kp=CTC_KP.tolist(), Kd=CTC_KD.tolist(), Kp_unit="1/s^2", Kd_unit="1/s"),
        },
        comparison_note="PD and CTC gains differ in units and values; this compares specified configurations, not simple controller superiority.",
        final_error_note="Error at 4 seconds only; not a proven asymptotic steady-state bias.",
        error_definition="q_target - q, without angle wrapping",
        settling_definition="Per joint: abs(error) <= 0.02*max(abs(initial_error),0.01 rad) for all remaining samples; null if unsettled within 4 s",
        trajectory_difference_definition="Mismatch minus baseline for the same controller at matching times; per-joint maximum absolute difference and signed final difference",
        torque_sampling="Step-start torque held within RK2; final torque row is diagnostic, excluded from torque maxima",
        cases={}, trajectory_differences={},
    )
    for (controller, model), data in results.items():
        time, states, torques, errors = data
        name = controller + "_" + model
        np.savetxt(output / (name + ".csv"), np.column_stack(data), delimiter=",", header=CSV_HEADER, comments="")
        controller_params, plant_params = model_parameters(model)
        settled, thresholds = settling_times(time, errors)
        summary["cases"][name] = dict(
            controller_params=controller_params, plant_params=plant_params, samples=len(time),
            final_angle_error_rad=errors[-1].tolist(), settling_time_s=settled,
            settling_threshold_rad=thresholds.tolist(),
            max_abs_torque_Nm=np.max(np.abs(torques[:-1]), axis=0).tolist(),
        )
    for controller in CONTROLLERS:
        baseline = results[controller, "baseline"]
        mismatch = results[controller, "mismatch"]
        np.testing.assert_array_equal(baseline[0], mismatch[0])
        delta = mismatch[1] - baseline[1]
        summary["trajectory_differences"][controller] = dict(
            max_abs_angle_difference_rad=np.max(np.abs(delta[:, :2]), axis=0).tolist(),
            max_abs_velocity_difference_rad_s=np.max(np.abs(delta[:, 2:]), axis=0).tolist(),
            final_angle_difference_rad=delta[-1, :2].tolist(),
        )
    for filename, index, ylabel in (("angles.png", 1, "Angle [rad]"), ("errors.png", 3, "Target - angle [rad]"), ("torques.png", 2, "Torque [N m]")):
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
        for col, controller in enumerate(CONTROLLERS):
            for joint in (0, 1):
                ax = axes[joint, col]
                for model in MODELS:
                    data = results[controller, model]
                    ax.plot(data[0], data[index][:, joint], label=model)
                if filename != "torques.png":
                    ax.axhline(TARGET[joint] if filename == "angles.png" else 0., color="black", linestyle="--", label="Target")
                ax.set_title(f"{controller}, J{joint + 1}; target={TARGET[joint]} rad")
                ax.set_xlabel("Time [s]")
                ax.set_ylabel(ylabel)
                ax.grid(True)
                ax.legend()
        fig.tight_layout()
        fig.savefig(output / filename, dpi=150)
        plt.close(fig)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "step_8_8")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    results = {case: simulate(*case) for case in CASES}
    print(json.dumps(save_results(results, args.output), indent=2))
    print("Output:", args.output.resolve())


if __name__ == "__main__":
    main()
