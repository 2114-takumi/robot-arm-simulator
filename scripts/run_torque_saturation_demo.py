"""既存のPD＋重力補償・計算トルク制御に外部トルク制限を適用する。"""

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
from robot_arm.actuator import saturate_torque
from robot_arm.controllers import pd_gravity_torque
from robot_arm.computed_torque import computed_torque
from robot_arm.dynamics import gravity_vector, integrate_dynamics, forward_dynamics
from scripts.run_coupled_dynamics_demo import physical_parameters
from scripts.run_pd_gravity_demo import settling_times
from scripts.run_computed_torque_demo import KP, KD, CTC_KP, CTC_KD, TARGET, DT, DURATION

CONTROLLERS = ("PD_gravity", "computed_torque")
LIMITS = {"unlimited": None, "moderate": np.array([20., 4.]), "strict": np.array([10., 2.])}
CASES = tuple((controller, limit) for controller in CONTROLLERS for limit in LIMITS)
SAT_ATOL = 1e-12  # N·m
SAT_RTOL = 1e-12
CSV_HEADER = (
    "time_s,theta1_rad,theta2_rad,omega1_rad_s,omega2_rad_s,"
    "tau_command1_Nm,tau_command2_Nm,tau_actual1_Nm,tau_actual2_Nm,error1_rad,error2_rad"
)


def simulate(controller, limit):
    """時刻、状態、要求トルク、実トルク、角度誤差（終端を含む）を返す。"""
    if controller not in CONTROLLERS or limit not in LIMITS:
        raise ValueError("Unknown controller or torque limit")
    parameters = physical_parameters()
    gravity_params = {k: parameters[k] for k in ("L1", "m1", "m2", "lc1", "lc2")}
    time = np.arange(round(DURATION / DT) + 1) * DT
    states = np.zeros((len(time), 4))
    commands = np.zeros((len(time), 2))
    actual = np.zeros_like(commands)
    for i in range(len(time)):
        q, omega = states[i, :2], states[i, 2:]
        if controller == "PD_gravity":
            commands[i] = pd_gravity_torque(q, omega, TARGET, Kp=KP, Kd=KD, **gravity_params, g=9.81)
        else:
            commands[i] = computed_torque(q, omega, TARGET, CTC_KP, CTC_KD, parameters)
        actual[i] = commands[i] if LIMITS[limit] is None else saturate_torque(commands[i], LIMITS[limit])
        if i < len(time) - 1:
            # 飽和後のトルクをステップ開始時に確定し、RK2中点でも保持する。
            states[i + 1] = integrate_dynamics(
                *states[i], *actual[i], DT, **parameters, b1=0., b2=0., g=9.81, method="rk2"
            )
    errors = TARGET - states[:, :2]
    if not np.all(np.isfinite(np.column_stack((time, states, commands, actual, errors)))):
        raise ValueError("Nonfinite simulation data")
    return time, states, commands, actual, errors


def saturation_statistics(commands, actual):
    """終端行（未適用）を除く制御ステップの飽和回数と割合を返す。

    abs(command-actual) > atol + rtol*abs(command) を飽和と判定する。
    関節別と、いずれかの関節が飽和したステップを区別する。
    """
    mask = np.abs(commands[:-1] - actual[:-1]) > SAT_ATOL + SAT_RTOL * np.abs(commands[:-1])
    return dict(
        control_steps=len(mask), per_joint_steps=np.sum(mask, axis=0).tolist(),
        per_joint_fraction=np.mean(mask, axis=0).tolist(),
        any_joint_steps=int(np.count_nonzero(np.any(mask, axis=1))),
        any_joint_fraction=float(np.mean(np.any(mask, axis=1))),
    )


def target_equilibrium():
    """静止時はtau=g(q_target)が必要。制限で実現可能かを判定する。"""
    p = physical_parameters()
    gravity = gravity_vector(*TARGET, **{k: p[k] for k in ("L1", "m1", "m2", "lc1", "lc2")}, g=9.81)
    result = dict(required_gravity_torque_Nm=gravity.tolist(), limits={})
    for name, bound in LIMITS.items():
        actual = gravity.copy() if bound is None else saturate_torque(gravity, bound)
        possible = np.ones(2, dtype=bool) if bound is None else np.abs(gravity) <= bound
        result["limits"][name] = dict(
            feasible_per_joint=possible.tolist(), static_equilibrium_possible=bool(np.all(possible)),
            actual_torque_at_target_Nm=actual.tolist(),
            acceleration_at_target_rad_s2=forward_dynamics(*TARGET, 0., 0., *actual, **p, g=9.81).tolist(),
        )
    return result


def save_results(results, output):
    """6条件のCSV、比較PNG、summary.jsonを新規ディレクトリへ保存する。"""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    summary = dict(
        gains={
            "PD_gravity": dict(Kp=KP.tolist(), Kd=KD.tolist(), Kp_unit="N m/rad", Kd_unit="N m s/rad"),
            "computed_torque": dict(Kp=CTC_KP.tolist(), Kd=CTC_KD.tolist(), Kp_unit="1/s^2", Kd_unit="1/s"),
        },
        torque_limits_Nm={k: None if v is None else v.tolist() for k, v in LIMITS.items()},
        initial_state=[0., 0., 0., 0.], target_rad=TARGET.tolist(), target_velocity_rad_s=[0., 0.],
        dt_s=DT, duration_s=DURATION, method="rk2 midpoint", parameters=physical_parameters(),
        g=9.81, b1=0., b2=0., gains_adjusted=False,
        torque_sampling="Step-start actual torque held within RK2; final row is diagnostic, not applied",
        torque_maximum_definition="Maximum absolute torque over applied control steps, excluding final row",
        saturation_definition="abs(command-actual) > atol + rtol*abs(command); exclude final row; denominator=1600 control steps",
        saturation_atol_Nm=SAT_ATOL, saturation_rtol=SAT_RTOL,
        error_definition="q_target - q, without angle wrapping",
        settling_definition="Per joint: abs(error) <= 0.02*max(abs(initial_error),0.01 rad) for all remaining samples; null if unsettled within 4 s",
        convergence_definition="Both joints settled in the observed 4 s; no claim of infinite-time convergence",
        comparison_note="Specified methods and gains of different units; not equal-gain superiority",
        target_equilibrium=target_equilibrium(), cases={},
    )
    for (controller, limit), data in results.items():
        time, states, commands, actual, errors = data
        name = controller + "_" + limit
        np.savetxt(output / (name + ".csv"), np.column_stack(data), delimiter=",", comments="", header=CSV_HEADER)
        settled, thresholds = settling_times(time, errors)
        summary["cases"][name] = dict(
            samples=len(time), final_angle_error_rad=errors[-1].tolist(),
            max_abs_angle_error_rad=np.max(np.abs(errors), axis=0).tolist(),
            max_abs_velocity_rad_s=np.max(np.abs(states[:, 2:]), axis=0).tolist(),
            max_abs_command_torque_Nm=np.max(np.abs(commands[:-1]), axis=0).tolist(),
            max_abs_actual_torque_Nm=np.max(np.abs(actual[:-1]), axis=0).tolist(),
            saturation=saturation_statistics(commands, actual), settling_time_s=settled,
            settling_threshold_rad=thresholds.tolist(),
            converged_per_joint=[value is not None for value in settled],
            converged=all(value is not None for value in settled),
        )
    save_plots(results, output)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return summary


def save_plots(results, output):
    """状態は制御器別、トルクは6条件別に配置して読み分けられるようにする。"""
    for filename, data_index, columns, ylabel in (
        ("angles.png", 1, (0, 1), "Angle [rad]"),
        ("angular_velocities.png", 1, (2, 3), "Angular velocity [rad/s]"),
        ("errors.png", 4, (0, 1), "Target - angle [rad]"),
    ):
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
        for col, controller in enumerate(CONTROLLERS):
            for joint in (0, 1):
                ax = axes[joint, col]
                for limit in LIMITS:
                    data = results[controller, limit]
                    ax.plot(data[0], data[data_index][:, columns[joint]], label=limit)
                ax.axhline(TARGET[joint] if filename == "angles.png" else 0., color="black", linestyle="--", label="Target")
                ax.set_title(f"{controller}, J{joint + 1}; target={TARGET[joint]} rad")
                ax.set_ylabel(ylabel)
                ax.set_xlabel("Time [s]")
                ax.grid(True)
                ax.legend()
        fig.tight_layout()
        fig.savefig(output / filename, dpi=150)
        plt.close(fig)
    fig, axes = plt.subplots(6, 2, figsize=(14, 20), sharex=True)
    for row, (controller, limit) in enumerate(CASES):
        time, _, commands, actual, _ = results[controller, limit]
        for joint in (0, 1):
            ax = axes[row, joint]
            ax.plot(time, commands[:, joint], label="Command", color="tab:blue", linestyle="--")
            ax.plot(time, actual[:, joint], label="Actual", color="tab:orange", alpha=0.8)
            if LIMITS[limit] is not None:
                bound = LIMITS[limit][joint]
                ax.axhline(bound, color="gray", linestyle=":", label="Limits")
                ax.axhline(-bound, color="gray", linestyle=":")
            ax.set_title(f"{controller}, {limit}, J{joint + 1}")
            ax.set_ylabel("Torque [N m]")
            ax.grid(True)
            ax.legend()
    for ax in axes[-1]:
        ax.set_xlabel("Time [s]")
    fig.tight_layout()
    fig.savefig(output / "torques.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "step_8_4")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    results = {case: simulate(*case) for case in CASES}
    print(json.dumps(save_results(results, args.output), indent=2))
    print("Output:", args.output.resolve())


if __name__ == "__main__":
    main()
