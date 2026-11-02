"""PD＋重力補償と計算トルク制御を比較する。"""

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
from scripts.run_pd_gravity_demo import settling_times
from robot_arm.dynamics import integrate_dynamics
from scripts.run_coupled_dynamics_demo import physical_parameters

# PDとCTCのゲインは単位が異なる。
KP = np.array([30.0, 20.0])
KD = np.array([8.0, 5.0])
TARGET = np.array([0.3, -0.4])
DT = 0.0025
DURATION = 4.0
CTC_KP = np.array([25.0, 25.0])
CTC_KD = np.array([10.0, 10.0])
CASES = ("PD_gravity", "computed_torque")


def simulate(case):
    """ケース名から時刻[s]、状態、入力トルク[N·m]、角度誤差[rad]を返す。"""
    if case not in CASES:
        raise ValueError("Unknown control case")
    parameters = physical_parameters()
    time = np.arange(round(DURATION / DT) + 1) * DT
    states = np.zeros((len(time), 4))
    torques = np.zeros((len(time), 2))
    for index in range(len(time)):
        q, omega = states[index, :2], states[index, 2:]
        if case == "PD_gravity":
            torques[index] = pd_gravity_torque(
                q, omega, TARGET, Kp=KP, Kd=KD, L1=parameters["L1"],
                m1=parameters["m1"], m2=parameters["m2"],
                lc1=parameters["lc1"], lc2=parameters["lc2"], g=9.81
            )
        else:
            torques[index] = computed_torque(q, omega, TARGET, CTC_KP, CTC_KD, parameters)
        if index < len(time) - 1:
            # 制御入力はステップ開始時に更新し、RK2の中間段階では一定。
            states[index + 1] = integrate_dynamics(
                *states[index], *torques[index], DT, **parameters,
                b1=0.0, b2=0.0, g=9.81, method="rk2"
            )
    errors = TARGET - states[:, :2]
    if not np.all(np.isfinite(np.column_stack((time, states, torques, errors)))):
        raise ValueError("Nonfinite simulation data")
    return time, states, torques, errors


def save_results(results, output):
    """CSV・比較PNG・指標JSONを新規出力先へ保存し、指標の辞書を返す。"""
    output.mkdir(parents=True, exist_ok=False)
    summary = dict(
        gains={
            "PD_gravity": dict(Kp=KP.tolist(), Kd=KD.tolist(), Kp_unit="N m/rad", Kd_unit="N m s/rad"),
            "computed_torque": dict(Kp=CTC_KP.tolist(), Kd=CTC_KD.tolist(), Kp_unit="1/s^2", Kd_unit="1/s"),
        },
        comparison_note="Methods with specified gains of different units; not equal-gain superiority",
        acceleration_note="q_ddot=v at updates with the same frictionless model; not throughout held-torque RK2 steps",
        convergence_definition="Both joints settle within the 2% band during the observed 4 s; not an asymptotic proof",
        gains_adjusted=False, target_rad=TARGET.tolist(), target_velocity_rad_s=[0, 0], initial_state=[0, 0, 0, 0],
        dt_s=DT, duration_s=DURATION, method="rk2 midpoint", parameters=physical_parameters(),
        g=9.81, b1=0.0, b2=0.0,
        error_definition="q_target - q, without angle wrapping",
        settling_definition="Per joint: abs(error) <= 0.02*max(abs(initial_error),0.01 rad) for all remaining samples; null if unsettled within 4 s",
        final_error_note="Finite-horizon reference for steady-state bias; not a proven asymptotic value",
        torque_sampling="start of each step, held constant within RK2; final row is not applied",
    )
    for case, (time, states, torques, errors) in results.items():
        np.savetxt(
            output / (case + ".csv"), np.column_stack((time, states, torques, errors)),
            delimiter=",", comments="",
            header="time_s,theta1_rad,theta2_rad,omega1_rad_s,omega2_rad_s,tau1_Nm,tau2_Nm,error1_rad,error2_rad",
        )
        settled, thresholds = settling_times(time, errors)
        summary[case] = dict(
            samples=len(time), converged=all(value is not None for value in settled),
            converged_per_joint=[value is not None for value in settled],
            final_angle_error_rad=errors[-1].tolist(),
            max_abs_angle_error_rad=np.max(np.abs(errors), axis=0).tolist(),
            max_abs_velocity_rad_s=np.max(np.abs(states[:, 2:]), axis=0).tolist(),
            max_abs_torque_Nm=np.max(np.abs(torques), axis=0).tolist(),
            settling_time_s=settled, settling_threshold_rad=thresholds.tolist(),
        )
    for filename, data_index, columns, ylabel in (
        ("angles.png", 1, (0, 1), "Angle [rad]"),
        ("angular_velocities.png", 1, (2, 3), "Angular velocity [rad/s]"),
        ("errors.png", 3, (0, 1), "Target - angle [rad]"),
        ("torques.png", 2, (0, 1), "Torque [N m]"),
    ):
        fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        for joint, (axis, column) in enumerate(zip(axes, columns)):
            for case, data in results.items():
                axis.plot(data[0], data[data_index][:, column], label=case)
            if filename == "angles.png":
                axis.axhline(TARGET[joint], color="black", linestyle="--", label="Target")
            elif filename in ("errors.png", "angular_velocities.png"):
                axis.axhline(0, color="black", linestyle="--", label="Target error" if filename == "errors.png" else "Target velocity")
            axis.set_title(f"Joint {joint + 1}; target angle = {TARGET[joint]:.1f} rad")
            axis.set_ylabel(ylabel)
            axis.grid(True)
            axis.legend()
        axes[-1].set_xlabel("Time [s]")
        fig.tight_layout()
        fig.savefig(output / filename, dpi=150)
        plt.close(fig)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    """比較実験を実行する。--outputで再実行時の新規保存先を指定する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "step_8_3")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    results = {case: simulate(case) for case in CASES}
    print(json.dumps(save_results(results, args.output), indent=2))
    print("Output:", args.output.resolve())


if __name__ == "__main__":
    main()
