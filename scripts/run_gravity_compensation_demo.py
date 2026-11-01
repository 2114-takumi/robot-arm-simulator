"""RK2で重力補償あり・なしの静止姿勢維持を比較する独立実験。"""

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
from robot_arm.dynamics import gravity_vector, integrate_dynamics
from scripts.run_coupled_dynamics_demo import physical_parameters


def compensation_torque(theta1, theta2, parameters):
    """現在角度 [rad] から静止時に必要な重力補償トルク [N·m] を返す。"""
    return gravity_vector(
        theta1, theta2, L1=parameters["L1"], m1=parameters["m1"],
        m2=parameters["m2"], lc1=parameters["lc1"], lc2=parameters["lc2"], g=9.81
    )


def simulate(compensated):
    """2秒間の時刻 [s]、状態 [rad, rad, rad/s, rad/s]、トルク [N·m] を返す。"""
    dt = 0.0025
    time = np.arange(801) * dt
    states = np.zeros((len(time), 4))
    torques = np.zeros((len(time), 2))
    parameters = physical_parameters()
    for index in range(len(time)):
        # 各時刻の現在姿勢から補償量を求める。なしのケースはゼロのまま。
        if compensated:
            torques[index] = compensation_torque(*states[index, :2], parameters)
        if index < len(time) - 1:
            # 入力はこの1ステップ中一定。RK2中間段階で制御を再計算しない。
            states[index + 1] = integrate_dynamics(
                *states[index], *torques[index], dt, **parameters,
                b1=0.0, b2=0.0, g=9.81, method="rk2"
            )
    if not np.all(np.isfinite(states)) or not np.all(np.isfinite(torques)):
        raise ValueError("Simulation produced nonfinite data")
    # 最終行のトルクは最終状態での指令値を記録し、次ステップには適用しない。
    return time, states, torques


def save_results(results, output):
    """両ケースのCSV・比較PNG・誤差JSONを新規ディレクトリへ保存する。"""
    output.mkdir(parents=True, exist_ok=False)
    summary = dict(
        dt_s=0.0025, duration_s=2.0, method="rk2 midpoint", g=9.81,
        b1=0.0, b2=0.0, parameters=physical_parameters(),
        initial_state=[0.0, 0.0, 0.0, 0.0],
        torque_sampling="current state; held constant within each step; final row is not applied",
    )
    for name, (time, states, torques) in results.items():
        np.savetxt(
            output / (name + ".csv"), np.column_stack((time, states, torques)),
            delimiter=",", comments="",
            header="time_s,theta1_rad,theta2_rad,omega1_rad_s,omega2_rad_s,tau1_Nm,tau2_Nm",
        )
        angle_error = states[:, :2] - states[0, :2]
        summary[name] = dict(
            samples=len(time), initial_torque_Nm=torques[0].tolist(),
            final_state=states[-1].tolist(),
            max_abs_angle_error_rad=np.max(np.abs(angle_error), axis=0).tolist(),
            max_angle_error_norm_rad=float(np.max(np.linalg.norm(angle_error, axis=1))),
            max_abs_velocity_rad_s=np.max(np.abs(states[:, 2:]), axis=0).tolist(),
        )
    for filename, columns, labels, unit in (
        ("angles.png", (0, 1), ("theta1", "theta2 (relative)"), "Angle [rad]"),
        ("angular_velocities.png", (2, 3), ("omega1", "omega2"), "Angular velocity [rad/s]"),
        ("torques.png", (0, 1), ("tau1", "tau2"), "Torque [N m]"),
    ):
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        for axis, column, label in zip(axes, columns, labels):
            for name, (time, states, torques) in results.items():
                values = torques if filename == "torques.png" else states
                axis.plot(time, values[:, column], label=name)
            axis.set_title(label)
            axis.set_ylabel(unit)
            axis.grid(True)
            axis.legend()
        axes[-1].set_xlabel("Time [s]")
        fig.tight_layout()
        fig.savefig(output / filename, dpi=150)
        plt.close(fig)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    """重力補償比較を実行する。--output で再実行用の新規出力先を指定できる。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "step_8_1")
    args = parser.parse_args()
    results = {"uncompensated": simulate(False), "compensated": simulate(True)}
    summary = save_results(results, args.output)
    print(json.dumps(summary, indent=2))
    print("Output:", args.output.resolve())


if __name__ == "__main__":
    main()
