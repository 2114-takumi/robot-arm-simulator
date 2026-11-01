"""水平伸展姿勢から放した2リンクロボットの自由運動を比較する。"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from robot_arm.dynamics import forward_dynamics, integrate_dynamics, mass_matrix


def physical_parameters():
    """一様な細い棒を仮定したSI単位のテスト値を返す（実機仕様ではない）。"""
    L1, L2 = 1.0, 0.8
    m1, m2 = 1.0, 0.8
    return dict(
        L1=L1, m1=m1, m2=m2, lc1=0.5, lc2=0.4,
        I1=m1 * L1**2 / 12, I2=m2 * L2**2 / 12,
    )


def calculate_energy(state, parameters, g=9.81):
    """状態 [rad, rad, rad/s, rad/s] から T, V, E [J] を返す。"""
    theta1, theta2, omega1, omega2 = state
    omega = np.array([omega1, omega2])
    M = mass_matrix(theta1, theta2, **parameters)
    kinetic = 0.5 * omega @ M @ omega
    # 重心の高さ。水平伸展姿勢の高さを0とする。
    y_com1 = parameters["lc1"] * np.sin(theta1)
    y_com2 = parameters["L1"] * np.sin(theta1) + parameters["lc2"] * np.sin(theta1 + theta2)
    potential = parameters["m1"] * g * y_com1 + parameters["m2"] * g * y_com2
    return np.array([kinetic, potential, kinetic + potential])


def simulate(method, dt=0.0025, duration=2.0):
    """無摩擦・無モータトルクの自由運動の時刻 [s]、状態、T/V/E [J] を返す。"""
    if not np.isfinite(dt) or dt <= 0 or not np.isfinite(duration) or duration <= 0:
        raise ValueError("dt and duration must be positive and finite")
    steps = round(duration / dt)
    if steps < 1 or not np.isclose(steps * dt, duration, rtol=0, atol=1e-12):
        raise ValueError("duration must be an integer multiple of dt")
    parameters = physical_parameters()
    time = np.arange(steps + 1) * dt
    states = np.zeros((steps + 1, 4))
    energies = np.zeros((steps + 1, 3))
    energies[0] = calculate_energy(states[0], parameters)
    for index in range(steps):
        states[index + 1] = integrate_dynamics(
            *states[index], 0.0, 0.0, dt, **parameters,
            b1=0.0, b2=0.0, g=9.81, method=method
        )
        energies[index + 1] = calculate_energy(states[index + 1], parameters)
    if not np.all(np.isfinite(states)) or not np.all(np.isfinite(energies)):
        raise ValueError("simulation produced nonfinite values")
    return time, states, energies


def save_comparison(results, output):
    """両手法の角度・角速度・エネルギー比較PNGを専用出力先へ保存する。"""
    for filename, columns, labels, ylabel in (
        ("angles.png", (0, 1), ("theta1", "theta2 (relative)"), "Angle [rad]"),
        ("angular_velocities.png", (2, 3), ("omega1", "omega2"), "Angular velocity [rad/s]"),
    ):
        fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        for axis, column, label in zip(axes, columns, labels):
            for method, (time, states, _) in results.items():
                axis.plot(time, states[:, column], label=method, linestyle="--" if method == "euler" else "-")
            axis.set_title(label)
            axis.set_ylabel(ylabel)
            axis.grid(True)
            axis.legend()
        axes[-1].set_xlabel("Time [s]")
        fig.tight_layout()
        fig.savefig(output / filename, dpi=150)
        plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for method, (time, _, energies) in results.items():
        total = energies[:, 2]
        axes[0].plot(time, total, label=method)
        axes[1].plot(time, total - total[0], label=method)
    for axis, label in zip(axes, ("Total energy [J]", "E(t) - E(0) [J]")):
        axis.set_ylabel(label)
        axis.grid(True)
        axis.legend()
    axes[-1].set_xlabel("Time [s]")
    fig.tight_layout()
    fig.savefig(output / "energy_comparison.png", dpi=150)
    plt.close(fig)


def main():
    """基本条件で比較を実行し、CSV・PNG・検証結果JSONを新規ディレクトリへ出力する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New output directory; existing paths are rejected")
    args = parser.parse_args()
    output = args.output or ROOT / "outputs" / ("coupled_dynamics_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    # 既存のディレクトリへの出力を拒否し、assetsやbaselineを上書きしない。
    output.mkdir(parents=True, exist_ok=False)
    results = {}
    summary = dict(dt_s=0.0025, duration_s=2.0, parameters=physical_parameters(), g=9.81, b1=0.0, b2=0.0, tau=[0.0, 0.0])
    initial_alpha = forward_dynamics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, **physical_parameters())
    summary["initial_acceleration_rad_s2"] = initial_alpha.tolist()
    for method in ("euler", "rk2"):
        time, states, energies = simulate(method)
        # 初期は速度0のため、中点でも姿勢は同じ。cは速度の2乗なので、
        # RK2の差分加速度はdt→0で初期加速度に収束する。
        if method == "euler":
            np.testing.assert_allclose(states[1, 2:] / 0.0025, initial_alpha, rtol=0, atol=1e-12)
        results[method] = (time, states, energies)
        np.savetxt(
            output / (method + ".csv"), np.column_stack((time, states, energies)), delimiter=",",
            header="time_s,theta1_rad,theta2_rad,omega1_rad_s,omega2_rad_s,T_J,V_J,E_J", comments=""
        )
        error = np.abs(energies[:, 2] - energies[0, 2])
        summary[method] = dict(
            samples=len(time), final_state=states[-1].tolist(),
            max_energy_error_J=float(np.max(error)),
            short_time_max_energy_error_J=float(np.max(error[time <= 0.2])),
            theta_min_rad=states[:, :2].min(axis=0).tolist(),
            theta_max_rad=states[:, :2].max(axis=0).tolist(),
            omega_min_rad_s=states[:, 2:].min(axis=0).tolist(),
            omega_max_rad_s=states[:, 2:].max(axis=0).tolist(),
        )
    save_comparison(results, output)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Output:", output.resolve())


if __name__ == "__main__":
    main()
