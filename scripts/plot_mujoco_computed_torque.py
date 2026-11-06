"""Step 8-6の既存ランナーで自作RK2とMuJoCo RK4の応答を描画する。"""

from pathlib import Path
from datetime import datetime
import importlib.util
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_computed_torque_demo import DT, DURATION, TARGET, simulate


def load_mujoco_runner():
    path = ROOT / "tests" / "test_mujoco_computed_torque.py"
    spec = importlib.util.spec_from_file_location("mujoco_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.mujoco is None:
        raise RuntimeError("MuJoCoをimportできるPython環境で実行してください。")
    return module.run_mujoco


def plot_comparison(time, custom, mujoco, title, ylabel, path, target=None):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for j, ax in enumerate(axes):
        ax.plot(time, custom[:, j], label="Custom RK2")
        ax.plot(time, mujoco[:, j], "--", label="MuJoCo RK4")
        if target is not None:
            ax.axhline(target[j], color="black", linestyle=":", label="Target")
        ax.set_title(f"Joint {j + 1}")
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.legend()
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    run_mujoco = load_mujoco_runner()
    time, states_c, torques_c, errors_c = simulate("computed_torque")
    time_m, states_m, torques_m = run_mujoco(DT, DURATION)
    np.testing.assert_array_equal(time, time_m)
    for array in (states_c, torques_c, states_m, torques_m):
        if not np.all(np.isfinite(array)):
            raise ValueError("Nonfinite simulation data")
    errors_m = TARGET - states_m[:, :2]

    # 毎回新しい保存先を作り、既存の生成物を上書きしない。
    output = ROOT / "outputs" / ("mujoco_computed_torque_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(parents=True, exist_ok=False)
    for filename, custom, mujoco, title, ylabel, target in (
        ("angles.png", states_c[:, :2], states_m[:, :2], "Joint angles", "Angle [rad]", TARGET),
        ("angular_velocities.png", states_c[:, 2:], states_m[:, 2:], "Joint velocities", "Angular velocity [rad/s]", np.zeros(2)),
        ("errors.png", errors_c, errors_m, "Angle errors", "Target - angle [rad]", np.zeros(2)),
        ("torques.png", torques_c, torques_m, "Commanded torques", "Torque [N m]", None),
    ):
        plot_comparison(time, custom, mujoco, title, ylabel, output / filename, target)

    summary = dict(
        dt_s=DT, duration_s=DURATION, target_rad=TARGET.tolist(),
        note="Same sampled controller; torque held within each step. Different integrators need not match exactly.",
        max_abs_angle_difference_rad=np.max(np.abs(states_c[:, :2] - states_m[:, :2]), axis=0).tolist(),
        max_abs_velocity_difference_rad_s=np.max(np.abs(states_c[:, 2:] - states_m[:, 2:]), axis=0).tolist(),
        max_abs_torque_difference_Nm=np.max(np.abs(torques_c - torques_m), axis=0).tolist(),
        custom_final_angle_error_rad=errors_c[-1].tolist(),
        mujoco_final_angle_error_rad=errors_m[-1].tolist(),
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Output:", output)


if __name__ == "__main__":
    main()
