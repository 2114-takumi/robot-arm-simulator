"""Step 7-6のCSVからEuler法とRK2法の自由運動を比較するGIFを作る。"""

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from robot_arm.kinematics import forward_kinematics


def load_csv(path):
    """CSVの列名を使って時刻[s]と関節角度[rad]を読み、検証して返す。"""
    data = np.genfromtxt(path, delimiter=",", names=True, encoding="utf-8")
    required = ("time_s", "theta1_rad", "theta2_rad")
    if data.dtype.names is None or not all(name in data.dtype.names for name in required):
        raise ValueError("CSV must contain time_s, theta1_rad, theta2_rad")
    data = np.atleast_1d(data)
    time = data["time_s"]
    angles = np.column_stack((data["theta1_rad"], data["theta2_rad"]))
    if len(time) < 2 or not np.all(np.isfinite(time)) or not np.all(np.isfinite(angles)):
        raise ValueError("CSV must contain at least two finite states")
    if not np.all(np.diff(time) > 0):
        raise ValueError("CSV times must increase strictly")
    return time, angles


def link_coordinates(angles, L1=1.0, L2=0.8):
    """相対角theta2を既存FKへ渡し、各時刻の原点・関節・先端座標[m]を返す。"""
    x1, y1, x2, y2 = forward_kinematics(angles[:, 0], angles[:, 1], L1, L2)
    zeros = np.zeros(len(angles))
    x = np.column_stack((zeros, x1, x2))
    y = np.column_stack((zeros, y1, y2))
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("FK produced nonfinite coordinates")
    return x, y


def select_frames(time, fps=30):
    """等間隔時刻から約fps個/秒の標本を選び、先頭と末尾を必ず含める。"""
    intervals = np.diff(time)
    if not np.allclose(intervals, intervals[0], rtol=1e-8, atol=1e-12):
        raise ValueError("Animation requires uniformly sampled CSV times")
    frame_count = max(2, round((time[-1] - time[0]) * fps) + 1)
    return np.unique(np.rint(np.linspace(0, len(time) - 1, frame_count)).astype(int))


def create_animation(input_dir, output_dir):
    """CSVを同期して間引き、固定縮尺の比較GIFを新規ディレクトリに保存する。"""
    euler_time, euler_angles = load_csv(input_dir / "euler.csv")
    rk2_time, rk2_angles = load_csv(input_dir / "rk2.csv")
    if not np.array_equal(euler_time, rk2_time):
        raise ValueError("Euler and RK2 CSV times must match")
    time = euler_time
    frames = select_frames(time)
    coordinates = [link_coordinates(euler_angles), link_coordinates(rk2_angles)]
    output_dir.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.5))
    artists = []
    # 全長1.8mに余白を加え、両手法で同じ表示範囲と縮尺を使う。
    limit = (1.0 + 0.8) * 1.1
    for axis, title in zip(axes, ("Euler", "RK2 midpoint")):
        axis.set(xlim=(-limit, limit), ylim=(-limit, limit), xlabel="x [m]", ylabel="y [m]", title=title)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True)
        arm, = axis.plot([], [], "-", color="steelblue", linewidth=3)
        axis.plot(0, 0, "ks", label="Base")
        joint, = axis.plot([], [], "o", color="darkorange", label="Joint 2")
        tip, = axis.plot([], [], "o", color="crimson", label="Tip")
        clock = axis.text(0.03, 0.95, "", transform=axis.transAxes, va="top")
        axis.legend(loc="lower left", fontsize=8)
        artists.append((arm, joint, tip, clock))

    def update(index):
        """選んだCSV行をそのまま描画し、時刻表示も同じ行から更新する。"""
        changed = []
        for (x, y), (arm, joint, tip, clock) in zip(coordinates, artists):
            arm.set_data(x[index], y[index])
            joint.set_data([x[index, 1]], [y[index, 1]])
            tip.set_data([x[index, 2]], [y[index, 2]])
            clock.set_text(f"t = {time[index]:.3f} s")
            changed.extend((arm, joint, tip, clock))
        return changed

    update(frames[0])
    fig.tight_layout()
    animation = FuncAnimation(fig, update, frames=frames, interval=1000 / 30, blit=False, repeat=False)
    path = output_dir / "coupled_dynamics_comparison.gif"
    animation.save(path, writer=PillowWriter(fps=30), dpi=100)
    plt.close(fig)
    return path, len(frames)


def main():
    """既定のStep 7-6入力からStep 7-7出力へGIFを書き出す。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "outputs" / "step_7_6")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "step_7_7")
    args = parser.parse_args()
    path, count = create_animation(args.input, args.output)
    print(f"Saved {count} frames at 30 fps: {path}")


if __name__ == "__main__":
    main()
