"""Export portfolio visuals from the frozen regression data without running main.py."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

try:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    import numpy as np
    from PIL import Image, ImageSequence
except ImportError as exc:
    raise SystemExit(
        f"Missing visualization dependency: {exc}. "
        "Install with: python -m pip install numpy matplotlib pillow"
    ) from exc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from robot_arm.kinematics import forward_kinematics

BLUE = "#2563eb"
TEAL = "#087f8c"
AMBER = "#d18a22"
NAVY = "#182f48"
GRAY = "#607086"
TARGET = "#d34c4c"
INTERVAL_MS = 100
PNG_DPI = 160

KEYS = {
    "time": "fixed_target.time_s",
    "joint1": "fixed_target.pid_theta1_rad",
    "joint2": "fixed_target.after_theta2_rad",
    "p": "fixed_target.p_theta2_rad",
    "pd": "fixed_target.pd_theta2_rad",
    "pid": "fixed_target.pid_theta2_rad",
    # Captured Figure 1 target and IK pose; Figure 3 target angle is in degrees.
    "target": "plot.figure_01.axis_0.collection_0.offsets",
    "pose_x": "plot.figure_01.axis_0.line_1.x",
    "pose_y": "plot.figure_01.axis_0.line_1.y",
    "target_deg": "plot.figure_03.axis_0.line_1.y",
}


def load_reference(directory):
    manifest = json.loads((directory / "arrays_manifest.json").read_text(encoding="utf-8-sig"))
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8-sig"))
    environment = json.loads((directory / "environment.json").read_text(encoding="utf-8-sig"))
    if environment.get("status") != "complete" or summary.get("schema_version") != 1:
        raise ValueError("Expected a complete schema-1 reference bundle")
    data = {}
    with np.load(directory / "arrays.npz", allow_pickle=False) as arrays:
        for name, key in KEYS.items():
            values = arrays[key].copy()
            info = manifest[key]
            if list(values.shape) != info["shape"] or str(values.dtype) != info["dtype"]:
                raise ValueError(f"Reference array disagrees with its manifest: {key}")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"Non-finite reference data: {key}")
            data[name] = values
    for name in ("time", "joint1", "joint2", "p", "pd", "pid"):
        if data[name].shape != (201,):
            raise ValueError(f"Expected 201 fixed-target samples: {KEYS[name]}")
        expected_unit = "s" if name == "time" else "rad"
        if manifest[KEYS[name]]["unit"] != expected_unit:
            raise ValueError(f"Unexpected unit: {KEYS[name]}")
    if not np.all(np.diff(data["time"]) > 0):
        raise ValueError("Simulation time must increase")
    if data["target"].shape != (1, 2) or not np.all(data["target_deg"] == data["target_deg"][0]):
        raise ValueError("Expected the captured fixed target")
    data["L1"] = environment["parameters"]["L1_m"]
    data["L2"] = environment["parameters"]["L2_m"]
    data["summary"] = summary
    data["reference_commit"] = environment["git_commit"]
    return data


def animation_coordinates(data):
    """Use exactly the same source frames and joint histories as Figure 1."""
    frames = np.arange(0, 201, 2)
    coordinates = np.array([
        forward_kinematics(data["joint1"][i], data["joint2"][i], data["L1"], data["L2"])
        for i in frames
    ])
    return frames, coordinates


def style_axis(axis):
    axis.set_axisbelow(True)
    axis.grid(True, color="#e0e7ef", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#9aa9ba")
    axis.tick_params(colors=GRAY)


def save_png(figure, path, data, description):
    figure.savefig(path, dpi=PNG_DPI, facecolor="white", metadata={
        "Description": f"Frozen baseline {data['reference_commit']}. {description}",
    })
    plt.close(figure)


def export_gif(data, path):
    frames, coordinates = animation_coordinates(data)
    x_values = coordinates[:, [0, 2]]
    y_values = coordinates[:, [1, 3]]
    limits_x, limits_y = (-0.2, 2.0), (-0.35, 1.15)
    if not (x_values.min() >= limits_x[0] and x_values.max() <= limits_x[1]
            and y_values.min() >= limits_y[0] and y_values.max() <= limits_y[1]):
        raise ValueError("The fixed animation viewport would clip the arm")
    figure, axis = plt.subplots(figsize=(7.2, 6.0), dpi=100)
    figure.subplots_adjust(left=0.10, right=0.97, bottom=0.18, top=0.84)
    figure.suptitle("2-Link Robot Arm", fontsize=18, fontweight="bold", color=NAVY, y=0.97)
    target_x, target_y = data["target"][0]
    figure.text(0.5, 0.905,
                f"Fixed target ({target_x:.1f}, {target_y:.1f}) m | PID control",
                ha="center", color=GRAY, fontsize=11)
    axis.set(xlim=limits_x, ylim=limits_y, xlabel="X [m]", ylabel="Y [m]")
    axis.set_aspect("equal", adjustable="box")
    style_axis(axis)
    target_pose, = axis.plot(data["pose_x"], data["pose_y"], color="#b5c0cd",
                             linestyle="--", linewidth=1.5, label="Target pose")
    common = {"linewidth": 4.5, "marker": "o", "markersize": 9,
              "markerfacecolor": "white", "markeredgewidth": 2, "zorder": 3}
    link1, = axis.plot([], [], color=BLUE, label="Link 1", **common)
    link2, = axis.plot([], [], color=TEAL, label="Link 2", **common)
    goal = axis.scatter(target_x, target_y, marker="x", s=95, linewidths=2.5,
                        color=TARGET, label="Target", zorder=4)
    clock = axis.text(0.03, 0.94, "", transform=axis.transAxes, color=NAVY,
                      fontsize=12, fontweight="bold")
    figure.legend(handles=[link1, link2, goal, target_pose], loc="lower center", ncol=4,
                  frameon=False, bbox_to_anchor=(0.5, 0.025))

    def draw(index):
        x1, y1, x2, y2 = coordinates[index]
        link1.set_data([0, x1], [0, y1])
        link2.set_data([x1, x2], [y1, y2])
        clock.set_text(f"t = {data['time'][frames[index]]:.1f} s")
        return link1, link2, clock

    animation = FuncAnimation(figure, draw, frames=range(len(frames)),
                              interval=INTERVAL_MS, repeat=True)
    animation.save(path, writer=PillowWriter(fps=1000 / INTERVAL_MS), dpi=100)
    plt.close(figure)
    with Image.open(path) as gif:
        count = gif.n_frames
        loop = gif.info.get("loop")
        size = list(gif.size)
        durations = [frame.info["duration"] for frame in ImageSequence.Iterator(gif)]
    if count != 101 or loop != 0 or durations != [INTERVAL_MS] * 101:
        raise ValueError("The saved GIF did not retain all 101 frames and their timing")
    return {"path": str(path), "frames": count, "duration_seconds": sum(durations) / 1000,
            "fps": 1000 / INTERVAL_MS, "loop": "infinite", "pixels": size,
            "bytes": path.stat().st_size, "source_frames": "0, 2, ..., 200",
            "additional_frame_decimation": False}


def export_pid(data, path):
    figure, axis = plt.subplots(figsize=(9.6, 5.4), layout="constrained")
    for name, label, color in [("p", "P", BLUE), ("pd", "PD", AMBER), ("pid", "PID", TEAL)]:
        axis.plot(data["time"], np.rad2deg(data[name]), label=label, color=color, linewidth=2.2)
    target = float(data["target_deg"][0])
    axis.axhline(target, color=NAVY, linestyle="--", linewidth=1.5,
                 label=f"Target ({target:.2f} deg)")
    axis.set(xlim=(data["time"][0], data["time"][-1]), xlabel="Time [s]",
             ylabel="Joint 2 angle [deg]", title="Joint 2: P, PD and PID response")
    style_axis(axis)
    axis.legend(frameon=False, loc="upper right", ncol=2)
    save_png(figure, path, data, "Fixed-gain P/PD/PID; common -0.1 N m disturbance.")
    return {"path": str(path), "samples_per_curve": len(data["time"]),
            "source_keys": [KEYS[name] for name in ("time", "p", "pd", "pid")],
            "target_angle_deg": target, "pid_uses_optimized_gains": False}


def export_integrators(data, path):
    fixed = data["summary"]["fixed_target"]
    errors_mm = np.array([fixed["euler_line_error_m"], fixed["rk2_line_error_m"]]) * 1000
    if not np.all(np.isfinite(errors_mm)) or not np.all(errors_mm > 0):
        raise ValueError("The log-scale comparison requires positive finite errors")
    rk2_condition = data["summary"]["rk2_step_runs"][0]
    if rk2_condition["dt_s"] != 0.05 or rk2_condition["final_error_mm"] != errors_mm[1]:
        raise ValueError("Expected the dt=0.05 s RK2 reference condition")
    figure, axis = plt.subplots(figsize=(9.6, 5.4), layout="constrained")
    bars = axis.bar(["Euler", "RK2 midpoint"], errors_mm, color=[BLUE, TEAL], width=0.48, zorder=3)
    axis.set_yscale("log")
    axis.set_ylim(errors_mm.min() / 4, errors_mm.max() * 5)
    axis.set_ylabel("Endpoint position error [mm] (log scale)")
    axis.set_xlabel("Numerical integration method")
    axis.set_title("Euler vs RK2 midpoint\ndt = 0.05 s | 20 steps | 1.0 s", pad=16)
    style_axis(axis)
    for bar, error, precision in zip(bars, errors_mm, (4, 7)):
        axis.annotate(f"{error:.{precision}f} mm", (bar.get_x() + bar.get_width() / 2, error),
                      xytext=(0, 10), textcoords="offset points", ha="center",
                      color=NAVY, fontsize=13, fontweight="bold")
    save_png(figure, path, data, "Euler/RK2 endpoint errors, dt=0.05 s, 20 steps, 1 s; log y-axis.")
    return {"path": str(path), "dt_seconds": 0.05, "steps": 20,
            "duration_seconds": 1.0, "errors_mm": errors_mm.tolist(), "y_scale": "log",
            "source_keys": ["fixed_target.euler_line_error_m", "fixed_target.rk2_line_error_m"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, default=ROOT / "tests/baseline/reference")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "assets")
    args = parser.parse_args()
    reference = args.reference_dir.resolve()
    output = args.output_dir.resolve()
    if output.is_relative_to(reference):
        parser.error("The output directory must be outside the reference data directory")
    protected = [p for p in reference.iterdir() if p.is_file()]
    original_graph = ROOT / "assets/trajectory_tracking_comparison.png"
    protected.append(original_graph)
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    data = load_reference(reference)
    output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11,
                         "text.color": NAVY, "axes.labelcolor": NAVY,
                         "axes.titlecolor": NAVY, "axes.titleweight": "bold",
                         "axes.titlesize": 15, "figure.facecolor": "white"}):
        result = {"reference_directory": str(reference), "reference_commit": data["reference_commit"],
                  "backend": matplotlib.get_backend(),
                  "gif": export_gif(data, output / "robot_arm_demo.gif"),
                  "pid": export_pid(data, output / "pid_response_comparison.png"),
                  "integrators": export_integrators(data, output / "euler_rk2_comparison.png")}
    if any(hashlib.sha256(p.read_bytes()).hexdigest() != digest for p, digest in hashes.items()):
        raise RuntimeError("A protected reference file or original graph changed")
    result["reference_and_original_graph_unchanged"] = True
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
