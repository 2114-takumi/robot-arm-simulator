"""Run the existing script without a GUI and export numerical reference data."""

import argparse
import ast
import copy
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import runpy
import subprocess
import sys
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]

# Capture BEFORE these top-level assignments execute. State copies prevent later
# assignments and in-place holding updates from changing the saved observations.
CHECKPOINTS = {
    "kinematic_p": ("J", ["settling_time"]),
    "gain_1d": ("kp_values", ["best_kd", "best_cost_kd", "min_cost", "min_overshoot"]),
    "gain_2d": ("kp_values_3d", ["best_kp", "best_kd_2d", "min_cost_2d"]),
    "fixed_target": ("trajectory_dt", [
        "position_error", "position_settling_time", "predicted_error",
        "delta_theta", "delta_position", "approximation_error_mm",
        "line_final_error", "rk2_error", "theta_rk2", "ts_p", "ts_pd", "ts_pid",
        "best_gains_3d", "min_cost_3d", "before_settling_time", "after_settling_time",
        "velocity", "velocity_fd", "velocity_difference", "J_target",
    ]),
    "constant_tracking_end": ("hold_duration", [
        "theta_ff", "omega_ff", "integral_ff", "previous_error_ff",
    ]),
    "constant_hold_end": ("T", [
        "theta_ff", "omega_ff", "integral_ff", "previous_error_ff",
    ]),
    "quintic_tracking_end": ("hold_theta_q", [
        "quintic_actual_theta", "quintic_actual_omega",
        "quintic_integral_error", "quintic_previous_error",
    ]),
    "quintic_hold_end": ("comparison_time", [
        "hold_theta_q", "hold_omega_q", "hold_integral_q", "hold_previous_error_q",
    ]),
}

LOOP_STATES = {
    "constant.pid_ff": ("trajectory_ff_error_history", ["theta_ff", "omega_ff", "integral_ff", "previous_error_ff"]),
    "constant.hold": ("hold_error_history", ["theta_ff", "omega_ff", "integral_ff", "previous_error_ff"]),
    "quintic.pid_ff": ("quintic_position_error_history", ["quintic_actual_theta", "quintic_actual_omega", "quintic_integral_error", "quintic_previous_error"]),
    "quintic.hold": ("hold_error_q", ["hold_theta_q", "hold_omega_q", "hold_integral_q", "hold_previous_error_q"]),
    "comparison.constant": ("constant_error_history", ["constant_theta_actual", "constant_omega_actual", "constant_integral", "constant_previous_error"]),
}


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protected_hashes(source):
    paths = [source, ROOT / "README.md", *(ROOT / "assets").rglob("*")]
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths if path.is_file()}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def checkpoint_lines(source):
    tree = ast.parse(source)
    result = {}
    for stage, (variable, _) in CHECKPOINTS.items():
        matches = [node.lineno for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == variable for target in node.targets)]
        if len(matches) != 1:
            raise ValueError(f"Checkpoint {stage}: expected one assignment to {variable}; got {matches}")
        result[matches[0]] = stage
    loop_lines = {}
    for prefix, (history, _) in LOOP_STATES.items():
        matches = [node for node in tree.body if isinstance(node, ast.For) and any(
            isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
            and child.func.attr == "append" and isinstance(child.func.value, ast.Name)
            and child.func.value.id == history for child in ast.walk(node))]
        if len(matches) != 1:
            raise ValueError(f"Expected one recording loop for {history}")
        loop_lines[matches[0].body[0].lineno] = prefix
    return result, loop_lines


def capture(output, source):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.backend_bases import FigureManagerBase
    from matplotlib.figure import Figure
    import numpy as np

    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    before = protected_hashes(source)
    lines, loop_lines = checkpoint_lines(source.read_text(encoding="utf-8-sig"))
    snapshots = {}
    observed_states = {prefix: [] for prefix in LOOP_STATES}
    observed_iterations = {}
    pid_runs = []
    rk2_runs = []
    saves = []
    output.mkdir(parents=True)
    stdout, stderr = io.StringIO(), io.StringIO()
    original_savefig = Figure.savefig
    source_filename = str(source)
    simulation_filename = str(ROOT / "robot_arm" / "simulation.py")

    def trace(frame, event, value):
        function = frame.f_code.co_name
        if frame.f_code.co_filename != source_filename:
            if frame.f_code.co_filename != simulation_filename or function not in ("simulate_pid", "simulate_rk2"):
                return None
        if event == "call":
            if function in ("simulate_pid", "simulate_rk2"):
                frame.f_trace_lines = False
            elif function != "<module>":
                return None
        elif event == "line" and function == "<module>":
            if frame.f_lineno in lines:
                stage = lines[frame.f_lineno]
                if stage not in snapshots:
                    snapshots[stage] = {name: copy.deepcopy(frame.f_locals[name])
                                        for name in CHECKPOINTS[stage][1]}
            if frame.f_lineno in loop_lines:
                prefix = loop_lines[frame.f_lineno]
                # A multiline assignment can emit this line event twice. Observe
                # each loop iteration once, before any controller update.
                iteration = frame.f_locals["step" if prefix.endswith(".hold") else "k"]
                if observed_iterations.get(prefix) != iteration:
                    observed_states[prefix].append([frame.f_locals[name].copy() for name in LOOP_STATES[prefix][1]])
                    observed_iterations[prefix] = iteration
        elif event == "return" and function == "simulate_pid":
            local, global_ = frame.f_locals, frame.f_globals
            pid_runs.append({
                "parameters": {name: float(local[name]) for name in ("kp", "ki", "kd")},
                "dt_s": float(local["dt"] if "dt" in local else global_["dt"]),
                "theta_rad": copy.deepcopy(value),
                "final_theta_rad": float(local["theta"]),
                "final_omega_rad_s": float(local["omega"]),
                "final_integral_rad_times_s": float(local["integral"]),
            })
        elif event == "return" and function == "simulate_rk2":
            rk2_runs.append({"dt_s": float(frame.f_locals["test_dt"]),
                             "final_theta_rad": frame.f_locals["theta"].copy(),
                             "final_error_mm": float(value)})
        return trace

    def redirected_savefig(figure, filename, *args, **kwargs):
        requested = Path(filename).resolve()
        expected = ROOT / "assets" / "trajectory_tracking_comparison.png"
        if requested != expected:
            raise ValueError(f"Unexpected image write requested: {requested}")
        actual = output / "trajectory_tracking_comparison.png"
        if actual.exists():
            raise FileExistsError(f"Refusing to overwrite image: {actual}")
        saves.append({"requested": "assets/trajectory_tracking_comparison.png",
                      "actual": actual.name, "dpi": kwargs.get("dpi"),
                      "bbox_inches": kwargs.get("bbox_inches")})
        return original_savefig(figure, actual, *args, **kwargs)

    previous_trace, previous_cwd = sys.gettrace(), Path.cwd()
    try:
        os.chdir(ROOT)
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr), \
                patch.object(sys, "path", [str(ROOT), *sys.path]), \
                patch.object(FigureManagerBase, "window", SimpleNamespace(
                    lift=lambda: None, attributes=lambda *args: None), create=True), \
                patch.object(Figure, "savefig", redirected_savefig), \
                patch.object(plt, "show", lambda *args, **kwargs: None):
            sys.settrace(trace)
            try:
                state = runpy.run_path(source_filename, run_name="__main__")
            finally:
                sys.settrace(previous_trace)
    finally:
        os.chdir(previous_cwd)

    if set(snapshots) != set(CHECKPOINTS):
        raise ValueError(f"Missing snapshots: {set(CHECKPOINTS) - set(snapshots)}")
    if len(saves) != 1 or not pid_runs or not rk2_runs:
        raise ValueError("Expected image export / gain sweep / RK2 runs were not captured")

    arrays, manifest = {}, {}

    def add(key, value, unit, origin):
        array = np.array(value, copy=True)
        if array.dtype.kind not in "biuf" or not np.all(np.isfinite(array)):
            raise ValueError(f"Non-finite or unsupported array: {key}")
        arrays[key] = array
        manifest[key] = {"shape": list(array.shape), "dtype": str(array.dtype),
                         "unit": unit, "source": origin}

    # Existing full-precision histories, grouped by the experiment rather than
    # transient scratch variables. No source formulas or gains are replaced.
    specs = [
        ("kinematic_p.time_s", "time_history", "s"),
        ("kinematic_p.theta1_rad", "theta1_history", "rad"),
        ("kinematic_p.theta2_rad", "theta2_history", "rad"),
        ("fixed_target.time_s", "dynamic_time_history", "s"),
        ("fixed_target.p_theta2_rad", "dynamic_theta_history", "rad"),
        ("fixed_target.pd_theta2_rad", "pd_theta_history", "rad"),
        ("fixed_target.pid_theta2_rad", "pid_theta_history", "rad"),
        ("fixed_target.pid_theta1_rad", "theta1_pid_history", "rad"),
        ("fixed_target.before_theta2_rad", "before_history", "rad"),
        ("fixed_target.after_theta2_rad", "after_history", "rad"),
        ("fixed_target.position_error_mm", "fixed_target_error_history", "mm"),
        ("velocity_motion.x_m", "line_x_history", "m"),
        ("velocity_motion.y_m", "line_y_history", "m"),
        ("constant.time_s", "trajectory_time", "s"),
        ("constant.desired_theta1_rad", "theta1_desired_history", "rad"),
        ("constant.desired_theta2_rad", "theta2_desired_history", "rad"),
        ("constant.desired_omega_rad_s", "desired_joint_velocity_history", "rad/s"),
        ("constant.desired_alpha_rad_s2", "alpha_desired_array", "rad/s^2"),
        ("constant.feedforward_torque_nm", "torque_ff_array", "N*m"),
        ("constant.pid_only_theta1_rad", "theta1_actual_history", "rad"),
        ("constant.pid_only_theta2_rad", "theta2_actual_history", "rad"),
        ("constant.pid_only_error_mm", "trajectory_pid_error_history", "mm"),
        ("constant.pid_ff_error_mm", "trajectory_ff_error_history", "mm"),
        ("constant.pid_only_x_error_mm", "x_error_history", "mm"),
        ("constant.pid_only_y_error_mm", "y_error_history", "mm"),
        ("constant.hold_time_s", "hold_time_history", "s"),
        ("constant.hold_error_mm", "hold_error_history", "mm"),
        ("constant.tip_velocity_at_1s_m_s", "tip_velocity_start", "m/s"),
        ("quintic.visualization_time_s", "quintic_time", "s"),
        ("quintic.visualization_x_m", "quintic_x", "m"),
        ("quintic.visualization_vx_m_s", "quintic_vx", "m/s"),
        ("quintic.visualization_ax_m_s2", "quintic_ax", "m/s^2"),
        ("quintic.time_s", "quintic_tracking_time", "s"),
        ("quintic.desired_theta_rad", "quintic_theta_history", "rad"),
        ("quintic.desired_omega_rad_s", "quintic_omega_history", "rad/s"),
        ("quintic.desired_alpha_rad_s2", "quintic_alpha_history", "rad/s^2"),
        ("quintic.feedforward_torque_nm", "quintic_torque_ff_history", "N*m"),
        ("quintic.pid_ff_error_mm", "quintic_position_error_history", "mm"),
        ("quintic.tip_velocity_at_1s_m_s", "quintic_tip_velocity_final", "m/s"),
        ("quintic.hold_time_s", "hold_time_q", "s"),
        ("quintic.hold_error_mm", "hold_error_q", "mm"),
        ("quintic.tip_velocity_at_2s_m_s", "tip_velocity_final_q", "m/s"),
        ("comparison.constant.time_s", "comparison_time", "s"),
        ("comparison.constant.error_mm", "constant_error_history", "mm"),
        ("comparison.constant.desired_theta_rad", "constant_theta_desired", "rad"),
        ("comparison.constant.desired_omega_rad_s", "constant_omega_desired", "rad/s"),
        ("comparison.constant.desired_alpha_rad_s2", "constant_alpha_desired", "rad/s^2"),
        ("comparison.constant.feedforward_torque_nm", "constant_torque_ff", "N*m"),
    ]
    for key, variable, unit in specs:
        add(key, state[variable], unit, variable)

    stage_states = {
        "constant_tracking_end": ("constant.tracking_final", ["theta_ff", "omega_ff", "integral_ff", "previous_error_ff"]),
        "constant_hold_end": ("constant.hold_final", ["theta_ff", "omega_ff", "integral_ff", "previous_error_ff"]),
        "quintic_tracking_end": ("quintic.tracking_final", ["quintic_actual_theta", "quintic_actual_omega", "quintic_integral_error", "quintic_previous_error"]),
        "quintic_hold_end": ("quintic.hold_final", ["hold_theta_q", "hold_omega_q", "hold_integral_q", "hold_previous_error_q"]),
    }
    fields = [("theta_rad", "rad"), ("omega_rad_s", "rad/s"),
              ("integral_rad_times_s", "rad*s"), ("previous_error_rad", "rad")]
    for stage, (prefix, variables) in stage_states.items():
        for variable, (field, unit) in zip(variables, fields):
            add(f"{prefix}.{field}", snapshots[stage][variable], unit, f"{stage}:{variable}")
    for prefix, stage in [("constant.hold", "constant_hold_end"), ("quintic.hold", "quintic_hold_end")]:
        observed_states[prefix].append([snapshots[stage][name].copy() for name in LOOP_STATES[prefix][1]])
    for prefix, observations in observed_states.items():
        if len(observations) != 21:
            raise ValueError(f"Expected 21 observed states for {prefix}; got {len(observations)}")
        for index, (field, unit) in enumerate(fields):
            add(f"{prefix}.state_history.{field}", [row[index] for row in observations], unit,
                "copies observed at loop start; hold includes the terminal state")
    for variable, field, unit in [
        ("J_target", "jacobian_target_m_rad", "m/rad"),
        ("delta_theta", "final_joint_error_rad", "rad"),
        ("delta_position", "predicted_position_error_m", "m"),
        ("velocity", "tip_velocity_at_1s_m_s", "m/s"),
        ("velocity_fd", "finite_difference_velocity_at_1s_m_s", "m/s"),
        ("theta_rk2", "rk2_final_theta_rad", "rad"),
    ]:
        add(f"fixed_target.{field}", snapshots["fixed_target"][variable], unit, f"fixed_target:{variable}")

    sweep_summary = []
    for index, run in enumerate(pid_runs):
        key = f"gain_sweep.run_{index:03d}.theta_rad"
        add(key, run["theta_rad"], "rad", "simulate_pid return history")
        sweep_summary.append({name: value for name, value in run.items() if name != "theta_rad"})
    rk2_summary = []
    for index, run in enumerate(rk2_runs):
        add(f"rk2.run_{index:03d}.final_theta_rad", run["final_theta_rad"], "rad", "simulate_rk2 return state")
        rk2_summary.append({name: value for name, value in run.items() if name != "final_theta_rad"})

    def tracking_metrics(error_key, hold_key, velocity_key):
        errors, holds, velocity = state[error_key], state[hold_key], state[velocity_key]
        return {"maximum_tracking_error_mm": float(np.max(errors)),
                "mean_tracking_error_mm": float(np.mean(errors)),
                "tracking_final_error_mm": float(errors[-1]),
                "tip_speed_at_1s_mm_s": float(np.linalg.norm(velocity) * 1000),
                "maximum_hold_error_mm": float(np.max(holds)),
                "hold_final_error_at_2s_mm": float(holds[-1])}

    fixed = snapshots["fixed_target"]
    summary = {
        "schema_version": 1,
        "constant": tracking_metrics("constant_error_history", "hold_error_history", "tip_velocity_start"),
        "quintic": tracking_metrics("quintic_position_error_history", "hold_error_q", "quintic_tip_velocity_final"),
        "kinematic_p_settling_time_s": snapshots["kinematic_p"]["settling_time"],
        "fixed_target": {key: (None if fixed[name] is None else float(fixed[name])) for key, name in {
            "p_joint2_settling_time_s": "ts_p", "pd_joint2_settling_time_s": "ts_pd",
            "pid_joint2_settling_time_s": "ts_pid", "before_joint2_settling_time_s": "before_settling_time",
            "after_joint2_settling_time_s": "after_settling_time", "tip_settling_time_s": "position_settling_time",
            "final_position_error_m": "position_error", "jacobian_predicted_error_m": "predicted_error",
            "jacobian_approximation_error_mm": "approximation_error_mm", "euler_line_error_m": "line_final_error",
            "rk2_line_error_m": "rk2_error", "pid_gain_search_cost": "min_cost_3d",
            "jacobian_vs_difference_velocity_error_m_s": "velocity_difference",
        }.items()},
        "best_pid_gains_kp_ki_kd": list(fixed["best_gains_3d"]),
        "gain_1d": snapshots["gain_1d"],
        "gain_2d": snapshots["gain_2d"],
        "pid_sweep_runs": sweep_summary,
        "rk2_step_runs": rk2_summary,
    }

    # Record plot data and styling without opening desktop windows.
    plots = {"figures": []}
    for number in plt.get_fignums():
        figure = plt.figure(number)
        figure.canvas.draw()
        item = {"number": number, "size_inches": figure.get_size_inches().tolist(),
                "dpi": float(figure.dpi), "axes": []}
        for axis_index, axis in enumerate(figure.axes):
            axis_item = {"title": axis.get_title(), "xlabel": axis.get_xlabel(),
                         "ylabel": axis.get_ylabel(), "xlim": list(axis.get_xlim()),
                         "ylim": list(axis.get_ylim()), "xscale": axis.get_xscale(),
                         "yscale": axis.get_yscale(), "aspect": axis.get_aspect(), "lines": [],
                         "collections": [], "patches": [],
                         "legend": [] if axis.get_legend() is None else [text.get_text() for text in axis.get_legend().get_texts()],
                         "xgrid": any(line.get_visible() for line in axis.get_xgridlines()),
                         "ygrid": any(line.get_visible() for line in axis.get_ygridlines())}
            for line_index, line in enumerate(axis.lines):
                prefix = f"plot.figure_{number:02d}.axis_{axis_index}.line_{line_index}"
                add(prefix + ".x", line.get_xdata(), "axis units", "Matplotlib Line2D xdata")
                add(prefix + ".y", line.get_ydata(), "axis units", "Matplotlib Line2D ydata")
                axis_item["lines"].append({"array_prefix": prefix, "label": line.get_label(),
                                           "color": line.get_color(), "linestyle": line.get_linestyle(),
                                           "linewidth": float(line.get_linewidth()), "marker": line.get_marker()})
            for collection_index, collection in enumerate(axis.collections):
                prefix = f"plot.figure_{number:02d}.axis_{axis_index}.collection_{collection_index}"
                add(prefix + ".offsets", collection.get_offsets(), "axis units", "Matplotlib collection offsets")
                add(prefix + ".facecolors", collection.get_facecolors(), "RGBA", "Matplotlib collection colors")
                axis_item["collections"].append({"array_prefix": prefix, "label": collection.get_label()})
            for patch_index, artist in enumerate(axis.patches):
                prefix = f"plot.figure_{number:02d}.axis_{axis_index}.patch_{patch_index}"
                add(prefix + ".vertices", artist.get_path().vertices, "patch coordinates", "Matplotlib patch path")
                add(prefix + ".transform", artist.get_patch_transform().get_matrix(), "patch transform", "Matplotlib patch transform")
                axis_item["patches"].append({"array_prefix": prefix, "label": artist.get_label(),
                                            "facecolor": list(artist.get_facecolor()), "alpha": artist.get_alpha()})
            item["axes"].append(axis_item)
        plots["figures"].append(item)

    # The original callback is exercised for representative frames; no movie or
    # GUI timing is claimed as validated by this noninteractive capture.
    animation_callback = state["animation_update"] if "animation_update" in state else state["update"]
    for frame in (0, 20, 100, 200):
        artist = animation_callback(frame)[0]
        add(f"animation.frame_{frame:03d}.x_m", artist.get_xdata(), "m", "original update(frame)")
        add(f"animation.frame_{frame:03d}.y_m", artist.get_ydata(), "m", "original update(frame)")
    plt.close("all")
    after = protected_hashes(source)
    if before != after:
        raise RuntimeError("An existing protected file changed during capture")
    np.testing.assert_array_equal(state["constant_error_history"], state["trajectory_ff_error_history"])

    np.savez_compressed(output / "arrays.npz", **arrays)
    write_json(output / "arrays_manifest.json", manifest)
    write_json(output / "summary.json", summary)
    write_json(output / "plots.json", plots)
    (output / "stdout.txt").write_text(stdout.getvalue(), encoding="utf-8")
    (output / "stderr.txt").write_text(stderr.getvalue(), encoding="utf-8")
    environment = {
        "schema_version": 1, "status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git("rev-parse", "HEAD"), "git_branch": git("branch", "--show-current"),
        "git_status_at_capture": git("status", "--short"),
        "source_file": str(source.relative_to(ROOT)), "source_sha256": before[str(source.relative_to(ROOT))],
        "source_git_blob": git("hash-object", "--path=main.py", str(source)),
        "source_matches_commit": subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(source)], cwd=ROOT).returncode == 0,
        "versions": {"Python": platform.python_version(), "NumPy": np.__version__, "Matplotlib": matplotlib.__version__},
        "parameters": {"L1_m": float(state["L1"]), "L2_m": float(state["L2"]),
                       "tracking_kp": float(state["tracking_kp"]), "tracking_ki": float(state["tracking_ki"]),
                       "tracking_kd": float(state["tracking_kd"]), "tracking_inertia_kg_m2": float(state["tracking_inertia"]),
                       "tracking_damping_nm_s_rad": float(state["tracking_damping"]),
                       "hold_dt_s": float(state["hold_dt"]), "quintic_duration_s": float(state["T"])},
        "platform": platform.platform(), "python_executable": sys.executable, "backend": matplotlib.get_backend(),
        "execution": {"method": "runpy.run_path on the unchanged source; line/return tracing copies observations",
                      "gui_window_lift_attributes": "no-op dummy window on the Agg manager", "plt_show": "no-op",
                      "image_exports": saves, "interactive_gui_validated": False},
        "protected_sha256_before": before, "protected_sha256_after": after,
        "array_count": len(arrays), "figure_count": len(plots["figures"]),
        "checkpoint_lines": {stage: line for line, stage in lines.items()},
        "observed_loop_lines": {prefix: line for line, prefix in loop_lines.items()},
        "pid_sweep_run_count": len(pid_runs), "rk2_run_count": len(rk2_runs),
        "artifact_sha256": {path.name: sha256(path) for path in sorted(output.iterdir()) if path.is_file()},
    }
    write_json(output / "environment.json", environment)
    print(json.dumps({"output": str(output), "constant": summary["constant"], "quintic": summary["quintic"],
                      "arrays": len(arrays), "figures": len(plots["figures"]), "protected_files_unchanged": True}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "tests" / "baseline" / "reference")
    args = parser.parse_args()
    capture(args.output.resolve(), ROOT / "main.py")


if __name__ == "__main__":
    main()
