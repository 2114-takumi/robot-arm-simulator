"""Step 8-15: どちらかの関節の飽和時に両関節の積分更新を停止する。"""

import argparse
import csv
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
from robot_arm.computed_torque_integral import computed_torque_integral
from robot_arm.dynamics import integrate_dynamics
from scripts.run_ctc_integral_demo import update_integral, INTEGRAL_LIMIT
from scripts.run_ctc_torque_saturation_demo import (
    GAINS, MODELS, LIMITS, GROUPS, DT, DURATION, TARGET, SAT_ATOL, SAT_RTOL,
    INTEGRAL_ATOL, CSV_HEADER as PREVIOUS_CSV_HEADER, model_parameters,
    metrics as previous_metrics, target_equilibrium,
)
from scripts.run_kd_sweep_demo import meets_joint2_goal

METHODS = ("standard", "conditional")
CASES = tuple((method, gain, model, limit) for method in METHODS
              for gain, model in GROUPS for limit in LIMITS)
CSV_HEADER = PREVIOUS_CSV_HEADER + ",integral_update_stopped"
COMPARE_KEYS = ("overshoot_percent", "settling_time_s", "final_abs_angle_error_rad",
                "max_abs_integral_rad_s", "max_abs_command_torque_Nm", "max_abs_actual_torque_Nm",
                "saturation_fraction", "longest_saturation_time_s")


def update_with_saturation(integral, error, command, actual, method, dt=DT):
    """同じステップの飽和で両関節を一緒に停止。返却値は(z_next, stopped)。"""
    if method not in METHODS:
        raise ValueError("Unknown integration method")
    command, actual = (np.asarray(v, dtype=float) for v in (command, actual))
    for value in (command, actual):
        if value.shape != (2,) or not np.all(np.isfinite(value)):
            raise ValueError("Command and actual torque must be finite two-component vectors")
    # 既存関数で形状・有限値・dtを検証し、通常更新の候補を計算する。
    candidate = update_integral(integral, error, dt)
    saturated = np.any(np.abs(command-actual) > SAT_ATOL + SAT_RTOL*np.abs(command))
    stopped = bool(method == "conditional" and saturated)
    return (np.array(integral, dtype=float, copy=True) if stopped else candidate), stopped


def simulate(method, gain, model, limit):
    """tau_kはz_kから計算。実トルク保持でRK2を進め、同じ段の誤差で積分更新。"""
    if method not in METHODS or gain not in GAINS or model not in MODELS or limit not in LIMITS:
        raise ValueError("Unknown method, gain, model or torque limit")
    cp, pp = model_parameters(model)
    kp, kd, ki = (np.full(2, GAINS[gain][key]) for key in ("Kp", "Kd", "Ki"))
    time = np.arange(round(DURATION / DT)+1)*DT
    states = np.zeros((len(time), 4))
    commands = np.zeros((len(time), 2))
    actual = np.zeros_like(commands)
    integrals = np.zeros_like(commands)
    stopped = np.zeros(len(time), dtype=bool)
    for i in range(len(time)):
        q, omega = states[i, :2], states[i, 2:]
        commands[i] = computed_torque_integral(q, omega, TARGET, integrals[i], kp, kd, ki, cp)
        actual[i] = commands[i] if LIMITS[limit] is None else saturate_torque(commands[i], LIMITS[limit])
        if i < len(time)-1:
            states[i+1] = integrate_dynamics(*states[i], *actual[i], DT, **pp,
                                             b1=0., b2=0., g=9.81, method="rk2")
            integrals[i+1], stopped[i] = update_with_saturation(
                integrals[i], TARGET-q, commands[i], actual[i], method)
    errors = TARGET-states[:, :2]
    data = time, states, commands, actual, errors, integrals, stopped
    if not np.all(np.isfinite(np.column_stack(data))):
        raise ValueError("Nonfinite simulation data")
    return data


def metrics(data):
    """Step 8-14の指標を再利用。停止段は実際の積分クリップ回数に含めない。"""
    values = previous_metrics(data[:6])
    _, _, _, _, errors, integrals, stopped = data
    count = int(np.count_nonzero(stopped[:-1]))
    values["integral_stopped_steps"] = count
    values["integral_stopped_fraction"] = count / (len(stopped)-1)
    values["integral_stopped_steps_per_joint"] = [count, count]
    raw = integrals[:-1] + DT*errors[:-1]
    clipped = (np.abs(raw) > INTEGRAL_LIMIT + INTEGRAL_ATOL) & ~stopped[:-1, None]
    values["integral_limits"]["clipping_update_steps"] = np.sum(clipped, axis=0).tolist()
    return values


def case_name(method, gain, model, limit):
    return f"{method}_{gain}_{model}_{limit}"


def comparison_values(values):
    return {**{k: values[k] for k in COMPARE_KEYS if k in values},
            "saturation_fraction": values["saturation"]["per_joint_fraction"],
            "longest_saturation_time_s": values["saturation"]["max_continuous_time_s"]}


def compare_pair(standard, conditional):
    """各指標を独立して比較。未整定は有限整定時間より悪いと扱う。"""
    a, b = comparison_values(standard), comparison_values(conditional)
    worse = {}
    for key in COMPARE_KEYS:
        worse[key] = []
        for old, new in zip(a[key], b[key]):
            if new is None or old is None:
                worse[key].append(new is None and old is not None)
            else:
                worse[key].append(bool(new > old + 1e-12 + 1e-10*abs(old)))
    return dict(standard=a, conditional=b, conditional_worse=worse)


def save_results(results, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    equilibrium = {model: target_equilibrium(model) for model in MODELS}
    summary = dict(
        methods=list(METHODS), gains=GAINS, gains_adjusted=False,
        gain_units=dict(Kp="1/s^2", Ki="1/s^3", Kd="1/s"),
        torque_limits_Nm={k: None if v is None else v.tolist() for k, v in LIMITS.items()},
        target_rad=TARGET.tolist(), target_velocity_rad_s=[0., 0.],
        initial_state=[0., 0., 0., 0.], initial_integral_rad_s=[0., 0.],
        dt_s=DT, duration_s=DURATION, method="rk2 midpoint", b1=0., b2=0., g=9.81,
        integral_limit_rad_s=INTEGRAL_LIMIT,
        model_definition="Controller parameters remain nominal; mismatch changes only plant m2 and I2 by factor 1.2",
        integral_update="tau_k uses z_k. After command/actual torque calculation, conditional mode freezes BOTH z components if ANY joint is saturated in step k; otherwise z_(k+1)=clip(z_k+dt*error_k,-0.5,0.5). Standard mode always uses this clipped update. z_(k+1) is first used in step k+1.",
        saturation_definition="abs(command-actual)>atol+rtol*abs(command), any joint triggers both-joint freeze; exclude final diagnostic row; denominator=8000 applied control steps",
        saturation_atol_Nm=SAT_ATOL, saturation_rtol=SAT_RTOL,
        torque_sampling="Only actual torque enters dynamics and is held through RK2 midpoint. Final torque row is diagnostic, not applied. Final stopped flag is false (no update).",
        stopped_definition="Number of intentionally frozen updates, NOT number of unchanged integral samples; standard mode count=0. Both joints have the same stopped count.",
        integral_limit_definition="Entry count counts transitions to either boundary; at-limit samples may include frozen steps. Clipping count excludes frozen steps. Boundary tolerance=1e-12 rad s.",
        settling_definition="Per joint: abs(error)<=0.02*max(abs(initial_error),0.01 rad) for all remaining samples through 20 s; null if unsettled",
        overshoot_definition="100*max(0,max((q-target)*sign(target)))/abs(target), initial q=0",
        torque_squared_integral_definition="Sum of actual held torque squared times interval duration; (N m)^2 s, not electrical energy",
        regression_definition="Each lower-is-better metric compared separately; conditional worse if new>old+1e-12+1e-10*abs(old). Unsettled exceeds any finite settling time. No combined score.",
        interpretation="Conditional integration cannot supply unavailable static holding torque. Finite 20-second errors/settling do not prove asymptotic stability or guarantee physical robot performance.",
        target_equilibrium=equilibrium, cases={}, comparisons={},
    )
    rows = []
    for case, data in results.items():
        method, gain, model, limit = case
        name = case_name(*case)
        cp, pp = model_parameters(model)
        values = metrics(data)
        feasible = equilibrium[model]["limits"][limit]["static_hold_possible"]
        summary["cases"][name] = dict(
            integration_method=method, gain=gain, model=model, limit=limit,
            controller_params=cp, plant_params=pp, samples=len(data[0]), static_hold_possible=feasible,
            joint2_step813_goal_met=meets_joint2_goal(values), **values)
        np.savetxt(output / (name+".csv"), np.column_stack(data), delimiter=",", comments="", header=CSV_HEADER)
        for j in (0, 1):
            row = dict(method=method, gain=gain, model=model, limit=limit, joint=j+1,
                       static_hold_possible=feasible, integral_stopped_steps=values["integral_stopped_steps"],
                       integral_stopped_fraction=values["integral_stopped_fraction"])
            row.update({k: v[j] for k, v in values.items() if isinstance(v, list)})
            row.update({"saturation_"+k: v[j] for k, v in values["saturation"].items() if isinstance(v, list)})
            row.update({"integral_"+k: v[j] for k, v in values["integral_limits"].items()})
            rows.append(row)
    write_csv(output / "metrics.csv", rows)
    comparisons = []
    for gain, model in GROUPS:
        for limit in LIMITS:
            a, b = (summary["cases"][case_name(method, gain, model, limit)] for method in METHODS)
            compared = compare_pair(a, b)
            compared["static_hold_possible"] = a["static_hold_possible"]
            summary["comparisons"][f"{gain}_{model}_{limit}"] = compared
            for j in (0, 1):
                row = dict(gain=gain, model=model, limit=limit, joint=j+1)
                for section in ("standard", "conditional", "conditional_worse"):
                    row.update({section+"_"+k: v[j] for k, v in compared[section].items()})
                comparisons.append(row)
    write_csv(output / "comparison.csv", comparisons)
    save_plots(results, output)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return summary


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def draw_response(ax, data, index, joint, method):
    color = "tab:blue" if method == "standard" else "tab:orange"
    if index == 2:
        for values, label, style in ((data[2], "command", "--"), (data[3], "actual", "-")):
            ax.step(data[0], np.r_[values[:-1, joint], values[-2, joint]], where="post",
                    label=f"{method} {label}", color=color, linestyle=style)
    else:
        ax.plot(data[0], data[index][:, joint], label=method, color=color,
                linestyle="-" if method == "standard" else "--")


def decorate_axis(ax, index, joint, limit):
    if index == 5:
        for bound in (-INTEGRAL_LIMIT, INTEGRAL_LIMIT):
            ax.axhline(bound, color="gray", linestyle=":")
    elif index == 2:
        if LIMITS[limit] is not None:
            for bound in (-LIMITS[limit][joint], LIMITS[limit][joint]):
                ax.axhline(bound, color="gray", linestyle=":")
    else:
        ax.axhline(TARGET[joint] if index == 1 else 0., color="black", linestyle=":", label="Target")
        if index == 4:
            ax.axhspan(-.02*abs(TARGET[joint]), .02*abs(TARGET[joint]), color="gray", alpha=.15)
    ax.grid(True)
    ax.legend(fontsize=8)
    ax.set_xlabel("Time [s]")


def save_plots(results, output):
    plots = (("angles", 1, "Angle [rad]"), ("errors", 4, "Target - angle [rad]"),
             ("integrals", 5, "Integral [rad s]"), ("torques", 2, "Torque [N m]"))
    for limit in LIMITS:
        for filename, index, ylabel in plots:
            fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True, layout="constrained")
            for col, (gain, model) in enumerate(GROUPS):
                for j in (0, 1):
                    ax = axes[j, col]
                    for method in METHODS:
                        draw_response(ax, results[method, gain, model, limit], index, j, method)
                    decorate_axis(ax, index, j, limit)
                    ax.set(title=f"{gain}, {model}, Joint {j+1}", ylabel=ylabel)
            fig.suptitle(f"{limit}: standard vs conditional integration")
            fig.savefig(output / f"{filename}_{limit}.png", dpi=150)
            plt.close(fig)
    fig, axes = plt.subplots(4, 2, figsize=(13, 13), sharex=True, layout="constrained")
    for row, (_, index, ylabel) in enumerate(plots):
        for j in (0, 1):
            ax = axes[row, j]
            for method in METHODS:
                draw_response(ax, results[method, "B", "mismatch", "moderate"], index, j, method)
            decorate_axis(ax, index, j, "moderate")
            ax.set(title=f"Joint {j+1}", ylabel=ylabel, xlim=(0., 2.))
    fig.suptitle("Focus: B, mismatch +20%, limits [20, 4] — first 2 seconds")
    fig.savefig(output / "focus_B_mismatch_moderate.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "step_8_15")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    results = {}
    for case in CASES:
        results[case] = simulate(*case)
        print(f"Completed {case}", flush=True)
    summary = save_results(results, args.output)
    for name, v in summary["cases"].items():
        print(name, "overshoot:", v["overshoot_percent"], "settling:", v["settling_time_s"],
              "stopped:", v["integral_stopped_steps"])
    print("Output:", args.output.resolve())


if __name__ == "__main__":
    main()
