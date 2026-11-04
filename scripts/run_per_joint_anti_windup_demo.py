"""Step 8-16: 通常積分・全関節停止・関節別停止を同じ36条件で比較する。"""

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
from robot_arm.computed_torque_integral import computed_torque_integral
from robot_arm.dynamics import integrate_dynamics
from scripts.run_ctc_integral_demo import update_integral, INTEGRAL_LIMIT
from scripts.run_ctc_torque_saturation_demo import (
    GAINS, MODELS, LIMITS, GROUPS, DT, DURATION, TARGET, SAT_ATOL, SAT_RTOL,
    INTEGRAL_ATOL, CSV_HEADER as STATE_CSV_HEADER, model_parameters,
    metrics as previous_metrics, target_equilibrium,
)
from scripts.run_anti_windup_demo import (
    update_with_saturation as previous_update, compare_pair, write_csv, decorate_axis,
)
from scripts.run_kd_sweep_demo import meets_joint2_goal

METHODS = ("standard", "conditional", "per_joint")
LABELS = dict(standard="Standard", conditional="Global freeze", per_joint="Per-joint freeze")
COLORS = dict(zip(METHODS, ("tab:blue", "tab:orange", "tab:green")))
CASES = tuple((method, gain, model, limit) for method in METHODS
              for gain, model in GROUPS for limit in LIMITS)
CSV_HEADER = STATE_CSV_HEADER + ",integral_update_stopped1,integral_update_stopped2"


def update_with_saturation(integral, error, command, actual, method, dt=DT):
    """返却値(z_next, stopped[2])。同じ段の飽和を使い、次段の積分値を決める。"""
    if method in ("standard", "conditional"):
        updated, stopped = previous_update(integral, error, command, actual, method, dt)
        return updated, np.full(2, stopped, dtype=bool)
    if method != "per_joint":
        raise ValueError("Unknown integration method")
    command, actual = (np.asarray(value, dtype=float) for value in (command, actual))
    for value in (command, actual):
        if value.shape != (2,) or not np.all(np.isfinite(value)):
            raise ValueError("Command and actual torque must be finite two-component vectors")
    candidate = update_integral(integral, error, dt)
    stopped = np.abs(command-actual) > SAT_ATOL + SAT_RTOL*np.abs(command)
    return np.where(stopped, integral, candidate), stopped


def simulate(method, gain, model, limit):
    """tau_kにz_kを使用し、実トルク一定のRK2更新後にz_(k+1)を保存。"""
    if method not in METHODS or gain not in GAINS or model not in MODELS or limit not in LIMITS:
        raise ValueError("Unknown method, gain, model or torque limit")
    cp, pp = model_parameters(model)
    kp, kd, ki = (np.full(2, GAINS[gain][key]) for key in ("Kp", "Kd", "Ki"))
    time = np.arange(round(DURATION/DT)+1)*DT
    states = np.zeros((len(time), 4))
    commands = np.zeros((len(time), 2))
    actual = np.zeros_like(commands)
    integrals = np.zeros_like(commands)
    stopped = np.zeros_like(commands, dtype=bool)
    for i in range(len(time)):
        q, omega = states[i, :2], states[i, 2:]
        commands[i] = computed_torque_integral(q, omega, TARGET, integrals[i], kp, kd, ki, cp)
        actual[i] = commands[i] if LIMITS[limit] is None else saturate_torque(commands[i], LIMITS[limit])
        if i < len(time)-1:
            states[i+1] = integrate_dynamics(*states[i], *actual[i], DT, **pp,
                                             b1=0., b2=0., g=9.81, method="rk2")
            integrals[i+1], stopped[i] = update_with_saturation(
                integrals[i], TARGET-q, commands[i], actual[i], method)
    data = time, states, commands, actual, TARGET-states[:, :2], integrals, stopped
    if not np.all(np.isfinite(np.column_stack(data))):
        raise ValueError("Nonfinite simulation data")
    return data


def metrics(data):
    """最終行は診断用。停止回数・飽和割合の分母は実制御ステップ数。"""
    values = previous_metrics(data[:6])
    _, _, _, _, errors, integral, stopped = data
    steps = len(stopped)-1
    count = np.sum(stopped[:-1], axis=0)
    values["integral_stopped_steps_per_joint"] = count.tolist()
    values["integral_stopped_fraction_per_joint"] = (count/steps).tolist()
    values["integral_stopped_any_joint_steps"] = int(np.count_nonzero(np.any(stopped[:-1], axis=1)))
    values["integral_stopped_both_joints_steps"] = int(np.count_nonzero(np.all(stopped[:-1], axis=1)))
    raw = integral[:-1]+DT*errors[:-1]
    clipped = (np.abs(raw) > INTEGRAL_LIMIT+INTEGRAL_ATOL) & ~stopped[:-1]
    values["integral_limits"]["clipping_update_steps"] = clipped.sum(axis=0).tolist()
    return values


def compare_methods(reference, candidate):
    """各指標を別々に比較。増加は常に総合的な悪化を意味するわけではない。"""
    comparison = compare_pair(reference, candidate)
    return dict(reference=comparison["standard"], candidate=comparison["conditional"],
                candidate_increased=comparison["conditional_worse"])


def case_name(method, gain, model, limit):
    return f"{method}_{gain}_{model}_{limit}"


def save_results(results, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    equilibrium = {model: target_equilibrium(model) for model in MODELS}
    summary = dict(
        methods=LABELS, gains=GAINS, gains_adjusted=False,
        gain_units=dict(Kp="1/s^2", Ki="1/s^3", Kd="1/s"),
        torque_limits_Nm={key: None if value is None else value.tolist() for key, value in LIMITS.items()},
        target_rad=TARGET.tolist(), target_velocity_rad_s=[0., 0.], initial_state=[0.]*4,
        initial_integral_rad_s=[0., 0.], integral_limit_rad_s=INTEGRAL_LIMIT,
        dt_s=DT, duration_s=DURATION, integrator="rk2 midpoint", b1=0., b2=0., g=9.81,
        model_definition="Nominal controller; mismatch changes only plant m2 and I2 by factor 1.2.",
        integral_update="tau_k uses z_k. Compute command, actual and same-step saturation. Standard updates both; conditional freezes both if any saturated; per_joint freezes only saturated joints. Otherwise z_(j,k+1)=clip(z_(j,k)+dt*error_(j,k),-0.5,0.5). Next value first used at k+1.",
        saturation_definition="abs(command-actual)>atol+rtol*abs(command); per-joint and any-joint fractions have denominator 8000 applied control steps, excluding final diagnostic row.",
        saturation_atol_Nm=SAT_ATOL, saturation_rtol=SAT_RTOL,
        torque_sampling="Only actual torque enters dynamics, held during RK2 midpoint. Final torque row is diagnostic; final stopped flags are false (no update).",
        stopped_definition="Intentional update pauses per joint, even when error=0. Frozen updates are excluded from clipping counts. Unchanged integral alone is not a pause.",
        integral_limit_definition="Boundary entry count counts transitions; at-limit updates may include frozen steps. Boundary tolerance=1e-12 rad s.",
        settling_definition="Per joint: abs(error)<=0.02*max(abs(initial_error),0.01 rad) through all remaining samples; null if unsettled at 20 s.",
        overshoot_definition="100*max(0,max((q-target)*sign(target)))/abs(target), initial q=0.",
        torque_squared_integral_definition="Sum of actual held torque squared times interval duration; (N m)^2 s, not electrical energy.",
        comparison_definition="Each metric compared independently; increased if new>old+1e-12+1e-10*abs(old). Unsettled exceeds finite settling time. Increased integral magnitude is not automatically worse control performance. No combined score.",
        interpretation="Unavailable static holding torque cannot be fixed by anti-windup. Finite 20 s errors do not establish asymptotic steady-state errors or stability and do not guarantee hardware performance. Joint coordinates are dynamically coupled; per-joint freezing need not improve both joints.",
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
            joint2_goal_met=meets_joint2_goal(values), **values)
        np.savetxt(output/(name+".csv"), np.column_stack(data), delimiter=",", comments="", header=CSV_HEADER)
        for j in (0, 1):
            row = dict(method=method, gain=gain, model=model, limit=limit, joint=j+1,
                       static_hold_possible=feasible)
            row.update({key: value[j] for key, value in values.items() if isinstance(value, list)})
            row.update({"saturation_"+key: value[j] for key, value in values["saturation"].items() if isinstance(value, list)})
            row.update({"integral_"+key: value[j] for key, value in values["integral_limits"].items()})
            rows.append(row)
    write_csv(output/"metrics.csv", rows)
    comparison_rows = []
    for reference_method in ("standard", "conditional"):
        for gain, model in GROUPS:
            for limit in LIMITS:
                reference = summary["cases"][case_name(reference_method, gain, model, limit)]
                candidate = summary["cases"][case_name("per_joint", gain, model, limit)]
                comparison = compare_methods(reference, candidate)
                comparison["static_hold_possible"] = reference["static_hold_possible"]
                key = f"per_joint_vs_{reference_method}_{gain}_{model}_{limit}"
                summary["comparisons"][key] = comparison
                for j in (0, 1):
                    row = dict(reference_method=reference_method, candidate_method="per_joint",
                               gain=gain, model=model, limit=limit, joint=j+1)
                    for section in ("reference", "candidate", "candidate_increased"):
                        row.update({section+"_"+k: v[j] for k, v in comparison[section].items()})
                    comparison_rows.append(row)
    write_csv(output/"comparison.csv", comparison_rows)
    save_plots(results, output)
    (output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return summary


def draw_response(ax, data, index, joint, method):
    if index == 2:
        for values, label, style in ((data[2], "command", "--"), (data[3], "actual", "-")):
            ax.step(data[0], np.r_[values[:-1, joint], values[-2, joint]], where="post",
                    label=f"{LABELS[method]} {label}", color=COLORS[method], linestyle=style)
    else:
        ax.plot(data[0], data[index][:, joint], label=LABELS[method], color=COLORS[method],
                linestyle={"standard": "-", "conditional": "--", "per_joint": "-."}[method])


def save_plots(results, output):
    plots = (("angles", 1, "Angle [rad]"), ("errors", 4, "Target - angle [rad]"),
             ("integrals", 5, "Integral [rad s]"), ("torques", 2, "Torque [N m]"))
    for limit in LIMITS:
        for filename, index, ylabel in plots:
            fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharex=True, layout="constrained")
            for col, (gain, model) in enumerate(GROUPS):
                for j in (0, 1):
                    ax = axes[j, col]
                    for method in METHODS:
                        draw_response(ax, results[method, gain, model, limit], index, j, method)
                    decorate_axis(ax, index, j, limit)
                    ax.set(title=f"{gain}, {model}, Joint {j+1}", ylabel=ylabel)
            fig.suptitle(f"{limit}: standard / global freeze / per-joint freeze")
            fig.savefig(output/f"{filename}_{limit}.png", dpi=150)
            plt.close(fig)
    fig, axes = plt.subplots(4, 2, figsize=(14, 14), sharex=True, layout="constrained")
    for row, (_, index, ylabel) in enumerate(plots):
        for j in (0, 1):
            ax = axes[row, j]
            for method in METHODS:
                draw_response(ax, results[method, "B", "mismatch", "moderate"], index, j, method)
            decorate_axis(ax, index, j, "moderate")
            ax.set(title=f"Joint {j+1}", ylabel=ylabel, xlim=(0., 10.))
    fig.suptitle("Focus: B, mismatch +20%, limits [20, 4] - first 10 seconds")
    fig.savefig(output/"focus_B_mismatch_moderate.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"outputs"/"step_8_16")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    results = {}
    for case in CASES:
        results[case] = simulate(*case)
        print(f"Completed {case}", flush=True)
    summary = save_results(results, args.output)
    for name, values in summary["cases"].items():
        print(name, "overshoot:", values["overshoot_percent"], "settling:", values["settling_time_s"],
              "stopped:", values["integral_stopped_steps_per_joint"])
    print("Output:", args.output.resolve())


if __name__ == "__main__":
    main()
