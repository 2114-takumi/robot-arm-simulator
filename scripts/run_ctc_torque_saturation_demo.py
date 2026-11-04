"""Step 8-14: 固定ゲインCTC+Iのトルク飽和と積分蓄積を12条件で検証。"""

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
from robot_arm.dynamics import gravity_vector, forward_dynamics, integrate_dynamics
from scripts.run_ctc_integral_demo import update_integral, INTEGRAL_LIMIT
from scripts.run_computed_torque_demo import TARGET, DT
from scripts.run_model_mismatch_demo import model_parameters
from scripts.run_robustness_demo import GAINS, metrics as response_metrics
from scripts.run_torque_saturation_demo import LIMITS, SAT_ATOL, SAT_RTOL, saturation_statistics
from scripts.run_kd_sweep_demo import meets_joint2_goal

DURATION = 20.
MODELS = ("baseline", "mismatch")
GROUPS = tuple((gain, model) for gain in GAINS for model in MODELS)
CASES = tuple((gain, model, limit) for gain, model in GROUPS for limit in LIMITS)
INTEGRAL_ATOL = 1e-12
CSV_HEADER = (
    "time_s,theta1_rad,theta2_rad,omega1_rad_s,omega2_rad_s,"
    "tau_command1_Nm,tau_command2_Nm,tau_actual1_Nm,tau_actual2_Nm,"
    "error1_rad,error2_rad,integral1_rad_s,integral2_rad_s"
)


def simulate(gain, model, limit):
    """要求・実トルクを分離。tau_kにはz_k、積分更新にはerror_kを使用。"""
    if gain not in GAINS or model not in MODELS or limit not in LIMITS:
        raise ValueError("Unknown gain, model or torque limit")
    cp, pp = model_parameters(model)
    kp, kd, ki = (np.full(2, GAINS[gain][key]) for key in ("Kp", "Kd", "Ki"))
    time = np.arange(round(DURATION / DT) + 1) * DT
    states = np.zeros((len(time), 4))
    commands = np.zeros((len(time), 2))
    actual = np.zeros_like(commands)
    integrals = np.zeros_like(commands)
    for i in range(len(time)):
        q, omega = states[i, :2], states[i, 2:]
        commands[i] = computed_torque_integral(q, omega, TARGET, integrals[i], kp, kd, ki, cp)
        actual[i] = commands[i] if LIMITS[limit] is None else saturate_torque(commands[i], LIMITS[limit])
        if i < len(time) - 1:
            states[i + 1] = integrate_dynamics(
                *states[i], *actual[i], DT, **pp, b1=0., b2=0., g=9.81, method="rk2")
            # 飽和の有無にかかわらず同じ積分更新。アンチワインドアップは追加しない。
            integrals[i + 1] = update_integral(integrals[i], TARGET - q, DT)
    errors = TARGET - states[:, :2]
    data = time, states, commands, actual, errors, integrals
    if not np.all(np.isfinite(np.column_stack(data))):
        raise ValueError("Nonfinite simulation data")
    return data


def saturation_mask(commands, actual):
    """終端診断行を除外。Step 8-4と同じ許容誤差で判定。"""
    return np.abs(commands[:-1] - actual[:-1]) > SAT_ATOL + SAT_RTOL * np.abs(commands[:-1])


def longest_run_steps(mask):
    """Trueが連続する最大ステップ数。途中で解消した区間を連結しない。"""
    padded = np.r_[False, np.asarray(mask, dtype=bool), False].astype(int)
    starts = np.flatnonzero(np.diff(padded) == 1)
    ends = np.flatnonzero(np.diff(padded) == -1)
    return int(np.max(ends - starts)) if len(starts) else 0


def integral_statistics(integrals, errors, saturated):
    """到達回数は境界への進入回数。境界に留まる更新数・clip回数とは区別。"""
    at_limit = np.isclose(np.abs(integrals), INTEGRAL_LIMIT, atol=INTEGRAL_ATOL, rtol=0.)
    boundary = np.where(at_limit, np.sign(integrals), 0.)
    entries = (boundary[1:] != 0.) & (boundary[1:] != boundary[:-1])
    raw = integrals[:-1] + DT * errors[:-1]
    clipped = np.abs(raw) > INTEGRAL_LIMIT + INTEGRAL_ATOL
    growing = np.abs(integrals[1:]) > np.abs(integrals[:-1]) + INTEGRAL_ATOL
    return dict(
        limit_entry_count=np.sum(entries, axis=0).tolist(),
        at_limit_update_steps=np.sum(at_limit[1:], axis=0).tolist(),
        clipping_update_steps=np.sum(clipped, axis=0).tolist(),
        abs_increase_during_joint_saturation_steps=np.sum(growing & saturated, axis=0).tolist(),
        abs_increase_during_any_saturation_steps=np.sum(growing & np.any(saturated, axis=1)[:, None], axis=0).tolist(),
    )


def metrics(data):
    time, states, commands, actual, errors, integrals = data
    values = response_metrics((time, states, actual, errors, integrals), TARGET)
    values["max_abs_actual_torque_Nm"] = values.pop("max_abs_torque_Nm")
    values["max_abs_command_torque_Nm"] = np.max(np.abs(commands[:-1]), axis=0).tolist()
    values["max_abs_command_actual_difference_Nm"] = np.max(np.abs(commands[:-1]-actual[:-1]), axis=0).tolist()
    values["final_integral_rad_s"] = integrals[-1].tolist()
    saturated = saturation_mask(commands, actual)
    stats = saturation_statistics(commands, actual)
    stats["max_continuous_steps"] = [longest_run_steps(saturated[:, j]) for j in (0, 1)]
    stats["max_continuous_time_s"] = [n * DT for n in stats["max_continuous_steps"]]
    stats["any_joint_max_continuous_time_s"] = longest_run_steps(np.any(saturated, axis=1)) * DT
    values["saturation"] = stats
    values["integral_limits"] = integral_statistics(integrals, errors, saturated)
    return values


def target_equilibrium(model):
    """実プラントのg(q_target)を使った静止保持の必要条件。"""
    _, pp = model_parameters(model)
    gravity = gravity_vector(*TARGET, **{k: pp[k] for k in ("L1", "m1", "m2", "lc1", "lc2")}, g=9.81)
    result = dict(plant_params=pp, required_gravity_torque_Nm=gravity.tolist(), limits={})
    for name, bound in LIMITS.items():
        feasible = np.ones(2, dtype=bool) if bound is None else np.abs(gravity) <= bound
        actual = gravity.copy() if bound is None else saturate_torque(gravity, bound)
        result["limits"][name] = dict(
            feasible_per_joint=feasible.tolist(), static_hold_possible=bool(np.all(feasible)),
            clipped_gravity_torque_Nm=actual.tolist(),
            acceleration_with_clipped_gravity_rad_s2=forward_dynamics(*TARGET, 0., 0., *actual, **pp).tolist(),
        )
    return result


def case_name(gain, model, limit):
    return f"{gain}_{model}_{limit}"


def save_results(results, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    equilibrium = {m: target_equilibrium(m) for m in MODELS}
    summary = dict(
        gains=GAINS, gain_units=dict(Kp="1/s^2", Ki="1/s^3", Kd="1/s"), gains_adjusted=False,
        torque_limits_Nm={k: None if v is None else v.tolist() for k, v in LIMITS.items()},
        initial_state=[0., 0., 0., 0.], initial_integral_rad_s=[0., 0.],
        target_rad=TARGET.tolist(), target_velocity_rad_s=[0., 0.],
        dt_s=DT, duration_s=DURATION, method="rk2 midpoint", b1=0., b2=0., g=9.81,
        integral_limit_rad_s=INTEGRAL_LIMIT, anti_windup_added=False,
        integral_update="tau_k uses z_k; z_(k+1)=clip(z_k+dt*error_k,-0.5,0.5), including during saturation",
        torque_sampling="Only actual torque enters dynamics, held throughout RK2; final torque row is diagnostic, excluded from saturation counts, maxima and applied torque squared integral",
        saturation_definition="abs(command-actual)>atol+rtol*abs(command); denominator=8000 control steps; per-joint and any-joint fractions separate; continuous time is longest uninterrupted saturated run times dt",
        saturation_atol_Nm=SAT_ATOL, saturation_rtol=SAT_RTOL,
        integral_atol_rad_s=INTEGRAL_ATOL,
        integral_limit_count_definition="Entry count: transitions into either +/-0.5 boundary, including a boundary-sign change; consecutive boundary samples count once. at_limit_update_steps counts z_(k+1) at a boundary; clipping_update_steps counts raw Euler updates beyond the bound plus tolerance. Initial z_0 excluded.",
        integral_growth_definition="Steps with abs(z_(k+1))>abs(z_k)+1e-12 during same-joint or any-joint saturation; descriptive accumulation evidence, not proof of an anti-windup counterfactual",
        settling_definition="Per joint: abs(error)<=0.02*max(abs(initial_error),0.01 rad) for all remaining samples through 20 s; null if unsettled",
        overshoot_definition="100*max(0,max((q-target)*sign(target)))/abs(target), initial q=0",
        torque_squared_integral_definition="Applied actual torque: sum(tau_k^2*(t_(k+1)-t_k)); (N m)^2 s, not electrical energy",
        interpretation="Static holding feasibility is an actuator condition, not a convergence guarantee. Strictly limited integral accumulation can still affect transients. No asymptotic convergence or physical robot performance claim from this finite simulation.",
        target_equilibrium=equilibrium, cases={},
    )
    rows = []
    for case, data in results.items():
        gain, model, limit = case
        name = case_name(*case)
        cp, pp = model_parameters(model)
        values = metrics(data)
        feasible = equilibrium[model]["limits"][limit]["static_hold_possible"]
        summary["cases"][name] = dict(
            gain=gain, model=model, limit=limit, controller_params=cp, plant_params=pp,
            samples=len(data[0]), static_hold_possible=feasible,
            both_joints_settled_within_20s=all(t is not None for t in values["settling_time_s"]),
            joint2_step813_goal_met=meets_joint2_goal(values), **values)
        np.savetxt(output / (name + ".csv"), np.column_stack(data), delimiter=",", comments="", header=CSV_HEADER)
        for j in (0, 1):
            row = dict(gain=gain, model=model, limit=limit, joint=j+1, static_hold_possible=feasible)
            row.update({k: v[j] for k, v in values.items() if isinstance(v, list)})
            row.update({"saturation_"+k: v[j] for k, v in values["saturation"].items() if isinstance(v, list)})
            row.update({"integral_"+k: v[j] for k, v in values["integral_limits"].items()})
            rows.append(row)
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    save_plots(results, output)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return summary


def save_plots(results, output):
    colors = {"unlimited": "tab:blue", "moderate": "tab:orange", "strict": "tab:red"}
    for filename, index, ylabel in (("angles.png", 1, "Angle [rad]"),
                                    ("errors.png", 4, "Target - angle [rad]"),
                                    ("integrals.png", 5, "Integral [rad s]")):
        fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True, layout="constrained")
        for col, (gain, model) in enumerate(GROUPS):
            for j in (0, 1):
                ax = axes[j, col]
                for limit in LIMITS:
                    data = results[gain, model, limit]
                    ax.plot(data[0], data[index][:, j], label=limit, color=colors[limit])
                if filename == "integrals.png":
                    ax.axhline(.5, color="black", linestyle=":", label="Integral limits")
                    ax.axhline(-.5, color="black", linestyle=":")
                else:
                    ax.axhline(TARGET[j] if filename == "angles.png" else 0., color="black", linestyle="--", label="Target")
                    if filename == "errors.png":
                        ax.axhspan(-.02*abs(TARGET[j]), .02*abs(TARGET[j]), color="gray", alpha=.2)
                ax.set(title=f"{gain}, {model}, Joint {j+1}", xlabel="Time [s]", ylabel=ylabel)
                ax.grid(True)
                ax.legend(fontsize=8)
        fig.savefig(output / filename, dpi=150)
        plt.close(fig)
    for gain, model in GROUPS:
        fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True, layout="constrained")
        for row, limit in enumerate(LIMITS):
            time, _, commands, actual, _, _ = results[gain, model, limit]
            for j in (0, 1):
                ax = axes[row, j]
                for values, label, style in ((commands, "Command", "--"), (actual, "Actual", "-")):
                    held = np.r_[values[:-1, j], values[-2, j]]
                    ax.step(time, held, where="post", label=label, linestyle=style)
                if LIMITS[limit] is not None:
                    bound = LIMITS[limit][j]
                    ax.axhline(bound, color="gray", linestyle=":", label="Limits")
                    ax.axhline(-bound, color="gray", linestyle=":")
                ax.set(title=f"{limit}, Joint {j+1}", ylabel="Torque [N m]", xlabel="Time [s]")
                ax.grid(True)
                ax.legend()
        fig.suptitle(f"Gain {gain}, {model}: requested and applied torque")
        fig.savefig(output / f"torques_{gain}_{model}.png", dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "step_8_14")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    results = {}
    for case in CASES:
        results[case] = simulate(*case)
        print(f"Completed {case}", flush=True)
    summary = save_results(results, args.output)
    for name, v in summary["cases"].items():
        print(name, "error:", v["final_angle_error_rad"], "settling:", v["settling_time_s"],
              "saturation:", v["saturation"]["per_joint_fraction"])
    print("Output:", args.output.resolve())


if __name__ == "__main__":
    main()
