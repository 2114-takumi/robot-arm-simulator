"""Step 8-17: CTC+Iの逆算補正を5候補で検証し、既存3方式と比較する。"""

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
from robot_arm.dynamics import mass_matrix, integrate_dynamics
from scripts.run_ctc_integral_demo import update_integral, INTEGRAL_LIMIT
from scripts.run_per_joint_anti_windup_demo import (
    METHODS, GAINS, MODELS, LIMITS, GROUPS, DT, DURATION, TARGET, SAT_ATOL, SAT_RTOL,
    INTEGRAL_ATOL, model_parameters, target_equilibrium, compare_methods,
    CSV_HEADER as PREVIOUS_CSV_HEADER, simulate as previous_simulate,
    metrics as previous_metrics, LABELS as PREVIOUS_LABELS,
)
from scripts.run_kd_sweep_demo import meets_joint2_goal

KAW_VALUES = (0.5, 1., 2., 5., 10.)
BACK_CALCULATION = {f"back_calculation_{k:g}": k for k in KAW_VALUES}
VARIANTS = (*METHODS, *BACK_CALCULATION)
CASES = tuple((variant, gain, model, limit) for variant in VARIANTS
              for gain, model in GROUPS for limit in LIMITS)
LABELS = {**PREVIOUS_LABELS, **{v: f"BC Kaw={k:g}" for v, k in BACK_CALCULATION.items()}}
MASS_KEYS = ("L1", "m1", "m2", "lc1", "lc2", "I1", "I2")
# Operational numerical-runaway guard, NOT a physical joint limit or a stability theorem.
STATE_GUARD = 1e6  # absolute angle [rad] or angular velocity [rad/s]
CSV_HEADER = PREVIOUS_CSV_HEADER + (
    ",delta_tau1_Nm,delta_tau2_Nm,back_calculation_correction1_rad,back_calculation_correction2_rad"
)


def vector(value):
    value = np.asarray(value, dtype=float)
    if value.shape != (2,) or not np.all(np.isfinite(value)):
        raise ValueError("Expected a finite two-component vector")
    return value


def validate_gains(ki, kaw):
    ki = vector(ki)
    kaw = np.asarray(kaw, dtype=float)
    if np.any(ki <= 0) or kaw.ndim != 0 or not np.isfinite(kaw) or kaw < 0:
        raise ValueError("Ki must be strictly positive; Kaw must be a finite nonnegative scalar")
    return ki, float(kaw)


def back_calculation_update(q, integral, error, command, actual, ki, kaw, controller_params, dt=DT):
    """z_next, delta_tau, correctionを返す。Kaw=0は一致検証用の無補正条件。

    correction = Kaw*solve(M_hat(q), actual-command)/Ki [rad]
    z_next = clip(z_k + dt*(error_k+correction_k), +/-0.5) [rad s]
    M_hatは制御器の想定モデル。逆行列は作らない。入力は変更しない。
    """
    q, integral, error, command, actual = map(vector, (q, integral, error, command, actual))
    ki, kaw = validate_gains(ki, kaw)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        delta_tau = actual-command
        matrix = mass_matrix(*q, **{key: controller_params[key] for key in MASS_KEYS})
        delta_v = np.linalg.solve(matrix, delta_tau)
        correction = kaw*delta_v/ki
        if not np.all(np.isfinite(correction)):
            raise FloatingPointError("Nonfinite back-calculation correction")
        updated = update_integral(integral, error+correction, dt)
    return updated, delta_tau, correction


class SimulationFailure(RuntimeError):
    """失敗時点と、それ以前の有限な診断行を保持する。"""

    def __init__(self, step, reason, partial_data):
        super().__init__(reason)
        self.step = step
        self.partial_data = partial_data


def simulate(kaw, gain, model, limit):
    """各段のtauにはz_kを使用。RK2内部では実トルクを固定する。"""
    if gain not in GAINS or model not in MODELS or limit not in LIMITS:
        raise ValueError("Unknown gain, model or limit")
    cp, pp = model_parameters(model)
    kp, kd, ki = (np.full(2, GAINS[gain][key]) for key in ("Kp", "Kd", "Ki"))
    _, kaw = validate_gains(ki, kaw)
    time = np.arange(round(DURATION/DT)+1)*DT
    states = np.zeros((len(time), 4))
    commands = np.zeros((len(time), 2))
    actual = np.zeros_like(commands)
    integral = np.zeros_like(commands)
    stopped = np.zeros_like(commands, dtype=bool)  # BC corrects; it does not freeze updates.
    errors = np.zeros_like(commands)
    delta_tau = np.zeros_like(commands)
    correction = np.zeros_like(commands)
    data = time, states, commands, actual, errors, integral, stopped, delta_tau, correction
    valid_rows = 0
    for i in range(len(time)):
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                if not np.all(np.isfinite(states[i])) or np.any(np.abs(states[i]) > STATE_GUARD):
                    raise FloatingPointError("Nonfinite state or numerical state guard exceeded")
                q, omega = states[i, :2], states[i, 2:]
                errors[i] = TARGET-q
                commands[i] = computed_torque_integral(q, omega, TARGET, integral[i], kp, kd, ki, cp)
                actual[i] = commands[i] if LIMITS[limit] is None else saturate_torque(commands[i], LIMITS[limit])
                z_next, delta_tau[i], correction[i] = back_calculation_update(
                    q, integral[i], errors[i], commands[i], actual[i], ki, kaw, cp)
                if not np.all(np.isfinite(np.concatenate([a[i:i+1].ravel() for a in data]))):
                    raise FloatingPointError("Nonfinite diagnostic row")
                valid_rows = i+1
                if i < len(time)-1:
                    states[i+1] = integrate_dynamics(*states[i], *actual[i], DT, **pp,
                                                     b1=0., b2=0., g=9.81, method="rk2")
                    integral[i+1] = z_next
        except (FloatingPointError, ValueError, np.linalg.LinAlgError, OverflowError) as exc:
            partial = tuple(a[:valid_rows].copy() for a in data)
            raise SimulationFailure(i, f"{type(exc).__name__}: {exc}", partial) from exc
    return data


def case_name(variant, gain, model, limit):
    return f"{variant}_{gain}_{model}_{limit}"


def run_case(case):
    """失敗条件も返却し、後続候補の実験を継続できるようにする。"""
    variant, gain, model, limit = case
    if variant not in VARIANTS:
        raise ValueError("Unknown variant")
    try:
        if variant in METHODS:
            previous = previous_simulate(variant, gain, model, limit)
            # 既存3方式の数値計算はそのまま再利用。BC補正項は適用しない。
            data = (*previous, previous[3]-previous[2], np.zeros_like(previous[2]))
        else:
            data = simulate(BACK_CALCULATION[variant], gain, model, limit)
        return dict(status="ok", data=data)
    except SimulationFailure as exc:
        return dict(status="failed", data=exc.partial_data, failed_step=exc.step,
                    failed_time_s=exc.step*DT, reason=str(exc))


def metrics(data):
    values = previous_metrics(data[:7])
    time, _, _, _, errors, integral, stopped, delta, correction = data
    raw = integral[:-1]+DT*(errors[:-1]+correction[:-1])
    clipped = (np.abs(raw) > INTEGRAL_LIMIT+INTEGRAL_ATOL) & ~stopped[:-1]
    values["integral_limits"]["clipping_update_steps"] = clipped.sum(axis=0).tolist()
    values["max_abs_correction_rad"] = np.max(np.abs(correction[:-1]), axis=0).tolist()
    values["max_abs_delta_tau_Nm"] = np.max(np.abs(delta[:-1]), axis=0).tolist()
    values["delta_tau_squared_integral_Nm2_s"] = np.sum(delta[:-1]**2*np.diff(time)[:, None], axis=0).tolist()
    return values


def write_rows(path, rows):
    # 失敗行には有限時間指標を捏造せず空欄を使う。
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_results(results, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    equilibrium = {model: target_equilibrium(model) for model in MODELS}
    summary = dict(
        variants=LABELS, Kaw_candidates_s_inverse=list(KAW_VALUES), candidates_are_optimized=False,
        gains=GAINS, gain_units=dict(Kp="s^-2", Ki="s^-3", Kd="s^-1", Kaw="s^-1"),
        target_rad=TARGET.tolist(), initial_state=[0.]*4, initial_integral_rad_s=[0., 0.],
        dt_s=DT, duration_s=DURATION, integrator="RK2 midpoint", b1=0., b2=0., g=9.81,
        integral_limit_rad_s=INTEGRAL_LIMIT,
        torque_limits_Nm={k: None if v is None else v.tolist() for k, v in LIMITS.items()},
        model_definition="Controller nominal. Only plant m2 and I2 multiplied by 1.2 for mismatch.",
        back_calculation="delta_tau=actual-command; delta_v=solve(M_hat(q),delta_tau); correction=Kaw*delta_v/Ki, elementwise Ki division; z_next=clip(z+dt*(error+correction),-0.5,0.5). Kaw=0 allowed only as validation ablation.",
        correction_units="rad; delta_v in rad/s^2; integral state in rad*s",
        timing="tau_k uses z_k; correction and error use step k; z_(k+1) first used at k+1. Only actual torque enters dynamics, held at RK2 midpoint. Final torque/delta/correction row diagnostic, not applied, excluded from step maxima and time integrals.",
        previous_methods="Step 8-16 simulation reused unchanged; BC correction columns zero for these methods. Integral stopped flags retain their original meaning.",
        saturation_definition="abs(command-actual)>atol+rtol*abs(command); denominator=8000 control steps, final diagnostic row excluded. BC uses the raw torque difference, without a tolerance deadband.",
        saturation_atol_Nm=SAT_ATOL, saturation_rtol=SAT_RTOL,
        integral_limit_definition="Entries count transitions into +/-0.5 boundaries (tolerance 1e-12). Clipping counts corrected pre-clamp updates; frozen updates excluded.",
        settling_definition="abs(error)<=0.02*max(abs(initial_error),0.01 rad) for all remaining samples, per joint. null means unsettled within 20 s.",
        overshoot_definition="100*max(0,max((q-target)*sign(target)))/abs(target), initial q=0",
        squared_integral_definition="Sum of held-interval squared torque/delta times dt, excluding diagnostic final row; (N m)^2 s, not electrical energy.",
        goal="Joint 2 overshoot <20% AND settling <=10 s; Joint 1 and other conditions assessed separately.",
        comparison_definition="Each metric separate; increased if new>old+1e-12+1e-10*abs(old). Unsettled exceeds finite settling time. No arbitrary aggregate score. Increased integral is not automatically worse performance.",
        failure_definition=f"Record overflow, nonfinite values, solve failures or abs(angle/velocity)>{STATE_GUARD:g}. Operational runaway guard, not a stability test. Failed CSV contains only validated diagnostic prefix; no final metrics assigned.",
        interpretation="Back-calculation cannot increase actuator limits or restore infeasible static holding. These 20 s trials do not prove asymptotic stability, steady-state error, or hardware performance.",
        target_equilibrium=equilibrium, cases={}, comparisons={}, goal_counts={},
    )
    rows = []
    for case, record in results.items():
        variant, gain, model, limit = case
        name = case_name(*case)
        cp, pp = model_parameters(model)
        entry = dict(variant=variant, Kaw_s_inverse=BACK_CALCULATION.get(variant), gain=gain,
                     model=model, limit=limit, controller_params=cp, plant_params=pp,
                     static_hold_possible=equilibrium[model]["limits"][limit]["static_hold_possible"],
                     samples=len(record["data"][0]), **{k: v for k, v in record.items() if k != "data"})
        if record["status"] == "ok":
            entry.update(metrics(record["data"]))
            entry["joint2_goal_met"] = meets_joint2_goal(entry)
        else:
            entry["joint2_goal_met"] = False
        summary["cases"][name] = entry
        np.savetxt(output/(name+".csv"), np.column_stack(record["data"]),
                   delimiter=",", comments="", header=CSV_HEADER)
        for j in (0, 1):
            row = {k: entry[k] for k in ("variant", "Kaw_s_inverse", "gain", "model", "limit", "status", "static_hold_possible", "joint2_goal_met")}
            row["joint"] = j+1
            if entry["status"] == "ok":
                row.update({k: v[j] for k, v in entry.items() if isinstance(v, list)})
                row.update({"saturation_"+k: v[j] for k, v in entry["saturation"].items() if isinstance(v, list)})
                row.update({"integral_"+k: v[j] for k, v in entry["integral_limits"].items()})
            else:
                row.update({k: entry[k] for k in ("reason", "failed_step", "failed_time_s")})
            rows.append(row)
    write_rows(output/"metrics.csv", rows)
    write_rows(output/"focus_comparison.csv", [r for r in rows if (r["gain"], r["model"], r["limit"]) == ("B", "mismatch", "moderate")])
    comparison_rows = []
    for variant in BACK_CALCULATION:
        for gain, model in GROUPS:
            for limit in LIMITS:
                candidate = summary["cases"][case_name(variant, gain, model, limit)]
                for reference_variant in METHODS:
                    reference = summary["cases"][case_name(reference_variant, gain, model, limit)]
                    key = f"{variant}_vs_{reference_variant}_{gain}_{model}_{limit}"
                    compared = dict(status="failed" if "failed" in (reference["status"], candidate["status"]) else "ok")
                    if compared["status"] == "ok":
                        compared.update(compare_methods(reference, candidate))
                    summary["comparisons"][key] = compared
                    for j in (0, 1):
                        row = dict(candidate=variant, reference=reference_variant, gain=gain, model=model,
                                   limit=limit, joint=j+1, status=compared["status"])
                        if compared["status"] == "ok":
                            for section in ("reference", "candidate", "candidate_increased"):
                                row.update({section+"_"+k: v[j] for k, v in compared[section].items()})
                        comparison_rows.append(row)
    write_rows(output/"comparison.csv", comparison_rows)
    for variant in VARIANTS:
        entries = [v for v in summary["cases"].values() if v["variant"] == variant]
        summary["goal_counts"][variant] = dict(total=len(entries), met=sum(v["joint2_goal_met"] for v in entries),
                                              failed=sum(v["status"] == "failed" for v in entries))
    (output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    save_plots(results, output)
    return summary


def draw_axis(ax, results, gain, model, limit, index, joint):
    colors = ("#666666", "#e69f00", "#009e73", "#0072b2", "#cc79a7", "#56b4e9", "#d55e00", "#332288")
    for variant, color in zip(VARIANTS, colors):
        record = results[variant, gain, model, limit]
        data = record["data"]
        label = LABELS[variant]+(" FAILED" if record["status"] == "failed" else "")
        style = "--" if variant in METHODS else "-"
        if index in (2, 3, 7, 8) and len(data[0]) > 1:
            # Successful final row is diagnostic; failed prefixes have no claimed full final interval.
            values = data[index][:, joint].copy()
            if record["status"] == "ok":
                values[-1] = values[-2]
            ax.step(data[0], values, where="post", label=label, color=color, linestyle=style)
        else:
            ax.plot(data[0], data[index][:, joint], label=label, color=color, linestyle=style)
    if index in (1, 4):
        ax.axhline(TARGET[joint] if index == 1 else 0., color="black", linestyle=":")
        if index == 4:
            ax.axhspan(-.02*abs(TARGET[joint]), .02*abs(TARGET[joint]), color="gray", alpha=.15)
    if index == 5:
        for bound in (-INTEGRAL_LIMIT, INTEGRAL_LIMIT):
            ax.axhline(bound, color="gray", linestyle=":")
    if index in (2, 3) and LIMITS[limit] is not None:
        for bound in (-LIMITS[limit][joint], LIMITS[limit][joint]):
            ax.axhline(bound, color="gray", linestyle=":")
    ax.grid(True, alpha=.4)
    ax.set_xlabel("Time [s]")


def save_plots(results, output):
    plots = (("angles", 1, "Angle [rad]"), ("errors", 4, "Target - angle [rad]"),
             ("integrals", 5, "Integral [rad s]"), ("command_torques", 2, "Command torque [N m]"),
             ("actual_torques", 3, "Actual torque [N m]"), ("corrections", 8, "BC correction [rad]"),
             ("delta_torques", 7, "Actual - command [N m]"))
    for limit in LIMITS:
        for filename, index, ylabel in plots:
            fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharex=True, layout="constrained")
            for col, (gain, model) in enumerate(GROUPS):
                for j in (0, 1):
                    draw_axis(axes[j, col], results, gain, model, limit, index, j)
                    axes[j, col].set(title=f"{gain}, {model}, Joint {j+1}", ylabel=ylabel)
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=9)
            failed = sum(v["status"] == "failed" for k, v in results.items() if k[3] == limit)
            fig.suptitle(f"{limit}: 3 references and 5 Kaw candidates; failed cases={failed}")
            fig.savefig(output/f"{filename}_{limit}.png", dpi=130)
            plt.close(fig)
    for end in (2, 20):
        fig, axes = plt.subplots(6, 2, figsize=(14, 18), sharex=True, layout="constrained")
        for row, (_, index, ylabel) in enumerate(plots[:6]):
            for j in (0, 1):
                draw_axis(axes[row, j], results, "B", "mismatch", "moderate", index, j)
                axes[row, j].set(title=f"Joint {j+1}", ylabel=ylabel, xlim=(0., end))
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=9)
        fig.suptitle(f"Focus: B, mismatch +20%, [20, 4] N m; first {end} s")
        fig.savefig(output/f"focus_B_mismatch_moderate_{end}s.png", dpi=130)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"outputs"/"step_8_17")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    results = {}
    for case in CASES:
        results[case] = run_case(case)
        print(case_name(*case), results[case]["status"], results[case].get("reason", ""), flush=True)
    summary = save_results(results, args.output)
    for variant in VARIANTS:
        values = summary["cases"][case_name(variant, "B", "mismatch", "moderate")]
        print(variant, values["status"], "settling:", values.get("settling_time_s"),
              "overshoot:", values.get("overshoot_percent"), "goal:", values["joint2_goal_met"])
    print("Output:", args.output.resolve())


if __name__ == "__main__":
    main()
