"""Step 8-18: 関節別Kawの25組探索と、選択候補の他条件での検証。"""

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
from robot_arm.dynamics import mass_matrix, integrate_dynamics
from scripts.run_ctc_integral_demo import update_integral
from scripts import run_back_calculation_demo as previous
from scripts.run_back_calculation_demo import (
    GAINS, MODELS, LIMITS, GROUPS, DT, DURATION, TARGET, INTEGRAL_LIMIT,
    MASS_KEYS, STATE_GUARD, CSV_HEADER, KAW_VALUES, model_parameters,
    target_equilibrium, metrics, compare_methods, write_rows, vector, SimulationFailure,
)

VECTORS = {f"per_joint_{a:g}_{b:g}": np.array([a, b]) for a in KAW_VALUES for b in KAW_VALUES}
REFERENCES = ("standard", "conditional", "per_joint", "back_calculation_5", "back_calculation_10")
FOCUS = ("B", "mismatch", "moderate")
# Division-before-multiplication differs from Step 8-17. Tolerance is explicit, not bit equality.
MATCH_ATOL = 1e-10
MATCH_RTOL = 1e-11


def validate_gains(ki, kaw):
    ki, kaw = vector(ki), vector(kaw)
    if np.any(ki <= 0) or np.any(kaw < 0):
        raise ValueError("Ki must be positive; Kaw must be a nonnegative two-component vector")
    return ki, kaw


def back_calculation_update(q, integral, error, command, actual, ki, kaw, controller_params, dt=DT):
    """指定の演算順序 Kaw_vector * (delta_v / Ki) を維持。ゼロKawは検証用。"""
    q, integral, error, command, actual = map(vector, (q, integral, error, command, actual))
    ki, kaw = validate_gains(ki, kaw)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        delta_tau = actual-command
        matrix = mass_matrix(*q, **{key: controller_params[key] for key in MASS_KEYS})
        delta_v = np.linalg.solve(matrix, delta_tau)
        correction = kaw * (delta_v / ki)
        if not np.all(np.isfinite(correction)):
            raise FloatingPointError("Nonfinite correction")
        updated = update_integral(integral, error+correction, dt)
    return updated, delta_tau, correction


def simulate(kaw, gain="B", model="mismatch", limit="moderate"):
    """tau_kはz_kを使用。補正・誤差は同時刻、z_nextは次段から使用。"""
    if gain not in GAINS or model not in MODELS or limit not in LIMITS:
        raise ValueError("Unknown gain, model or limit")
    cp, pp = model_parameters(model)
    kp, kd, ki = (np.full(2, GAINS[gain][key]) for key in ("Kp", "Kd", "Ki"))
    _, kaw = validate_gains(ki, kaw)
    time = np.arange(round(DURATION/DT)+1)*DT
    states = np.zeros((len(time), 4))
    command = np.zeros((len(time), 2))
    actual, errors, integral, delta, correction = (np.zeros_like(command) for _ in range(5))
    stopped = np.zeros_like(command, dtype=bool)
    data = time, states, command, actual, errors, integral, stopped, delta, correction
    valid_rows = 0
    for i in range(len(time)):
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                if not np.all(np.isfinite(states[i])) or np.any(np.abs(states[i]) > STATE_GUARD):
                    raise FloatingPointError("Nonfinite state or numerical state guard exceeded")
                q, omega = states[i, :2], states[i, 2:]
                errors[i] = TARGET-q
                command[i] = computed_torque_integral(q, omega, TARGET, integral[i], kp, kd, ki, cp)
                actual[i] = command[i] if LIMITS[limit] is None else saturate_torque(command[i], LIMITS[limit])
                z_next, delta[i], correction[i] = back_calculation_update(
                    q, integral[i], errors[i], command[i], actual[i], ki, kaw, cp)
                if not np.all(np.isfinite(np.concatenate([a[i:i+1].ravel() for a in data]))):
                    raise FloatingPointError("Nonfinite diagnostic row")
                valid_rows = i+1
                if i < len(time)-1:
                    states[i+1] = integrate_dynamics(*states[i], *actual[i], DT, **pp,
                                                     b1=0., b2=0., g=9.81, method="rk2")
                    integral[i+1] = z_next
        except (FloatingPointError, ValueError, np.linalg.LinAlgError, OverflowError) as exc:
            raise SimulationFailure(i, f"{type(exc).__name__}: {exc}",
                                    tuple(a[:valid_rows].copy() for a in data)) from exc
    return data


def run_case(case):
    variant, gain, model, limit = case
    if variant in REFERENCES:
        return previous.run_case(case)
    if variant not in VECTORS:
        raise ValueError("Unknown variant")
    try:
        return dict(status="ok", data=simulate(VECTORS[variant], gain, model, limit))
    except SimulationFailure as exc:
        return dict(status="failed", data=exc.partial_data, failed_step=exc.step,
                    failed_time_s=exc.step*DT, reason=str(exc))


def meets_goal(values):
    t1, t2 = values["settling_time_s"]
    return bool(t1 is not None and t2 is not None and t1 <= 4. and t2 <= 10.
                and values["overshoot_percent"][1] < 20.)


def objectives(values):
    t1, t2 = values["settling_time_s"]
    return np.array([np.inf if t1 is None else t1, np.inf if t2 is None else t2,
                     values["overshoot_percent"][1]])


def pareto_candidates(values):
    """J1整定・J2整定・J2OSを独立した目的として扱う（加重スコアなし）。"""
    return [key for key, value in values.items() if not any(
        np.all(objectives(other) <= objectives(value)) and np.any(objectives(other) < objectives(value))
        for other_key, other in values.items() if other_key != key)]


def select_candidates(focus_results):
    """達成候補の両端と全体の速応端を選択。重複時は非劣解から補充し4候補。"""
    values = {v: metrics(r["data"]) for v, r in focus_results.items() if r["status"] == "ok"}
    if len(values) < 3:
        raise ValueError("Fewer than three numerically successful focus candidates")
    feasible = [v for v in values if meets_goal(values[v])]
    frontier = pareto_candidates(values)
    selected, reasons = [], {}

    def add(variant, reason):
        if variant not in selected and len(selected) < 4:
            selected.append(variant)
            reasons[variant] = reason

    for pool, index, reason in (
        (feasible, 0, "Fastest J1 among goal-achieving candidates"),
        (feasible, 2, "Lowest J2 overshoot among goal-achieving candidates"),
        (frontier, 0, "Fastest J1 on the full three-objective Pareto frontier"),
        (frontier, 1, "Fastest J2 on the full three-objective Pareto frontier"),
    ):
        if pool:
            add(min(pool, key=lambda v: (objectives(values[v])[index], tuple(VECTORS[v]))), reason)
    for pool in (frontier, feasible, list(values)):
        for v in sorted(pool, key=lambda v: tuple(objectives(values[v]))):
            add(v, "Additional tradeoff candidate; lexicographic tie/fill rule, not an optimality score")
    return dict(selected=selected, reasons=reasons, goal_achieving=feasible, pareto_frontier=frontier)


def trajectory_comparison(data, expected):
    names = ("time", "state", "command", "actual", "error", "integral", "stopped", "delta_tau", "correction")
    return dict(
        bitwise_equal=all(np.array_equal(a, b) for a, b in zip(data, expected)),
        within_tolerance=all(np.allclose(a, b, atol=MATCH_ATOL, rtol=MATCH_RTOL) for a, b in zip(data, expected)),
        max_abs_difference={key: float(np.max(np.abs(a.astype(float)-b.astype(float))))
                            for key, a, b in zip(names, data, expected)},
        atol=MATCH_ATOL, rtol=MATCH_RTOL,
    )


def run_experiment(progress=False):
    results = {}
    def run(case):
        if case not in results:
            results[case] = run_case(case)
            if progress:
                print(case, results[case]["status"], flush=True)
    for variant in VECTORS:
        run((variant, *FOCUS))
    selection = select_candidates({v: results[v, *FOCUS] for v in VECTORS})
    if progress:
        print("Selected:", selection, flush=True)
    for variant in (*selection["selected"], *REFERENCES):
        for gain, model in GROUPS:
            for limit in LIMITS:
                run((variant, gain, model, limit))
    diagonal = {}
    for kaw in KAW_VALUES:
        data = results[f"per_joint_{kaw:g}_{kaw:g}", *FOCUS]
        reference_case = f"back_calculation_{kaw:g}", *FOCUS
        expected = results.get(reference_case)
        if expected is None:
            expected = previous.run_case(reference_case)
        if data["status"] == expected["status"] == "ok":
            diagonal[str(kaw)] = trajectory_comparison(data["data"], expected["data"])
        else:
            diagonal[str(kaw)] = dict(within_tolerance=False, reason="A trajectory failed")
    return results, selection, diagonal


def case_name(case):
    return "_".join(case)


def save_results(results, selection, diagonal, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    equilibrium = {model: target_equilibrium(model) for model in MODELS}
    summary = dict(
        Kaw_candidates=list(KAW_VALUES), gains=GAINS, dt_s=DT, duration_s=DURATION,
        target_rad=TARGET.tolist(), initial_state=[0.]*4, initial_integral_rad_s=[0., 0.],
        integrator="RK2 midpoint", b1=0., b2=0., g=9.81, integral_limit_rad_s=INTEGRAL_LIMIT,
        torque_limits_Nm={k: None if v is None else v.tolist() for k, v in LIMITS.items()},
        gain_units=dict(Kp="s^-2", Ki="s^-3", Kd="s^-1", Kaw="s^-1"),
        update="delta_tau=actual-command; delta_v=solve(M_hat,delta_tau); correction=Kaw_vector*(delta_v/Ki), division FIRST; z_next=clip(z+dt*(error+correction),-0.5,0.5). No torque recomputation at RK2 midpoint. tau_k uses z_k, next integral used at k+1.",
        model="Nominal controller; mismatch changes only plant m2 and I2 by 1.2.",
        goal="J1 settling<=4 s AND J2 settling<=10 s AND J2 overshoot<20%",
        settling="Per joint, abs(error)<=2%*max(abs(initial_error),0.01 rad) for all remaining samples. null if unsettled at 20 s.",
        overshoot="100*max(0,max((q-target)*sign(target)))/abs(target), initial q=0",
        saturation="abs(command-actual)>1e-12+1e-12*abs(command); fractions per 8000 control steps. BC uses raw torque difference without deadband.",
        sampling="Final torque/delta/correction row diagnostic only; excluded from applied-step maxima, fractions and squared-time integrals. Correction units rad, integral units rad*s. Squared torque integrals are not electrical energy.",
        integral_limits="Boundary entries use tolerance 1e-12 rad*s; clipping counts corrected raw updates, excluding frozen steps.",
        diagonal_note="Step 8-17 multiplies before division; specified Step 8-18 divides before multiplication. Compare all samples with explicit tolerances and separately report bit equality; do not claim bit equality when false.",
        selection=selection, diagonal_focus_comparison=diagonal,
        selection_note="Goal-attaining extremes plus Pareto speed extremes, up to four distinct candidates. No weighted score or universal optimality claim.",
        failures="Numerical exceptions/nonfinite values or abs(state)>1e6 are recorded with time/reason and finite diagnostic prefix. Not a stability theorem.",
        interpretation="Strict limits cannot supply target holding torque; anti-windup cannot repair actuator insufficiency. Twenty-second errors are not proven asymptotic steady-state errors; no hardware guarantee.",
        target_equilibrium=equilibrium, cases={}, comparisons={}, goal_counts={},
    )
    rows = []
    for case, record in results.items():
        variant, gain, model, limit = case
        cp, pp = model_parameters(model)
        kaw = VECTORS[variant].tolist() if variant in VECTORS else (
            [previous.BACK_CALCULATION[variant]]*2 if variant in previous.BACK_CALCULATION else None)
        entry = dict(variant=variant, Kaw=kaw, gain=gain, model=model, limit=limit,
                     controller_params=cp, plant_params=pp, static_hold_possible=equilibrium[model]["limits"][limit]["static_hold_possible"],
                     **{k: v for k, v in record.items() if k != "data"})
        if entry["status"] == "ok":
            entry.update(metrics(record["data"]))
            entry["goal_met"] = meets_goal(entry)
        else:
            entry["goal_met"] = False
        summary["cases"][case_name(case)] = entry
        np.savetxt(output/(case_name(case)+".csv"), np.column_stack(record["data"]),
                   delimiter=",", comments="", header=CSV_HEADER)
        for j in (0, 1):
            row = {k: entry[k] for k in ("variant", "gain", "model", "limit", "status", "goal_met", "static_hold_possible")}
            row.update(joint=j+1, Kaw1=None if kaw is None else kaw[0], Kaw2=None if kaw is None else kaw[1])
            if entry["status"] == "ok":
                row.update({k: v[j] for k, v in entry.items() if isinstance(v, list) and k != "Kaw"})
                for section in ("saturation", "integral_limits"):
                    row.update({section+"_"+k: v[j] for k, v in entry[section].items() if isinstance(v, list)})
            else:
                row.update({k: entry[k] for k in ("reason", "failed_step", "failed_time_s")})
            rows.append(row)
    write_rows(output/"metrics.csv", rows)
    write_rows(output/"focus_comparison.csv", [r for r in rows if (r["gain"], r["model"], r["limit"]) == FOCUS])
    comparison_rows = []
    for variant in selection["selected"]:
        for gain, model in GROUPS:
            for limit in LIMITS:
                a = summary["cases"][case_name((variant, gain, model, limit))]
                for ref in REFERENCES:
                    b = summary["cases"][case_name((ref, gain, model, limit))]
                    compared = compare_methods(b, a) if a["status"] == b["status"] == "ok" else dict(status="failed")
                    key = f"{variant}_vs_{ref}_{gain}_{model}_{limit}"
                    summary["comparisons"][key] = compared
                    for j in (0, 1):
                        row = dict(candidate=variant, reference=ref, gain=gain, model=model, limit=limit, joint=j+1,
                                   status=compared.get("status", "ok"))
                        for section in ("reference", "candidate", "candidate_increased"):
                            if section in compared:
                                row.update({section+"_"+k: v[j] for k, v in compared[section].items()})
                        comparison_rows.append(row)
    write_rows(output/"comparison.csv", comparison_rows)
    for variant in (*selection["selected"], *REFERENCES):
        entries = [v for v in summary["cases"].values() if v["variant"] == variant]
        summary["goal_counts"][variant] = dict(total=len(entries), met=sum(v["goal_met"] for v in entries),
                                              failed=sum(v["status"] == "failed" for v in entries))
    (output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    save_plots(results, selection, output)
    return summary


def label(variant):
    return f"Kaw={VECTORS[variant].tolist()}" if variant in VECTORS else previous.LABELS[variant]


def draw_axis(ax, results, variants, group, index, joint):
    for n, variant in enumerate(variants):
        record = results[variant, *group]
        data = record["data"]
        name = label(variant)+(" FAILED" if record["status"] == "failed" else "")
        style = "--" if variant in REFERENCES else "-"
        if index in (2, 3, 8) and len(data[0]) > 1:
            values = data[index][:, joint].copy()
            if record["status"] == "ok":
                values[-1] = values[-2]
            ax.step(data[0], values, where="post", label=name, linestyle=style, color=plt.get_cmap("tab10")(n))
        else:
            ax.plot(data[0], data[index][:, joint], label=name, linestyle=style, color=plt.get_cmap("tab10")(n))
    if index in (1, 4):
        ax.axhline(TARGET[joint] if index == 1 else 0., color="black", linestyle=":")
        if index == 4:
            ax.axhspan(-.02*abs(TARGET[joint]), .02*abs(TARGET[joint]), color="gray", alpha=.15)
    if index == 5:
        for v in (-.5, .5):
            ax.axhline(v, color="gray", linestyle=":")
    if index in (2, 3) and LIMITS[group[2]] is not None:
        for v in (-LIMITS[group[2]][joint], LIMITS[group[2]][joint]):
            ax.axhline(v, color="gray", linestyle=":")
    ax.set_xlabel("Time [s]")
    ax.grid(True, alpha=.3)


def save_plots(results, selection, output):
    variants = (*REFERENCES, *selection["selected"])
    plots = (("angles", 1, "Angle [rad]"), ("errors", 4, "Error [rad]"),
             ("integrals", 5, "Integral [rad s]"), ("command_torque", 2, "Command [N m]"),
             ("actual_torque", 3, "Actual [N m]"), ("correction", 8, "Correction [rad]"))
    for limit in LIMITS:
        for name, index, ylabel in plots:
            fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharex=True, layout="constrained")
            for c, (gain, model) in enumerate(GROUPS):
                for j in (0, 1):
                    draw_axis(axes[j, c], results, variants, (gain, model, limit), index, j)
                    axes[j, c].set(title=f"{gain}, {model}, J{j+1}", ylabel=ylabel)
            fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=3, fontsize=8)
            fig.suptitle(f"{limit}: selected candidates and five references")
            fig.savefig(output/f"{name}_{limit}.png", dpi=130)
            plt.close(fig)
    for end in (2, 20):
        fig, axes = plt.subplots(6, 2, figsize=(14, 18), sharex=True, layout="constrained")
        for r, (_, index, ylabel) in enumerate(plots):
            for j in (0, 1):
                draw_axis(axes[r, j], results, variants, FOCUS, index, j)
                axes[r, j].set(title=f"Joint {j+1}", ylabel=ylabel, xlim=(0., end))
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=3, fontsize=8)
        fig.suptitle(f"B, mismatch +20%, moderate: first {end} seconds")
        fig.savefig(output/f"focus_{end}s.png", dpi=130)
        plt.close(fig)
    for filename, keys in (
        ("grid_response", (("settling_time_s", "Settling [s]"), ("overshoot_percent", "Overshoot [%]"),
                           ("final_abs_angle_error_rad", "Abs error at 20 s [rad]"))),
        ("grid_saturation_integral", (("saturation_fraction", "Saturation fraction"),
                                      ("max_abs_integral_rad_s", "Max abs integral [rad s]"))),
    ):
        fig, axes = plt.subplots(2, len(keys), figsize=(6*len(keys), 10), layout="constrained")
        for c, (key, title) in enumerate(keys):
            for j in (0, 1):
                array = np.full((5, 5), np.nan)
                texts = {}
                for r, a in enumerate(KAW_VALUES):
                    for col, b in enumerate(KAW_VALUES):
                        record = results[f"per_joint_{a:g}_{b:g}", *FOCUS]
                        value = None
                        if record["status"] == "ok":
                            v = metrics(record["data"])
                            value = v["saturation"]["per_joint_fraction"][j] if key == "saturation_fraction" else v[key][j]
                            texts[r, col] = "unsettled" if value is None else f"{value:.3g}"+(" *" if meets_goal(v) else "")
                        else:
                            texts[r, col] = "FAILED"
                        if value is not None:
                            array[r, col] = value
                ax = axes[j, c]
                im = ax.imshow(np.ma.masked_invalid(array), cmap=plt.get_cmap("viridis").with_extremes(bad="lightgray"))
                for (r, col), text in texts.items():
                    ax.text(col, r, text, ha="center", va="center", fontsize=9,
                            bbox=dict(facecolor="white", alpha=.8, edgecolor="none", pad=1))
                ax.set(xticks=range(5), xticklabels=KAW_VALUES, yticks=range(5), yticklabels=KAW_VALUES,
                       xlabel="Kaw2 [s^-1]", ylabel="Kaw1 [s^-1]", title=f"J{j+1}: {title}")
                fig.colorbar(im, ax=ax, shrink=.8)
        fig.suptitle("Focus grid: * = all three performance targets met")
        fig.savefig(output/f"{filename}.png", dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"outputs"/"step_8_18")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    results, selection, diagonal = run_experiment(progress=True)
    save_results(results, selection, diagonal, args.output)
    print("Goal-achieving focus candidates:", selection["goal_achieving"])
    print("Output:", args.output.resolve())


if __name__ == "__main__":
    main()
