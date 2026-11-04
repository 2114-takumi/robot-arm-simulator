"""Step 8-13: 固定したCTC+IゲインA/Bを42条件で比較する。"""

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
from robot_arm.computed_torque_integral import computed_torque_integral
from robot_arm.dynamics import integrate_dynamics
from scripts.run_coupled_dynamics_demo import physical_parameters
from scripts.run_ctc_integral_demo import update_integral, INTEGRAL_LIMIT
from scripts.run_ki_sweep_demo import overshoot_percent
from scripts.run_pd_gravity_demo import settling_times
from scripts.run_kd_sweep_demo import meets_joint2_goal

GAINS = {"A": dict(Kp=25., Ki=7., Kd=10.), "B": dict(Kp=40., Ki=10., Kd=10.)}
ERROR_PERCENTAGES = (-30, -20, -10, 0, 10, 20, 30)
TARGETS = ((.3, -.4), (.6, -.8), (-.3, .5))
DT, DURATION = .0025, 20.
CASES = tuple((gain, percent, target) for gain in GAINS
              for percent in ERROR_PERCENTAGES for target in TARGETS)


def model_parameters(error_percent):
    """m2・I2のみ同率で変更。制御器とプラントは独立した辞書。"""
    if not np.isscalar(error_percent) or not np.isfinite(error_percent) or error_percent <= -100:
        raise ValueError("Model error must be finite and greater than -100 percent")
    controller_params = physical_parameters()
    plant_params = physical_parameters()
    for key in ("m2", "I2"):
        plant_params[key] *= 1. + error_percent / 100.
    return controller_params, plant_params


def simulate(gain, error_percent, target):
    """tau_kにはz_kを使用。RK2の途中では制御入力を更新しない。"""
    if gain not in GAINS:
        raise ValueError("Unknown gain setting")
    target = np.array(target, dtype=float, copy=True)
    if target.shape != (2,) or not np.all(np.isfinite(target)) or np.any(target == 0):
        raise ValueError("Target must contain two finite nonzero steps")
    cp, pp = model_parameters(error_percent)
    kp, kd, ki = (np.full(2, GAINS[gain][key]) for key in ("Kp", "Kd", "Ki"))
    time = np.arange(round(DURATION / DT) + 1) * DT
    states = np.zeros((len(time), 4))
    torques = np.zeros((len(time), 2))
    integrals = np.zeros_like(torques)
    for i in range(len(time)):
        q, omega = states[i, :2], states[i, 2:]
        torques[i] = computed_torque_integral(q, omega, target, integrals[i], kp, kd, ki, cp)
        if i < len(time) - 1:
            states[i + 1] = integrate_dynamics(
                *states[i], *torques[i], DT, **pp, b1=0., b2=0., g=9.81, method="rk2")
            integrals[i + 1] = update_integral(integrals[i], target - q, DT)
    errors = target - states[:, :2]
    data = time, states, torques, errors, integrals
    if not np.all(np.isfinite(np.column_stack(data))):
        raise ValueError("Nonfinite simulation data")
    return data


def metrics(data, target):
    """トルク二乗積分は実際に保持した8000区間を積算。終端診断値は除外。"""
    time, states, torques, errors, integrals = data
    settled, thresholds = settling_times(time, errors)
    return dict(
        final_angle_error_rad=errors[-1].tolist(),
        final_abs_angle_error_rad=np.abs(errors[-1]).tolist(),
        settling_time_s=settled, settling_threshold_rad=thresholds.tolist(),
        overshoot_percent=overshoot_percent(states[:, :2], target).tolist(),
        max_abs_torque_Nm=np.max(np.abs(torques[:-1]), axis=0).tolist(),
        max_abs_integral_rad_s=np.max(np.abs(integrals), axis=0).tolist(),
        torque_squared_integral_Nm2_s=np.sum(torques[:-1] ** 2 * np.diff(time)[:, None], axis=0).tolist(),
    )


def failure_reasons(values):
    reasons = []
    if values["overshoot_percent"][1] >= 20.:
        reasons.append("Joint 2 overshoot >=20%")
    settled = values["settling_time_s"][1]
    if settled is None:
        reasons.append("Joint 2 unsettled within 20 s")
    elif settled > 10.:
        reasons.append("Joint 2 settling time >10 s")
    return reasons


def case_name(gain, percent, target):
    return f"{gain}_error_{percent:+g}_target_{TARGETS.index(tuple(target)) + 1}"


def save_results(results, output):
    """新規ディレクトリだけに指標・比較表・ヒートマップを保存する。"""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    summary = dict(
        gains=GAINS, gain_units=dict(Kp="1/s^2", Ki="1/s^3", Kd="1/s"),
        error_percentages=list(ERROR_PERCENTAGES), targets_rad=[list(t) for t in TARGETS],
        dt_s=DT, duration_s=DURATION, initial_state=[0., 0., 0., 0.],
        initial_integral_rad_s=[0., 0.], target_velocity_rad_s=[0., 0.],
        b1=0., b2=0., g=9.81, method="rk2 midpoint", integral_limit_rad_s=INTEGRAL_LIMIT,
        model_error="Only plant m2 and I2 multiplied by 1+error_percent/100; controller stays nominal",
        integral_update="tau_k uses z_k; z_(k+1)=clip(z_k+dt*(target-q_k),-0.5,0.5); torque held within RK2",
        settling_definition="Per joint: abs(error)<=0.02*max(abs(initial_error),0.01 rad) for all remaining samples through 20 s; null if unsettled",
        overshoot_definition="100*max(0,max((q-target)*sign(target)))/abs(target); initial q=0",
        torque_squared_integral_definition="sum(tau_k^2*(t_(k+1)-t_k)), per joint, k=0..7999; zero-order hold integral; final diagnostic torque excluded. Unit (N m)^2 s. Control effort, NOT electrical energy consumption.",
        performance_goal="Joint 2 overshoot <20% AND settling time <=10 s; tested before rounding",
        interpretation="Fixed gain comparison, no optimization. Finite 20-second errors do not prove asymptotic steady-state errors. Simulated performance does not establish untested physical robot performance.",
        cases={}, gain_summary={},
    )
    rows = []
    for case, data in results.items():
        gain, percent, target = case
        values = metrics(data, target)
        cp, pp = model_parameters(percent)
        name = case_name(*case)
        summary["cases"][name] = dict(
            gain=gain, error_percent=percent, target_rad=list(target),
            controller_params=cp, plant_params=pp, samples=len(data[0]),
            joint2_goal_met=meets_joint2_goal(values), failure_reasons=failure_reasons(values), **values)
        for j in (0, 1):
            rows.append(dict(gain=gain, error_percent=percent, target1_rad=target[0],
                             target2_rad=target[1], joint=j+1,
                             joint2_goal_met=meets_joint2_goal(values), **{k: v[j] for k, v in values.items()}))
    for gain in GAINS:
        entries = {k: v for k, v in summary["cases"].items() if v["gain"] == gain}
        summary["gain_summary"][gain] = dict(
            total=len(entries), goal_met=sum(v["joint2_goal_met"] for v in entries.values()),
            failures={k: v["failure_reasons"] for k, v in entries.items() if not v["joint2_goal_met"]})
    write_csv(output / "metrics.csv", rows)
    comparison = []
    for percent in ERROR_PERCENTAGES:
        for target in TARGETS:
            a, b = (summary["cases"][case_name(g, percent, target)] for g in GAINS)
            for j in (0, 1):
                row = dict(error_percent=percent, target1_rad=target[0], target2_rad=target[1], joint=j+1)
                for gain, entry in (("A", a), ("B", b)):
                    row[f"{gain}_joint2_goal_met"] = entry["joint2_goal_met"]
                    row.update({f"{gain}_{key}": entry[key][j] for key in metrics(results[gain, percent, target], target)})
                comparison.append(row)
    write_csv(output / "comparison.csv", comparison)
    save_plots(summary, output)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return summary


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def save_plots(summary, output):
    """列は目標姿勢、行はモデル誤差。A/B・両関節を共通色尺度で比較。"""
    for key, unit in (("overshoot_percent", "%"), ("settling_time_s", "s"),
                      ("final_abs_angle_error_rad", "rad"), ("max_abs_torque_Nm", "N m"),
                      ("max_abs_integral_rad_s", "rad s"), ("torque_squared_integral_Nm2_s", "(N m)^2 s")):
        matrices = {(gain, j): np.array([
            [np.nan if (v := summary["cases"][case_name(gain, percent, target)][key][j]) is None else v
             for target in TARGETS] for percent in ERROR_PERCENTAGES]) for gain in GAINS for j in (0, 1)}
        finite = np.concatenate([v[np.isfinite(v)] for v in matrices.values()])
        vmax = float(np.max(finite)) if len(finite) else 1.
        cmap = plt.get_cmap("viridis").with_extremes(bad="lightgray")
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), layout="constrained")
        for col, gain in enumerate(GAINS):
            for j in (0, 1):
                ax = axes[j, col]
                values = matrices[gain, j]
                im = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=0., vmax=vmax, aspect="auto")
                for r in range(7):
                    for c in range(3):
                        v = values[r, c]
                        label = "unsettled" if np.isnan(v) else (f"{v:.2e}" if unit == "rad" else f"{v:.2f}")
                        ax.text(c, r, label, ha="center", va="center", fontsize=9,
                                color="black" if np.isnan(v) or v > .55*vmax else "white")
                ax.set(xticks=range(3), xticklabels=[str(t) for t in TARGETS],
                       yticks=range(7), yticklabels=[f"{p:+d}%" for p in ERROR_PERCENTAGES],
                       xlabel="Target [rad]", ylabel="Plant m2 / I2 error", title=f"Gain {gain}, Joint {j+1}")
        fig.suptitle(f"{key} [{unit}]")
        fig.colorbar(im, ax=axes.ravel().tolist(), label=unit, shrink=.85)
        fig.savefig(output / (key + ".png"), dpi=150)
        plt.close(fig)

    table_rows = []
    for target in TARGETS:
        for percent in ERROR_PERCENTAGES:
            row = [str(target), f"{percent:+d}%"]
            for gain in GAINS:
                v = summary["cases"][case_name(gain, percent, target)]
                settled = v["settling_time_s"][1]
                row += [f'{v["overshoot_percent"][1]:.2f}', "unsettled" if settled is None else f"{settled:.3f}",
                        "PASS" if v["joint2_goal_met"] else "FAIL"]
            table_rows.append(row)
    fig, ax = plt.subplots(figsize=(13, 11))
    ax.axis("off")
    table = ax.table(cellText=table_rows,
                     colLabels=["Target [rad]", "Error", "A OS [%]", "A Ts [s]", "A goal", "B OS [%]", "B Ts [s]", "B goal"],
                     cellLoc="center", loc="center", colWidths=[.2, .08, .12, .12, .1, .12, .12, .1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1., 1.8)
    for r, row in enumerate(table_rows, 1):
        for c in (4, 7):
            table[r, c].set_facecolor("#d4edda" if row[c] == "PASS" else "#f8d7da")
    ax.set_title("Joint 2: overshoot <20% and settling <=10 s (evaluated before rounding)")
    fig.tight_layout()
    fig.savefig(output / "comparison_table.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "step_8_13")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    results = {}
    for case in CASES:
        results[case] = simulate(*case)
        print(f"Completed {case}", flush=True)
    summary = save_results(results, args.output)
    print(json.dumps(summary["gain_summary"], indent=2))
    print("Output:", args.output.resolve())


if __name__ == "__main__":
    main()
