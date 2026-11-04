"""Step 8-11: Kp=25、Ki=7でKdを掃引し、過渡応答を比較。"""

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
from scripts.run_ctc_integral_demo import update_integral, CSV_HEADER, INTEGRAL_LIMIT
from scripts.run_computed_torque_demo import CTC_KP, DT, TARGET
from scripts.run_ki_sweep_demo import metrics as ki_metrics
from scripts.run_model_mismatch_demo import model_parameters
from scripts.run_pd_gravity_demo import settling_times

KD_VALUES = (6., 8., 10., 12., 14., 16., 20.)
KI = np.array([7., 7.])
MODELS = ("baseline", "mismatch")
DURATION = 20.
CASES = tuple((model, kd) for model in MODELS for kd in KD_VALUES)


def simulate(model, kd):
    """Step 8-9と同じ左端誤差・積分更新順序で20秒実行する。"""
    if not np.isscalar(kd) or not np.isfinite(kd) or kd <= 0:
        raise ValueError("Kd must be a finite positive scalar")
    controller_params, plant_params = model_parameters(model)
    gains = np.full(2, kd)
    time = np.arange(round(DURATION / DT) + 1) * DT
    states = np.zeros((len(time), 4))
    torques = np.zeros((len(time), 2))
    integrals = np.zeros_like(torques)
    for i in range(len(time)):
        q, omega = states[i, :2], states[i, 2:]
        torques[i] = computed_torque_integral(q, omega, TARGET, integrals[i], CTC_KP, gains, KI, controller_params)
        if i < len(time) - 1:
            states[i + 1] = integrate_dynamics(*states[i], *torques[i], DT, **plant_params, b1=0., b2=0., g=9.81, method="rk2")
            integrals[i + 1] = update_integral(integrals[i], TARGET - q, DT)
    errors = TARGET - states[:, :2]
    if not np.all(np.isfinite(np.column_stack((time, states, torques, errors, integrals)))):
        raise ValueError("Nonfinite simulation data")
    return time, states, torques, errors, integrals


def metrics(data):
    values = ki_metrics(data)
    values["final_abs_angle_error_rad"] = np.abs(data[3][-1]).tolist()
    return values


def meets_joint2_goal(values):
    """Joint 2の超過20%未満、整定10秒以内を丸め前の値で判定。"""
    settled = values["settling_time_s"][1]
    return bool(values["overshoot_percent"][1] < 20. and settled is not None and settled <= 10.)


def save_results(results, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    summary = dict(
        Kd_values=list(KD_VALUES), Kp=CTC_KP.tolist(), Ki=KI.tolist(),
        gain_units=dict(Kp="1/s^2", Kd="1/s", Ki="1/s^3"),
        dt_s=DT, duration_s=DURATION, target_rad=TARGET.tolist(), initial_state=[0., 0., 0., 0.],
        initial_integral_rad_s=[0., 0.], target_velocity_rad_s=[0., 0.],
        method="rk2 midpoint", b1=0., b2=0., g=9.81, integral_limit_rad_s=INTEGRAL_LIMIT,
        integral_update="tau_k uses z_k; z_(k+1)=clip(z_k+dt*(target-q_k),-0.5,0.5)",
        torque_sampling="Step-start torque held within RK2; final diagnostic torque excluded from maxima",
        settling_definition="Per joint: abs(error)<=0.02*max(abs(initial_error),0.01 rad) for all remaining samples through 20 s; null if unsettled",
        overshoot_definition="100*max(0,max((q-target)*sign(target)))/abs(target); initial q=0",
        tail_definition="18<=t<=20 s, 801 samples; mean absolute error is sample mean; signed change=e(20)-e(18); peak-to-peak also reported",
        interpretation="20-second values do not prove asymptotic steady-state error. Compare multiple metrics; no arbitrary scalar optimum score.",
        performance_goal="Mismatch Joint 2: overshoot <20% and settling <=10 s, evaluated before rounding",
        cases={},
    )
    rows = []
    for (model, kd), data in results.items():
        name = f"{model}_Kd_{kd:g}"
        np.savetxt(output / (name + ".csv"), np.column_stack(data), delimiter=",", comments="", header=CSV_HEADER)
        cp, pp = model_parameters(model)
        values = metrics(data)
        summary["cases"][name] = dict(model=model, Kd=[kd, kd], controller_params=cp, plant_params=pp, samples=len(data[0]), **values)
        for j in (0, 1):
            rows.append(dict(model=model, Kd=kd, joint=j+1, **{k: v[j] for k, v in values.items()}))
    with (output / "metrics.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary["mismatch_joint2_goal_met"] = {f"Kd_{kd:g}": meets_joint2_goal(summary["cases"][f"mismatch_Kd_{kd:g}"]) for kd in KD_VALUES}
    save_plots(results, summary, output)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return summary


def save_plots(results, summary, output):
    colors = plt.get_cmap("viridis")(np.linspace(0, 1, len(KD_VALUES)))
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    for col, model in enumerate(MODELS):
        for j in (0, 1):
            ax = axes[j, col]
            for kd, color in zip(KD_VALUES, colors):
                data = results[model, kd]
                ax.plot(data[0], data[1][:, j], label=f"Kd={kd:g}", color=color)
            ax.axhline(TARGET[j], linestyle="--", color="black", label="Target")
            ax.set(title=f"{model}, Joint {j+1}", xlabel="Time [s]", ylabel="Angle [rad]")
            ax.grid(True)
            ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "angles.png", dpi=150)
    plt.close(fig)
    for filename, key, label in (
        ("final_errors.png", "final_angle_error_rad", "Absolute error at 20 s [rad]"),
        ("settling_times.png", "settling_time_s", "2% settling time [s]"),
        ("overshoot.png", "overshoot_percent", "Overshoot [%]"),
    ):
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        for j, ax in enumerate(axes):
            for model, marker in zip(MODELS, ("o", "s")):
                values = [summary["cases"][f"{model}_Kd_{kd:g}"][key][j] for kd in KD_VALUES]
                y = [np.nan if v is None else abs(v) for v in values]
                line, = ax.plot(KD_VALUES, y, marker=marker, label=model)
                missing = [kd for kd, v in zip(KD_VALUES, values) if v is None]
                if missing:
                    ax.scatter(missing, [21.]*len(missing), marker="x", color=line.get_color(), label=f"{model}: unsettled by 20 s")
            if key == "settling_time_s":
                ax.set_ylim(0, 22)
                ax.axhline(20, linestyle=":", color="gray")
            if key == "final_angle_error_rad":
                ax.set_yscale("symlog", linthresh=1e-8)
            if j == 1 and key in ("settling_time_s", "overshoot_percent"):
                ax.axhline(10 if key == "settling_time_s" else 20, color="black", linestyle="--", label="Goal boundary")
            ax.set(title=f"Joint {j+1}", ylabel=label)
            ax.grid(True)
            ax.legend()
        axes[-1].set_xlabel("Kd [1/s] (same for both joints)")
        fig.tight_layout()
        fig.savefig(output / filename, dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "step_8_11")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    results = {case: simulate(*case) for case in CASES}
    summary = save_results(results, args.output)
    for name, values in summary["cases"].items():
        print(name, json.dumps({k: values[k] for k in ("final_angle_error_rad", "settling_time_s", "overshoot_percent", "max_abs_integral_rad_s", "last_2s_mean_abs_error_rad", "last_2s_error_change_rad")}))
    print("Mismatch Joint 2 goal:", summary["mismatch_joint2_goal_met"])
    print("Output:", args.output.resolve())


if __name__ == "__main__":
    main()
