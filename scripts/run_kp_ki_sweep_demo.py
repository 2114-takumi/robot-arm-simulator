"""Step 8-12: Kd=10でKp・Kiを掃引し、多指標で候補を比較する。"""

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
from scripts.run_ctc_integral_demo import update_integral, INTEGRAL_LIMIT
from scripts.run_computed_torque_demo import DT, TARGET
from scripts.run_model_mismatch_demo import model_parameters
from scripts.run_kd_sweep_demo import metrics, meets_joint2_goal

KP_VALUES = (15., 20., 25., 30., 40.)
KI_VALUES = (3., 5., 7., 10.)
KD = np.array([10., 10.])
MODELS = ("baseline", "mismatch")
DURATION = 20.
CASES = tuple((m, kp, ki) for m in MODELS for kp in KP_VALUES for ki in KI_VALUES)
OBJECTIVES = ("final_abs_angle_error_rad", "settling_time_s", "overshoot_percent", "max_abs_torque_Nm")


def case_name(model, kp, ki):
    return f"{model}_Kp_{kp:g}_Ki_{ki:g}"


def simulate(model, kp, ki):
    """tau_kはz_kを使用。z_(k+1)は左端誤差から更新し次ステップで使用。"""
    for value in (kp, ki):
        if not np.isscalar(value) or not np.isfinite(value) or value <= 0:
            raise ValueError("Kp and Ki must be positive finite scalars")
    cp, pp = model_parameters(model)
    kp_vector, ki_vector = np.full(2, kp), np.full(2, ki)
    time = np.arange(round(DURATION / DT) + 1) * DT
    states = np.zeros((len(time), 4))
    torques = np.zeros((len(time), 2))
    integrals = np.zeros_like(torques)
    for i in range(len(time)):
        q, omega = states[i, :2], states[i, 2:]
        torques[i] = computed_torque_integral(q, omega, TARGET, integrals[i], kp_vector, KD, ki_vector, cp)
        if i < len(time) - 1:
            states[i + 1] = integrate_dynamics(*states[i], *torques[i], DT, **pp, b1=0., b2=0., g=9.81, method="rk2")
            integrals[i + 1] = update_integral(integrals[i], TARGET - q, DT)
    errors = TARGET - states[:, :2]
    if not np.all(np.isfinite(np.column_stack((time, states, torques, errors, integrals)))):
        raise ValueError("Nonfinite simulation data")
    return time, states, torques, errors, integrals


def pareto_indices(vectors):
    """全軸を最小化。他候補が全軸以下かつ1軸で厳密に小さい場合のみ支配。"""
    values = np.asarray(vectors, dtype=float)
    return [i for i, v in enumerate(values) if not any(
        np.all(other <= v) and np.any(other < v) for other in values
    )]


def analyze_candidates(cases):
    """両モデル・両関節の最悪値4軸でPareto評価し、参照設定との差も保存。"""
    candidates = []
    for kp in KP_VALUES:
        for ki in KI_VALUES:
            values = [cases[case_name(m, kp, ki)] for m in MODELS]
            worst = {}
            for key in OBJECTIVES:
                components = [v for entry in values for v in entry[key]]
                worst[key] = None if any(v is None for v in components) else max(components)
            regressions = {}
            for model, current in zip(MODELS, values):
                reference = cases[case_name(model, 25., 7.)]
                regressions[model] = {}
                for key in OBJECTIVES:
                    # 未整定は有限の整定時間より悪いと扱う。
                    a = [np.inf if v is None else v for v in current[key]]
                    b = [np.inf if v is None else v for v in reference[key]]
                    regressions[model][key] = [bool(x > y) for x, y in zip(a, b)]
            candidates.append(dict(Kp=kp, Ki=ki, mismatch_joint2_goal_met=meets_joint2_goal(values[1]),
                                   worst_case_objectives=worst, worse_than_Kp25_Ki7=regressions))
    vectors = [[np.inf if c["worst_case_objectives"][key] is None else c["worst_case_objectives"][key]
                for key in OBJECTIVES] for c in candidates]
    frontier = set(pareto_indices(vectors))
    feasible = [i for i, c in enumerate(candidates) if c["mismatch_joint2_goal_met"]]
    feasible_frontier = {feasible[i] for i in pareto_indices([vectors[i] for i in feasible])} if feasible else set()
    for i, candidate in enumerate(candidates):
        candidate["pareto_nondominated_all"] = i in frontier
        candidate["pareto_nondominated_goal_met"] = i in feasible_frontier
    return candidates


def save_results(results, output):
    """40条件の指標CSV・JSON・ヒートマップを新規保存先に出力する。"""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    summary = dict(
        Kp_values=list(KP_VALUES), Ki_values=list(KI_VALUES), Kd=KD.tolist(),
        gain_units=dict(Kp="1/s^2", Ki="1/s^3", Kd="1/s"),
        dt_s=DT, duration_s=DURATION, method="rk2 midpoint", b1=0., b2=0., g=9.81,
        initial_state=[0.,0.,0.,0.], initial_integral_rad_s=[0.,0.], target_rad=TARGET.tolist(),
        target_velocity_rad_s=[0.,0.], integral_limit_rad_s=INTEGRAL_LIMIT,
        integral_update="tau_k uses z_k; z_(k+1)=clip(z_k+dt*(target-q_k),-0.5,0.5); torque held within RK2",
        torque_maximum_definition="Applied control steps only; exclude final diagnostic torque",
        settling_definition="Per joint: abs(error)<=0.02*max(abs(initial_error),0.01 rad) for all remaining samples through 20 s; null if unsettled",
        overshoot_definition="100*max(0,max((q-target)*sign(target)))/abs(target), initial q=0",
        tail_definition="18<=t<=20 s (801 samples); mean absolute error, signed e(20)-e(18), and peak-to-peak",
        performance_goal="Mismatch Joint 2 overshoot <20% AND settling time <=10 s, before rounding",
        pareto_definition="Minimize four separate worst-case objectives across BOTH models and BOTH joints: final absolute error, settling time, overshoot, maximum absolute torque. Unsettled time treated as infinity (stored as null). No weighted score. Report all-candidate and goal-feasible frontiers separately.",
        interpretation="Finite 20-second results do not prove asymptotic steady-state error; Pareto membership depends on these stated objectives and sampled gains.",
        cases={},
    )
    rows = []
    for (model, kp, ki), data in results.items():
        values = metrics(data)
        cp, pp = model_parameters(model)
        summary["cases"][case_name(model,kp,ki)] = dict(model=model, Kp=[kp,kp], Ki=[ki,ki],
            controller_params=cp, plant_params=pp, samples=len(data[0]), **values)
        for j in (0,1):
            rows.append(dict(model=model, Kp=kp, Ki=ki, Kd=10., joint=j+1, **{k:v[j] for k,v in values.items()}))
    summary["candidates"] = analyze_candidates(summary["cases"])
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    save_heatmaps(summary, output)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return summary


def save_heatmaps(summary, output):
    """両モデル・両関節に共通の色尺度。未整定を灰色と文字で区別。"""
    for filename, key, unit in (("overshoot.png", "overshoot_percent", "%"),
                               ("settling_times.png", "settling_time_s", "s"),
                               ("final_errors.png", "final_abs_angle_error_rad", "rad")):
        matrices = {}
        for model in MODELS:
            for j in (0,1):
                matrices[model,j] = np.array([[np.nan if summary["cases"][case_name(model,kp,ki)][key][j] is None
                    else summary["cases"][case_name(model,kp,ki)][key][j] for ki in KI_VALUES] for kp in KP_VALUES])
        vmax = max(float(np.nanmax(v)) for v in matrices.values())
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("lightgray")
        fig, axes = plt.subplots(2,2,figsize=(13,9),layout="constrained")
        for col, model in enumerate(MODELS):
            for j in (0,1):
                ax = axes[j,col]
                values = matrices[model,j]
                im = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=0., vmax=vmax, aspect="auto")
                for r in range(len(KP_VALUES)):
                    for c in range(len(KI_VALUES)):
                        v = values[r,c]
                        label = "unsettled" if np.isnan(v) else (f"{v:.2e}" if unit=="rad" else f"{v:.2f}")
                        ax.text(c,r,label,ha="center",va="center",fontsize=8,
                                color="black" if np.isnan(v) or v > .55*vmax else "white")
                ax.set(xticks=range(len(KI_VALUES)),xticklabels=[f"{v:g}" for v in KI_VALUES],
                       yticks=range(len(KP_VALUES)),yticklabels=[f"{v:g}" for v in KP_VALUES],
                       xlabel="Ki [1/s^3]",ylabel="Kp [1/s^2]",title=f"{model}, Joint {j+1}")
        fig.suptitle(f"{key} [{unit}] — Kd=10")
        fig.colorbar(im,ax=axes.ravel().tolist(),label=unit,shrink=.85)
        fig.savefig(output / filename,dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"outputs"/"step_8_12")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    results = {}
    for case in CASES:
        results[case] = simulate(*case)
        print(f"Completed {case}",flush=True)
    summary = save_results(results,args.output)
    for candidate in summary["candidates"]:
        print(json.dumps(candidate))
    print("Output:",args.output.resolve())


if __name__ == "__main__":
    main()
