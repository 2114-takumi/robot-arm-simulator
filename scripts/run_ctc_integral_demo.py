"""Step 8-9: モデル誤差に対するCTCとCTC+Iの8秒応答を比較する。"""

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
from robot_arm.computed_torque import computed_torque
from robot_arm.computed_torque_integral import computed_torque_integral
from robot_arm.dynamics import integrate_dynamics
from scripts.run_computed_torque_demo import CTC_KP, CTC_KD, TARGET, DT
from scripts.run_model_mismatch_demo import model_parameters
from scripts.run_pd_gravity_demo import settling_times

KI = np.array([10., 10.])
DURATION = 8.
INTEGRAL_LIMIT = 0.5
CONTROLLERS = ("CTC", "CTC_I")
MODELS = ("baseline", "mismatch")
CASES = tuple((c, m) for m in MODELS for c in CONTROLLERS)
CSV_HEADER = "time_s,theta1_rad,theta2_rad,omega1_rad_s,omega2_rad_s,tau1_Nm,tau2_Nm,error1_rad,error2_rad,integral1_rad_s,integral2_rad_s"


def update_integral(integral_error, error, dt):
    """左端誤差e_kでz_(k+1)=clip(z_k+dt*e_k, ±0.5)を計算する。"""
    integral_error, error = (np.asarray(v, dtype=float) for v in (integral_error, error))
    for value in (integral_error, error):
        if value.shape != (2,) or not np.all(np.isfinite(value)):
            raise ValueError("Integral and error must be finite two-component vectors")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive and finite")
    return np.clip(integral_error + dt * error, -INTEGRAL_LIMIT, INTEGRAL_LIMIT)


def simulate(controller, model):
    """各行はt_kの状態とz_k。tau_kはz_kを使い、z_(k+1)は次ステップから使う。"""
    if controller not in CONTROLLERS:
        raise ValueError("Unknown controller")
    controller_params, plant_params = model_parameters(model)
    time = np.arange(round(DURATION / DT) + 1) * DT
    states = np.zeros((len(time), 4))
    torques = np.zeros((len(time), 2))
    integrals = np.zeros_like(torques)
    for i in range(len(time)):
        q, omega = states[i, :2], states[i, 2:]
        if controller == "CTC":
            torques[i] = computed_torque(q, omega, TARGET, CTC_KP, CTC_KD, controller_params)
        else:
            torques[i] = computed_torque_integral(q, omega, TARGET, integrals[i], CTC_KP, CTC_KD, KI, controller_params)
        if i < len(time) - 1:
            states[i + 1] = integrate_dynamics(*states[i], *torques[i], DT, **plant_params, b1=0., b2=0., g=9.81, method="rk2")
            if controller == "CTC_I":
                integrals[i + 1] = update_integral(integrals[i], TARGET - q, DT)
    errors = TARGET - states[:, :2]
    if not np.all(np.isfinite(np.column_stack((time, states, torques, errors, integrals)))):
        raise ValueError("Nonfinite simulation data")
    return time, states, torques, errors, integrals


def response_metrics(data):
    """オーバーシュートと誤差の符号反転を含む有限時間の指標。"""
    time, states, torques, errors, integrals = data
    settled, thresholds = settling_times(time, errors)
    overshoot = np.maximum(0., np.max((states[:, :2] - TARGET) * np.sign(TARGET), axis=0))
    crossings = []
    for j in (0, 1):
        signs = np.sign(errors[np.abs(errors[:, j]) > 1e-6, j])
        crossings.append(int(np.count_nonzero(signs[1:] != signs[:-1])))
    return dict(
        angle_error_at_4s_rad=errors[round(4 / DT)].tolist(),
        angle_error_at_8s_rad=errors[-1].tolist(),
        settling_time_s=settled, settling_threshold_rad=thresholds.tolist(),
        max_abs_torque_Nm=np.max(np.abs(torques[:-1]), axis=0).tolist(),
        max_abs_integral_rad_s=np.max(np.abs(integrals), axis=0).tolist(),
        overshoot_rad=overshoot.tolist(), overshoot_percent=(100 * overshoot / np.abs(TARGET)).tolist(),
        error_zero_crossings=crossings,
        late_error_peak_to_peak_rad=np.ptp(errors[time >= 4.], axis=0).tolist(),
    )


def save_results(results, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    summary = dict(
        dt_s=DT, duration_s=DURATION, target_rad=TARGET.tolist(), initial_state=[0., 0., 0., 0.],
        target_velocity_rad_s=[0., 0.], initial_integral_rad_s=[0., 0.],
        Kp=CTC_KP.tolist(), Kd=CTC_KD.tolist(), Ki=KI.tolist(),
        gain_units=dict(Kp="1/s^2", Kd="1/s", Ki="1/s^3"),
        gain_note="Ki=[10,10] is an initial candidate, not optimized; CTC has no integral term.",
        method="rk2 midpoint", b1=0., b2=0., g=9.81, integral_limit_rad_s=INTEGRAL_LIMIT,
        integral_update="z_0=0; tau_k uses z_k; z_(k+1)=clip(z_k+dt*(target-q_k),-0.5,0.5). CTC integral history stays zero.",
        torque_sampling="tau_k held within RK2; final diagnostic row is not applied and excluded from torque maxima",
        settling_definition="Per joint: abs(error)<=0.02*max(abs(initial_error),0.01 rad) for all remaining samples through 8 s; null if unsettled",
        overshoot_definition="max(0,max((q-target)*sign(target-initial_q))) per joint",
        oscillation_definition="Error sign changes after removing samples with abs(error)<=1e-6 rad; also peak-to-peak error from 4 to 8 s (includes drift, not solely oscillation)",
        conclusion_note="8-second results do not prove zero asymptotic steady-state error.",
        cases={},
    )
    for (controller, model), data in results.items():
        name = controller + "_" + model
        np.savetxt(output / (name + ".csv"), np.column_stack(data), delimiter=",", comments="", header=CSV_HEADER)
        controller_params, plant_params = model_parameters(model)
        summary["cases"][name] = dict(controller_params=controller_params, plant_params=plant_params, samples=len(data[0]), **response_metrics(data))
    for filename, index, ylabel in (("angles.png", 1, "Angle [rad]"), ("errors.png", 3, "Target - angle [rad]"), ("torques.png", 2, "Torque [N m]"), ("integrals.png", 4, "Integral error [rad s]")):
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
        for col, model in enumerate(MODELS):
            for j in (0, 1):
                ax = axes[j, col]
                for controller in CONTROLLERS:
                    data = results[controller, model]
                    ax.plot(data[0], data[index][:, j], label=controller)
                if index in (1, 3):
                    ax.axhline(TARGET[j] if index == 1 else 0., color="black", linestyle="--", label="Target")
                if index == 4:
                    for bound in (-INTEGRAL_LIMIT, INTEGRAL_LIMIT):
                        ax.axhline(bound, color="gray", linestyle=":")
                ax.set_title(f"{model}, J{j + 1}; target={TARGET[j]} rad")
                ax.set_xlabel("Time [s]")
                ax.set_ylabel(ylabel)
                ax.grid(True)
                ax.legend()
        fig.tight_layout()
        fig.savefig(output / filename, dpi=150)
        plt.close(fig)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "step_8_9")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    results = {case: simulate(*case) for case in CASES}
    print(json.dumps(save_results(results, args.output), indent=2))
    print("Output:", args.output.resolve())


if __name__ == "__main__":
    main()
