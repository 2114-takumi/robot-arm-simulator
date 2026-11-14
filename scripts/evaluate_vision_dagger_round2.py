"""The original seed=123 / 50-episode benchmark, with separate R2 outputs."""
import argparse
import csv
import json
from pathlib import Path
import time

import cv2
import mujoco
import numpy as np
import torch

from vision_dagger_common import (
    PROJECT_ROOT, PYTORCH_ROOT, MODEL_PATH, episode_success, expert_ctc,
    forward_kinematics, load_policy, make_simulation, policy_action,
    sample_evaluation_episode, sha256,
)


def summarize(results):
    errors = np.array([r["tip_error"] for r in results])
    def pairs(prefix):
        return np.array([[r[prefix + "_1"], r[prefix + "_2"]] for r in results])
    return {
        "episodes": len(results), "success_count": sum(r["success"] for r in results),
        "success_rate": float(np.mean([r["success"] for r in results])),
        "mean_final_tip_error_m": float(errors.mean()),
        "median_final_tip_error_m": float(np.median(errors)),
        "p95_final_tip_error_m": float(np.percentile(errors, 95)),
        "mean_action_rmse_nm": pairs("action_rmse").mean(axis=0).tolist(),
        "largest_policy_torque_nm": pairs("max_policy_tau").max(axis=0).tolist(),
        "largest_expert_torque_nm": pairs("max_expert_tau").max(axis=0).tolist(),
        "failed_episode_ids": [r["episode"] for r in results if not r["success"]],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path,
                        default=PYTORCH_ROOT / "checkpoints/vision_dagger_r2_policy.pt")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "outputs/step_13_b_10_dagger_round2_eval")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--smoke", action="store_true", help="Two episodes only; not a comparable benchmark")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    if args.output_dir.exists():
        raise FileExistsError(f"Preserving existing evaluation: {args.output_dir}")
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(1)
    policy, checkpoint = load_policy(args.policy)
    policy.eval()
    model, data, renderer, camera = make_simulation()
    dt, duration = model.opt.timestep, 4.0
    steps = int(duration / dt)
    rng = np.random.default_rng(seed=123)
    results, episodes = [], []
    args.output_dir.mkdir(parents=True, exist_ok=False)
    metadata = {"status": "running", "seed": 123, "episodes": 2 if args.smoke else 50,
                "benchmark_comparable": not args.smoke,
                "timestep": dt, "duration": duration,
                "tip_threshold": 0.03, "velocity_threshold": 0.05,
                "checkpoint": str(args.policy.resolve()), "checkpoint_sha256": sha256(args.policy),
                "model_sha256": sha256(MODEL_PATH), "evaluator_sha256": sha256(__file__),
                "control": "student only, policy recomputed each physics step",
                "historical_baselines": {"original_vision_bc": 0.50, "dagger_r1": 0.76},
                "baseline_source": "User-reported earlier seed=123, 50-episode evaluation; not rerun here"}
    (args.output_dir / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    start = time.perf_counter()
    try:
        with (args.output_dir / "episodes.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = None
            for episode in range(metadata["episodes"]):
                q_start, dq_start, q_goal = sample_evaluation_episode(rng)
                episodes.append({"episode": episode, "q_start": q_start.tolist(),
                                 "dq_start": dq_start.tolist(), "q_goal": q_goal.tolist()})
                ball = forward_kinematics(q_goal)
                mujoco.mj_resetData(model, data)
                data.mocap_pos[0] = [ball[0], ball[1], 0.08]
                data.qpos[:], data.qvel[:] = q_start, dq_start
                mujoco.mj_forward(model, data)
                policy_taus, expert_taus = [], []
                with torch.no_grad():
                    for step in range(steps):
                        q, dq = data.qpos.copy(), data.qvel.copy()
                        renderer.update_scene(data, camera=camera)
                        rgb = renderer.render()
                        tau = policy_action(policy, checkpoint, rgb, q, dq)
                        expert = expert_ctc(q, dq, q_goal)
                        if not np.isfinite(np.r_[q, dq, tau, expert]).all():
                            raise FloatingPointError(f"Non-finite state/action at episode {episode}, step {step}")
                        policy_taus.append(tau.copy())
                        expert_taus.append(expert.copy())
                        data.qfrc_applied[:] = tau
                        mujoco.mj_step(model, data)
                final_q, final_dq = data.qpos.copy(), data.qvel.copy()
                tip_error = float(np.linalg.norm(forward_kinematics(final_q) - ball))
                if not np.isfinite(np.r_[final_q, final_dq, tip_error]).all():
                    raise FloatingPointError(f"Non-finite final state in episode {episode}")
                policy_taus, expert_taus = np.array(policy_taus), np.array(expert_taus)
                error = policy_taus - expert_taus
                result = {"episode": episode, "tip_error": tip_error}
                vectors = {"joint_error": final_q - q_goal, "final_dq": final_dq,
                           "action_rmse": np.sqrt(np.mean(error**2, axis=0)),
                           "max_action_error": np.max(np.abs(error), axis=0),
                           "max_policy_tau": np.max(np.abs(policy_taus), axis=0),
                           "max_expert_tau": np.max(np.abs(expert_taus), axis=0)}
                for name, values in vectors.items():
                    result.update({f"{name}_{i + 1}": float(value) for i, value in enumerate(values)})
                result["success"] = int(episode_success(tip_error, final_dq))
                results.append(result)
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(result))
                    writer.writeheader()
                writer.writerow(result)
                stream.flush()
                print(f"episode={episode:02d} success={result['success']} tip_error={tip_error:.6f} "
                      f"action_rmse={vectors['action_rmse']} elapsed={time.perf_counter()-start:.1f}s", flush=True)
    finally:
        renderer.close()
        (args.output_dir / "episodes.json").write_text(json.dumps(episodes, indent=2), encoding="utf-8")
    summary = {**metadata, **summarize(results), "status": "complete",
               "elapsed_seconds": time.perf_counter() - start}
    if not args.smoke:
        summary["success_rate_change_vs_r1"] = summary["success_rate"] - 0.76
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n" + ("SMOKE ONLY (not benchmark-comparable)" if args.smoke else "Fixed 50-episode generalization"))
    for label, key in [("Success rate", "success_rate"), ("Mean final tip error [m]", "mean_final_tip_error_m"),
                       ("Median final tip error [m]", "median_final_tip_error_m"),
                       ("95th percentile final tip error [m]", "p95_final_tip_error_m"),
                       ("Mean action RMSE [Policy vs CTC, Nm]", "mean_action_rmse_nm"),
                       ("Largest Policy torque [Nm, per joint]", "largest_policy_torque_nm"),
                       ("Largest Expert torque [Nm, per joint]", "largest_expert_torque_nm"),
                       ("Failed episode IDs (zero-based)", "failed_episode_ids")]:
        print(f"{label}: {summary[key]}")
    if not args.smoke:
        print("\nOriginal Vision BC (historical): 50%\nDAgger R1 (historical): 76%")
        print(f"{args.policy.stem} (this run): {summary['success_rate']:.0%}")
    print(f"Saved: {args.output_dir}")


if __name__ == "__main__":
    main()
