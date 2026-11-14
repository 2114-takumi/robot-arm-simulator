"""Collect R1-student states with CTC labels; never use CTC as control."""
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

# Training only: modest extension, retaining the unique positive-elbow goal.
TRAINING_RANGES = {
    "q_start_low": [-0.85, -1.0], "q_start_high": [0.85, 1.0],
    "dq_start_low": -0.45, "dq_start_high": 0.45,
    "q_goal_low": [-0.95, 0.15], "q_goal_high": [0.95, 1.35],
    "mixture": "20% original, 40% wider uniform, 20% short moves, 20% long moves",
    "short_move_joint_norm": [0.15, 0.75],
    "long_move_joint_norm": [1.6, 2.6],
}


def sample_training_episode(rng, episode):
    # Deterministic strata ensure short AND long moves, without using test cases.
    stratum = episode % 5
    if stratum == 0:
        return (*sample_evaluation_episode(rng), "original_range")
    for _ in range(10000):
        q = rng.uniform(TRAINING_RANGES["q_start_low"], TRAINING_RANGES["q_start_high"])
        dq = rng.uniform(-0.45, 0.45, size=2)
        goal = rng.uniform(TRAINING_RANGES["q_goal_low"], TRAINING_RANGES["q_goal_high"])
        distance = np.linalg.norm(goal - q)
        if stratum in (1, 2):
            return q, dq, goal, "wider_uniform"
        if stratum == 3 and 0.15 <= distance <= 0.75:
            return q, dq, goal, "short_move"
        if stratum == 4 and 1.6 <= distance <= 2.6:
            return q, dq, goal, "long_move"
    raise RuntimeError("Could not sample a training episode")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path,
                        default=PYTORCH_ROOT / "checkpoints/vision_dagger_r1_policy.pt")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "outputs/step_13_b_8_dagger_round2")
    parser.add_argument("--seed", type=int, default=789)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.seed in (0, 123, 456):
        parser.error("Use a fresh training seed, distinct from BC, evaluation and R1")
    if args.episodes < 1 or args.threads < 1:
        parser.error("episodes and threads must be positive")
    if args.output_dir.exists():
        raise FileExistsError(f"Preserving existing data: {args.output_dir}")
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(1)
    policy, checkpoint = load_policy(args.policy)
    model, data, renderer, camera = make_simulation()
    dt, duration, save_every = model.opt.timestep, 4.0, 20
    steps = int(duration / dt)
    rng = np.random.default_rng(args.seed)
    rows, episodes = [], []
    args.output_dir.mkdir(parents=True, exist_ok=False)
    image_dir = args.output_dir / "images"
    image_dir.mkdir()
    metadata = {
        "status": "collecting", "seed": args.seed, "episodes": args.episodes,
        "student_checkpoint": str(args.policy.resolve()), "student_sha256": sha256(args.policy),
        "model_sha256": sha256(MODEL_PATH), "timestep": dt, "duration": duration,
        "save_every": save_every, "training_ranges": TRAINING_RANGES,
        "control": "student_tau only at every physics step; no expert mixing or torque clipping",
        "labels": "tau1,tau2 = CTC(q,dq,q_goal); student_tau columns are diagnostics only",
        "camera": {"render_size": [640, 480], "image_size": [80, 60],
                   "lookat": [0.5, 0, 0], "distance": 4, "azimuth": 90, "elevation": -90},
    }
    manifest = args.output_dir / "manifest.json"
    manifest.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    header = ["episode", "time", "filename", "q1", "q2", "dq1", "dq2", "tau1", "tau2",
              "ball_x", "ball_y", "goal_q1", "goal_q2", "student_tau1", "student_tau2", "action_error"]
    start = time.perf_counter()
    try:
        with (args.output_dir / "labels.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            for episode in range(args.episodes):
                q_start, dq_start, q_goal, stratum = sample_training_episode(rng, episode)
                ball = forward_kinematics(q_goal)
                mujoco.mj_resetData(model, data)
                data.mocap_pos[0] = [ball[0], ball[1], 0.08]
                data.qpos[:], data.qvel[:] = q_start, dq_start
                mujoco.mj_forward(model, data)
                for step in range(steps):
                    q, dq = data.qpos.copy(), data.qvel.copy()
                    renderer.update_scene(data, camera=camera)
                    rgb = renderer.render()
                    student_tau = policy_action(policy, checkpoint, rgb, q, dq)
                    if not np.isfinite(np.r_[q, dq, student_tau]).all():
                        raise FloatingPointError(f"Non-finite student state/action: episode {episode}, step {step}")
                    if step % save_every == 0:
                        expert_tau = expert_ctc(q, dq, q_goal)
                        error = float(np.linalg.norm(student_tau - expert_tau))
                        if not np.isfinite(error):
                            raise FloatingPointError("Non-finite expert label")
                        filename = f"dagger_r2_{len(rows):06d}.png"
                        small = cv2.resize(rgb, (80, 60), interpolation=cv2.INTER_AREA)
                        if not cv2.imwrite(str(image_dir / filename), cv2.cvtColor(small, cv2.COLOR_RGB2BGR)):
                            raise OSError(f"Could not write {filename}")
                        row = [episode, step * dt, filename, *q, *dq, *expert_tau,
                               *ball, *q_goal, *student_tau, error]
                        writer.writerow(row)
                        rows.append(row)
                    # The expert label above is deliberately never applied to MuJoCo.
                    data.qfrc_applied[:] = student_tau
                    mujoco.mj_step(model, data)
                tip_error = float(np.linalg.norm(forward_kinematics(data.qpos) - ball))
                success = episode_success(tip_error, data.qvel)
                episodes.append({"episode": episode, "q_start": q_start.tolist(),
                                 "dq_start": dq_start.tolist(), "q_goal": q_goal.tolist(),
                                 "stratum": stratum, "move_joint_norm": float(np.linalg.norm(q_goal - q_start)),
                                 "move_tip_distance": float(np.linalg.norm(ball - forward_kinematics(q_start))),
                                 "final_tip_error": tip_error, "success": success})
                stream.flush()
                print(f"episode={episode:02d} stratum={stratum} success={success} "
                      f"tip_error={tip_error:.5f} samples={len(rows)} elapsed={time.perf_counter()-start:.1f}s", flush=True)
        errors = np.array([row[-1] for row in rows])
        metadata.update(status="complete", samples=len(rows),
                        student_success_rate=float(np.mean([e["success"] for e in episodes])),
                        action_error_percentiles=dict(zip(["p50", "p90", "p95", "max"],
                                                           np.percentile(errors, [50, 90, 95, 100]).tolist())),
                        labels_sha256=sha256(args.output_dir / "labels.csv"),
                        elapsed_seconds=time.perf_counter() - start, episode_settings=episodes)
    finally:
        renderer.close()
        manifest.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Collected {len(rows)} CTC-labelled samples in {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
