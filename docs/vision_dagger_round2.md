# Vision DAgger Round 2

This experiment adds data and hard-example learning to the existing R1 policy.
It preserves the CNN/MLP architecture, R1 normalization, camera, physics and
closed-loop evaluation criteria. Existing BC/R1 scripts, checkpoints and data
are retained. Generated artifacts remain under the ignored `outputs/` directory,
except for the final checkpoint in the sibling `pytorch-learning/checkpoints/`.

## Run

From `robot-arm-simulator`, using Python with the existing `torch`, `numpy`,
`opencv-python` and `mujoco` dependencies:

```powershell
python scripts/collect_dagger_round2.py
python scripts/train_dagger_round2.py
python scripts/evaluate_vision_dagger_round2.py
python -m unittest discover -s tests -p test_vision_dagger_round2.py -v
```

The sibling learning project also has `step_13_b_9_train_dagger_round2.py`,
which forwards to the same trainer. Training itself does not import MuJoCo;
the simulator dependency is needed only for collection, evaluation and tests.

All three experiment commands reject existing output directories/checkpoints.
For another run, provide new `--output-dir` paths, `--r2-dir` to the trainer,
and matching `--checkpoint` / `--policy` paths. Collector `--episodes 1` and
evaluator `--smoke` are explicit smoke runs and are not the final benchmark.

The existing `.venv` launchers on this machine reference missing Python
installations. Execution in this session used the installed Python 3.12 runtime
with the simulator's existing site-packages; the original environments were not
modified:

```powershell
$env:PYTHONPATH = (Resolve-Path '.venv/Lib/site-packages').Path
$daggerPython = "$env:USERPROFILE/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe"
& $daggerPython -u scripts/collect_dagger_round2.py
& $daggerPython -u scripts/train_dagger_round2.py
& $daggerPython -u scripts/evaluate_vision_dagger_round2.py
```

## Data and weighting

- Original BC: 4,000 training rows, selected by `episode < 80` exactly as before.
  Episodes 80–99 (1,000 rows) are held out and are never used for fitting,
  weight calibration, normalization or checkpoint selection.
- Round 1: all 5,000 existing rows.
- Round 2: 50 episodes × 4 seconds × 25 saved frames/second = 5,000 rows.
  Seed is 789; R1 used 456. The R1 policy supplies every control action at
  the unchanged 500 Hz physics rate. Saving at 25 Hz does not change control rate.
- Images, state, student action and CTC label are captured before the same
  physics step. `tau1,tau2` are CTC labels. `student_tau1,student_tau2` are used
  only for diagnostics and weights; `action_error` is their L2 difference in Nm.
- BC loss weight is 1. DAgger weight is
  `1 + 2 * clip(action_error / max(p90, 1e-6), 0, 1)`, with p90 computed only
  from the R1/R2 training errors. Weights are in [1, 3]. Each batch uses the
  weighted mean of per-sample MSE in normalized torque coordinates. All training
  rows appear each epoch with ordinary shuffled sampling.
- The R1 checkpoint initializes the model. Its four normalization tensors are
  carried forward unchanged. Adam fine-tunes for a fixed 30 epochs, batch size
  64, initial learning rate 1e-4, cosine decay towards 1e-5, gradient norm cap 5,
  and training seed 2026. The final epoch is selected without consulting the
  held-out offline test or the closed-loop benchmark.

## Training range extension

Units are radians and radians/second. Positive goal q2 retains the original
elbow convention.

| Variable | Original/R1 range | R2 wider range |
|---|---|---|
| q_start[0] | [-0.7, 0.7] | [-0.85, 0.85] |
| q_start[1] | [-0.8, 0.8] | [-1.0, 1.0] |
| dq_start, each joint | [-0.3, 0.3] | [-0.45, 0.45] |
| q_goal[0] | [-0.8, 0.8] | [-0.95, 0.95] |
| q_goal[1] | [0.2, 1.2] | [0.15, 1.35] |

The 50 episodes contain 10 original-range episodes, 20 wider uniform episodes,
10 short moves with `norm(q_goal - q_start)` in [0.15, 0.75], and 10 long moves
in [1.6, 2.6]. Short/long moves use rejection sampling within the same wider
bounds. `manifest.json` records every actual initial state and goal, joint-space
displacement, tip displacement, source checkpoint hash and model hash.

## Fixed evaluation

The evaluation keeps `np.random.default_rng(seed=123)`, the original draw order
and ranges, all 50 episodes, 4 seconds/episode, and timestep 0.002 seconds.
The renderer remains 640×480 with lookat [0.5, 0, 0], distance 4, azimuth 90,
elevation -90. RGB is resized with `INTER_AREA` to 80×60, converted to float32
CHW and divided by 255, exactly as in training. The model uses `eval()` and
`torch.no_grad()`.

Success means final tip error ≤0.03 m **and** absolute final speed ≤0.05 rad/s
for **both** joints. Action RMSE is first computed over all 2,000 steps per
episode, then averaged over episodes per joint. Torque peaks are absolute
maxima per joint across all steps and episodes. Failure IDs are zero-based.

Evaluation artifacts are `episodes.csv`, `episodes.json`, `manifest.json` and
`summary.json` under `outputs/step_13_b_10_dagger_round2_eval/`. The historical
BC 50% and R1 76% baselines are labeled historical; the new evaluation never
overwrites the existing `step_13_b_4` results.

## Verification

Ten regression tests check the exact old network forward pass, CTC and forward
kinematics, preprocessing and checkpoint inference, all 50 old random draws,
success boundaries, original train/test isolation, bounded loss weights,
teacher label selection, wider training strata, and student-only actuation.
The collector's test observes every applied torque during a mocked full episode.

The combined experiment cannot separately attribute any improvement to data
quantity, weighting or range extension. Those require independent ablations;
the fixed evaluation is not used to tune this run.

## Executed results (2026-11-14)

Collection produced 5,000 samples. All image files exist and all 5,000 stored
expert labels match an independent recomputation of CTC exactly. The target ball
was visible in the unchanged camera in all 50 initial states (minimum 360
segmentation pixels). The R1 student's success rate on the **wider training**
rollouts was 60%; this is not comparable to the fixed generalization benchmark.

The combined DAgger error p90 used for weights was 4.1929975129 Nm. Mean weights
were 1.5061 for R1 and 1.5974 for R2; BC remained at 1. The fixed 30-epoch run
used all 14,000 training samples. Final weighted normalized MSE was 0.00648887.
The offline held-out torque RMSE was [0.677881, 0.214223] Nm.

The saved model is `pytorch-learning/checkpoints/vision_dagger_r2_policy.pt` in
the sibling project. SHA-256:
`5ac66c51195610af10abcb125f66a997d1b73c74c017490ee1224b0c50679ea6`.
The architecture still has 619,234 parameters. Normalization tensors are exactly
equal to R1. Hash checks confirm that both BC and R1 checkpoints are unchanged.

Original Vision BC's historical success rate is 50%. R1 values below come from
the existing `outputs/step_13_b_4/vision_bc_generalization.csv`; R2 is the new full
seed-123 evaluation. R1's first two episodes were also rerun as a parity smoke
check, with matching success decisions and nearly identical errors.

| Metric | DAgger R1 (existing result) | DAgger R2 |
|---|---:|---:|
| Success | 38/50 (76%) | 36/50 (72%) |
| Mean final tip error, m | 0.031247 | 0.023852 |
| Median final tip error, m | 0.018403 | 0.021516 |
| 95th percentile final tip error, m | 0.106277 | 0.057354 |
| Mean action RMSE, Nm, joints 1/2 | [0.947564, 0.241158] | [0.570103, 0.172178] |
| Largest absolute policy torque, Nm, joints 1/2 | [90.630080, 21.753124] | [85.303909, 20.807884] |
| Largest absolute expert torque, Nm, joints 1/2 | [77.528770, 18.986397] | [77.528770, 18.986397] |

R2 failed episode IDs (zero-based):
`[7, 8, 9, 12, 16, 21, 22, 25, 36, 38, 41, 42, 45, 48]`.
All 14 failures exceed the 3 cm position threshold; none violate the final
velocity threshold. Five previous failures recovered (`4,6,13,24,34`), while
seven previous successes regressed (`16,21,36,38,41,42,45`).

**The requested combined change did not improve success rate:** it decreased
by 4 percentage points. Action RMSE, mean tip error and tail error improved,
but median tip error increased. The R1 checkpoint is retained as the stronger
policy on the requested success criterion. No evaluation threshold or episode
was changed, and no further checkpoint was selected using these results.
Further tuning would need a separate validation distribution; this run does
not establish which of the three combined changes caused the regressions.

Machine-readable results:

- `outputs/step_13_b_8_dagger_round2/manifest.json`
- `outputs/step_13_b_8_dagger_round2/camera_visibility_check.json`
- `outputs/step_13_b_9_train_dagger_round2/summary.json`
- `outputs/step_13_b_10_dagger_round2_eval/summary.json`
- `outputs/step_13_b_10_dagger_round2_eval/comparison.json`

All ten regression tests passed on the final implementation. The existing venv
launcher issue remains; the alternate Python invocation above successfully ran
collection, training and evaluation without altering the user's venv files.
